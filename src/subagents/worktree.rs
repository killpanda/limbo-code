//! git worktree isolation for writer subagents.
//!
//! A task with `isolation = "worktree"` runs in
//! `<workdir>/.limbo/worktrees/<task>` on branch `limbo/<session>/<task>`,
//! created from the current HEAD. Its changes are committed on that branch
//! when it finishes; the main agent merges the branch back with
//! `task_control(action="merge")`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(cwd: &Path, args: &[&str]) -> std::io::Result<std::process::Output> {
    Command::new("git").args(args).current_dir(cwd).output()
}

pub fn is_git_repo(workdir: &Path) -> bool {
    git(workdir, &["rev-parse", "--is-inside-work-tree"]).is_ok_and(|o| o.status.success())
}

pub fn branch_name(session_id: &str, task_id: &str) -> String {
    let tail: String = session_id
        .chars()
        .rev()
        .take(12)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("limbo/{tail}/{task_id}")
}

/// Keep `.limbo/` out of the main checkout's `git status`.
fn exclude_worktrees(workdir: &Path) {
    let Ok(out) = git(workdir, &["rev-parse", "--git-path", "info/exclude"]) else {
        return;
    };
    if !out.status.success() {
        return;
    }
    let mut p = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
    if !p.is_absolute() {
        p = workdir.join(p);
    }
    let cur = std::fs::read_to_string(&p).unwrap_or_default();
    if !cur.lines().any(|l| l == "/.limbo/") {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let prefix = if cur.is_empty() || cur.ends_with('\n') {
            ""
        } else {
            "\n"
        };
        let _ = std::fs::write(&p, format!("{cur}{prefix}/.limbo/\n"));
    }
}

/// `(path, branch)` of the task's worktree (created when missing).
pub fn create(
    workdir: &Path,
    session_id: &str,
    task_id: &str,
) -> anyhow::Result<(PathBuf, String)> {
    let path = workdir.join(".limbo").join("worktrees").join(task_id);
    let branch = branch_name(session_id, task_id);
    if path.is_dir() {
        return Ok((path, branch));
    }
    exclude_worktrees(workdir);
    std::fs::create_dir_all(path.parent().unwrap())?;
    let p = path.display().to_string();
    let mut out = git(workdir, &["worktree", "add", "-b", &branch, &p, "HEAD"])?;
    if !out.status.success() {
        // The branch may survive from an earlier attempt: reuse it.
        out = git(workdir, &["worktree", "add", &p, &branch])?;
    }
    if !out.status.success() {
        anyhow::bail!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok((path, branch))
}

/// Commit everything in the worktree; returns the `git diff --stat` text.
pub fn commit_all(path: &Path, message: &str) -> anyhow::Result<String> {
    git(path, &["add", "-A"])?;
    let stat = String::from_utf8_lossy(&git(path, &["diff", "--cached", "--stat"])?.stdout)
        .trim()
        .to_string();
    if !stat.is_empty() {
        let out = git(
            path,
            &[
                "-c",
                "user.name=limbo",
                "-c",
                "user.email=limbo@localhost",
                "commit",
                "-m",
                message,
            ],
        )?;
        if !out.status.success() {
            anyhow::bail!(
                "git commit failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
    }
    Ok(stat)
}

/// Merge `branch` into the workdir's current branch; conflicts are aborted.
pub fn merge(workdir: &Path, branch: &str) -> (bool, String) {
    match git(workdir, &["merge", "--no-edit", branch]) {
        Ok(o) if o.status.success() => (
            true,
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .last()
                .unwrap_or("")
                .to_string(),
        ),
        Ok(o) => {
            let _ = git(workdir, &["merge", "--abort"]);
            (
                false,
                String::from_utf8_lossy(&o.stderr).trim().to_string()
                    + &String::from_utf8_lossy(&o.stdout),
            )
        }
        Err(e) => (false, e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worktree_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let w = dir.path();
        let run = |args: &[&str]| {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(w)
                    .output()
                    .unwrap()
                    .status
                    .success(),
                "{args:?}"
            );
        };
        run(&["init", "-q", "-b", "main"]);
        std::fs::write(w.join("a.txt"), "1").unwrap();
        run(&["add", "."]);
        run(&[
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-qm",
            "init",
        ]);
        assert!(is_git_repo(w));
        let (path, branch) = create(w, "20261102-000000-abcdef", "t1").unwrap();
        std::fs::write(path.join("b.txt"), "2").unwrap();
        assert!(commit_all(&path, "work").unwrap().contains("b.txt"));
        let (ok, _) = merge(w, &branch);
        assert!(ok);
        assert!(w.join("b.txt").exists());
    }
}
