//! OS clipboard access through the native tools (wl-paste/wl-copy, xclip,
//! osascript/pbpaste/pbcopy). Reading classifies the clipboard as image >
//! files > text; writing copies text back (OSC 52 is ignored by some
//! terminals, notably macOS Terminal.app).

use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::config::limbo_home;

pub enum Clip {
    Image { data: Vec<u8>, ext: String },
    Files(Vec<PathBuf>),
    Text(String),
    Empty,
}

fn which(cmd: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|p| std::env::split_paths(&p).any(|d| d.join(cmd).is_file()))
}

fn run(args: &[&str]) -> Option<Vec<u8>> {
    if !which(args[0]) {
        return None;
    }
    let out = Command::new(args[0])
        .args(&args[1..])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    out.status.success().then_some(out.stdout)
}

fn run_text(args: &[&str]) -> Option<String> {
    run(args).map(|b| String::from_utf8_lossy(&b).into_owned())
}

fn paths_from_uri_list(list: &str) -> Vec<PathBuf> {
    list.lines()
        .map(str::trim)
        .filter(|l| l.starts_with("file://"))
        .filter_map(|l| url::Url::parse(l).ok()?.to_file_path().ok())
        .filter(|p| p.exists())
        .collect()
}

pub fn read() -> Clip {
    if cfg!(target_os = "macos") {
        return read_macos();
    }
    if std::env::var_os("WAYLAND_DISPLAY").is_some() && which("wl-paste") {
        let Some(types) = run_text(&["wl-paste", "--list-types"]) else {
            return Clip::Empty;
        };
        let types: Vec<&str> = types.split_whitespace().collect();
        if let Some(m) = types.iter().find(|t| t.starts_with("image/"))
            && let Some(data) = run(&["wl-paste", "--type", m]).filter(|d| !d.is_empty())
        {
            let ext = m[6..].split(';').next().unwrap_or("png").to_string();
            return Clip::Image { data, ext };
        }
        if types.contains(&"text/uri-list") {
            let files = paths_from_uri_list(
                &run_text(&["wl-paste", "--type", "text/uri-list"]).unwrap_or_default(),
            );
            if !files.is_empty() {
                return Clip::Files(files);
            }
        }
        return run_text(&["wl-paste", "--no-newline"])
            .filter(|t| !t.is_empty())
            .map_or(Clip::Empty, Clip::Text);
    }
    if which("xclip") {
        let Some(types) = run_text(&["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"])
        else {
            return Clip::Empty;
        };
        let types: Vec<&str> = types.split_whitespace().collect();
        if let Some(m) = types.iter().find(|t| t.starts_with("image/"))
            && let Some(data) =
                run(&["xclip", "-selection", "clipboard", "-t", m, "-o"]).filter(|d| !d.is_empty())
        {
            return Clip::Image {
                data,
                ext: m[6..].to_string(),
            };
        }
        if types.contains(&"text/uri-list") {
            let files = paths_from_uri_list(
                &run_text(&[
                    "xclip",
                    "-selection",
                    "clipboard",
                    "-t",
                    "text/uri-list",
                    "-o",
                ])
                .unwrap_or_default(),
            );
            if !files.is_empty() {
                return Clip::Files(files);
            }
        }
        return run_text(&["xclip", "-selection", "clipboard", "-o"])
            .filter(|t| !t.is_empty())
            .map_or(Clip::Empty, Clip::Text);
    }
    Clip::Empty
}

fn read_macos() -> Clip {
    let Some(info) = run_text(&["osascript", "-e", "clipboard info"]) else {
        return Clip::Empty;
    };
    if info.contains("«class PNGf»") || info.contains("«class TIFF»") {
        let out = run_text(&[
            "osascript",
            "-e",
            "set theData to the clipboard as «class PNGf»",
            "-e",
            "return theData",
        ])
        .unwrap_or_default();
        if let Some(hex) = out
            .trim()
            .strip_prefix("«data PNGf")
            .and_then(|s| s.strip_suffix('»'))
            && let Some(data) = decode_hex(hex)
        {
            return Clip::Image {
                data,
                ext: "png".into(),
            };
        }
    }
    if info.contains("«class furl»")
        && let Some(out) = run_text(&[
            "osascript",
            "-e",
            "set theFiles to the clipboard as «class furl»",
            "-e",
            "return POSIX path of theFiles",
        ])
    {
        let p = PathBuf::from(out.trim());
        if p.exists() {
            return Clip::Files(vec![p]);
        }
    }
    run_text(&["pbpaste"])
        .filter(|t| !t.is_empty())
        .map_or(Clip::Empty, Clip::Text)
}

fn decode_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

fn write_cmd(args: &[&str], text: &str) -> bool {
    use std::io::Write;
    if !which(args[0]) {
        return false;
    }
    let Ok(mut child) = Command::new(args[0])
        .args(&args[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(text.as_bytes());
    }
    child.wait().is_ok_and(|s| s.success())
}

/// Copy text to the OS clipboard; false when no native tool is available.
pub fn write_text(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    if cfg!(target_os = "macos") {
        return write_cmd(&["pbcopy"], text);
    }
    if std::env::var_os("WAYLAND_DISPLAY").is_some() && which("wl-copy") {
        return write_cmd(&["wl-copy"], text);
    }
    which("xclip") && write_cmd(&["xclip", "-selection", "clipboard"], text)
}

/// OSC 52: ask the terminal to set the clipboard (works over ssh).
pub fn osc52(text: &str) -> String {
    use base64::Engine;
    format!(
        "\x1b]52;c;{}\x07",
        base64::engine::general_purpose::STANDARD.encode(text)
    )
}

/// Persist clipboard image bytes under `~/.limbo/attachments`.
pub fn save_image(data: &[u8], ext: &str) -> std::io::Result<PathBuf> {
    let dir = limbo_home().join("attachments");
    std::fs::create_dir_all(&dir)?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = dir.join(format!("clip-{:012x}.{ext}", stamp & 0xffff_ffff_ffff));
    std::fs::write(&path, data)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_and_uris() {
        assert_eq!(
            decode_hex("89504e47").unwrap(),
            vec![0x89, b'P', b'N', b'G']
        );
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a b.txt");
        std::fs::write(&f, "x").unwrap();
        let uri = url::Url::from_file_path(&f).unwrap().to_string();
        assert_eq!(paths_from_uri_list(&format!("# c\n{uri}\n")), vec![f]);
        assert!(osc52("hi").starts_with("\x1b]52;c;aGk="));
    }
}
