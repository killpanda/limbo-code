//! Keep `limbo gateway` running in the background: a launchd agent on
//! macOS (`~/Library/LaunchAgents/dev.limbo.gateway.plist`) or a systemd
//! user unit on Linux (`~/.config/systemd/user/limbo-gateway.service`).
//!
//! Both start at login, restart the gateway when it exits and log to
//! `~/.limbo/channels/gateway.log`. The service inherits this shell's
//! `PATH` (the agent's tools need `rg`, `python3`, …; launchd's default
//! PATH has neither) and proxy variables.
//!
//! 🦀 Rust notes
//! - `cfg!(target_os = "macos")` is a compile-time constant in an ordinary
//!   `if`: both branches are type-checked on every platform, so the macOS
//!   path is compiled (and its file generation unit-tested) on Linux CI too.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::channels::store::channels_dir;

pub const LAUNCHD_LABEL: &str = "dev.limbo.gateway";
pub const SYSTEMD_UNIT: &str = "limbo-gateway.service";

/// Environment variables carried into the service.
const KEEP_ENV: [&str; 9] = [
    "PATH",
    "LANG",
    "LIMBO_HOME",
    "LIMBO_AUTH_FILE",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
];

fn kept_env() -> Vec<(String, String)> {
    KEEP_ENV
        .iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| (k.to_string(), v)))
        .collect()
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The launchd property list.
pub fn launchd_plist(exe: &Path, args: &[String], log: &Path, env: &[(String, String)]) -> String {
    let mut program = format!(
        "    <string>{}</string>\n",
        xml_escape(&exe.display().to_string())
    );
    for a in args {
        program.push_str(&format!("    <string>{}</string>\n", xml_escape(a)));
    }
    let mut envs = String::new();
    for (k, v) in env {
        envs.push_str(&format!(
            "    <key>{}</key>\n    <string>{}</string>\n",
            xml_escape(k),
            xml_escape(v)
        ));
    }
    let log = xml_escape(&log.display().to_string());
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{program}  </array>
  <key>EnvironmentVariables</key>
  <dict>
{envs}  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>10</integer>
  <key>ProcessType</key>
  <string>Background</string>
  <key>StandardOutPath</key>
  <string>{log}</string>
  <key>StandardErrorPath</key>
  <string>{log}</string>
</dict>
</plist>
"#
    )
}

/// The systemd user unit.
pub fn systemd_unit(exe: &Path, args: &[String], log: &Path, env: &[(String, String)]) -> String {
    let quote = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    let mut cmd = quote(&exe.display().to_string());
    for a in args {
        cmd.push(' ');
        cmd.push_str(&quote(a));
    }
    let envs: String = env
        .iter()
        .map(|(k, v)| format!("Environment={}\n", quote(&format!("{k}={v}"))))
        .collect();
    format!(
        "[Unit]\nDescription=limbo chat gateway (Telegram / WeChat)\nAfter=network-online.target\nWants=network-online.target\n\n\
[Service]\nExecStart={cmd}\n{envs}Restart=on-failure\nRestartSec=10\nStandardOutput=append:{log}\nStandardError=append:{log}\n\n\
[Install]\nWantedBy=default.target\n",
        log = log.display()
    )
}

fn home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

pub fn service_path() -> PathBuf {
    if cfg!(target_os = "macos") {
        home()
            .join("Library/LaunchAgents")
            .join(format!("{LAUNCHD_LABEL}.plist"))
    } else {
        home().join(".config/systemd/user").join(SYSTEMD_UNIT)
    }
}

pub fn log_path() -> PathBuf {
    channels_dir().join("gateway.log")
}

pub fn describe() -> String {
    let p = service_path();
    if p.exists() {
        format!("已安装（{}）· 日志 {}", p.display(), log_path().display())
    } else {
        "未安装（`limbo gateway install` 开机自启）".into()
    }
}

fn run(cmd: &mut Command) -> bool {
    match cmd.status() {
        Ok(s) => s.success(),
        Err(e) => {
            eprintln!("warning: {cmd:?}: {e}");
            false
        }
    }
}

fn uid() -> u32 {
    // SAFETY: getuid has no failure mode and touches no memory.
    unsafe { libc::getuid() }
}

/// Write and start the service. `extra_args`: global flags to keep
/// (`--config`, `--session-dir`).
pub fn install(extra_args: &[String]) -> anyhow::Result<i32> {
    let exe = std::env::current_exe()?;
    let mut args: Vec<String> = extra_args.to_vec();
    args.push("gateway".into());
    let log = log_path();
    std::fs::create_dir_all(channels_dir())?;
    let path = service_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let env = kept_env();
    if cfg!(target_os = "macos") {
        let target = format!("gui/{}", uid());
        // Replace a loaded older copy.
        let _ = Command::new("launchctl")
            .args(["bootout", &format!("{target}/{LAUNCHD_LABEL}")])
            .stderr(std::process::Stdio::null())
            .status();
        std::fs::write(&path, launchd_plist(&exe, &args, &log, &env))?;
        let p = path.display().to_string();
        if !run(Command::new("launchctl").args(["bootstrap", &target, &p])) {
            // Older macOS releases.
            run(Command::new("launchctl").args(["load", "-w", &p]));
        }
        println!("✓ 已安装 launchd 服务 {LAUNCHD_LABEL}：{}", path.display());
    } else {
        std::fs::write(&path, systemd_unit(&exe, &args, &log, &env))?;
        run(Command::new("systemctl").args(["--user", "daemon-reload"]));
        run(Command::new("systemctl").args(["--user", "enable", "--now", SYSTEMD_UNIT]));
        println!(
            "✓ 已安装 systemd 用户服务 {SYSTEMD_UNIT}：{}",
            path.display()
        );
        println!("  注销后也保持运行：loginctl enable-linger $USER");
    }
    println!("  日志：{}", log.display());
    Ok(0)
}

pub fn uninstall() -> anyhow::Result<i32> {
    let path = service_path();
    if cfg!(target_os = "macos") {
        let target = format!("gui/{}", uid());
        if !run(Command::new("launchctl").args(["bootout", &format!("{target}/{LAUNCHD_LABEL}")])) {
            run(Command::new("launchctl").args(["unload", "-w", &path.display().to_string()]));
        }
    } else {
        run(Command::new("systemctl").args(["--user", "disable", "--now", SYSTEMD_UNIT]));
    }
    if path.exists() {
        std::fs::remove_file(&path)?;
        println!("✓ 已移除 {}", path.display());
    } else {
        println!("服务未安装");
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plist_is_complete_and_escaped() {
        let p = launchd_plist(
            Path::new("/opt/limbo & co/limbo"),
            &["gateway".into()],
            Path::new("/Users/me/.limbo/channels/gateway.log"),
            &[("PATH".into(), "/opt/homebrew/bin:/usr/bin".into())],
        );
        assert!(p.contains("<string>dev.limbo.gateway</string>"));
        assert!(
            p.contains("<string>/opt/limbo &amp; co/limbo</string>\n    <string>gateway</string>")
        );
        assert!(p.contains("<key>PATH</key>\n    <string>/opt/homebrew/bin:/usr/bin</string>"));
        assert!(p.contains("<key>RunAtLoad</key>\n  <true/>"));
        assert_eq!(p.matches("<dict>").count(), p.matches("</dict>").count());
    }

    #[test]
    fn systemd_unit_quotes_arguments() {
        let u = systemd_unit(
            Path::new("/usr/local/bin/limbo"),
            &["--config".into(), "/a b/c.toml".into(), "gateway".into()],
            Path::new("/h/.limbo/channels/gateway.log"),
            &[("PATH".into(), "/usr/bin".into())],
        );
        assert!(u.contains(
            "ExecStart=\"/usr/local/bin/limbo\" \"--config\" \"/a b/c.toml\" \"gateway\""
        ));
        assert!(u.contains("Environment=\"PATH=/usr/bin\""));
        assert!(u.contains("Restart=on-failure"));
    }
}
