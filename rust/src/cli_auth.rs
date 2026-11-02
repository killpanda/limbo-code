//! `limbo auth …` / `limbo models` subcommands and the first-run login.
//!
//! Plain-terminal counterparts of the TUI model picker: `auth login` asks
//! for a provider and a key (not echoed), checks it against the provider
//! and stores it in `~/.limbo/auth.json`; `auth list` shows where every
//! provider's key comes from; `models` lists the catalog.

use std::io::{BufRead, Write};

use crate::auth;
use crate::config::Config;
use crate::llm::catalog::{self, Dialect, ModelSpec, ProviderSpec};

/// Display order of providers (unknown ones follow, by id).
const ORDER: [&str; 7] = [
    "kimi-coding",
    "moonshotai",
    "deepseek",
    "glm",
    "codex",
    "magpie",
    "custom",
];

/// One entry per provider in the catalog.
pub fn providers() -> Vec<ProviderSpec> {
    let mut out: Vec<ProviderSpec> = vec![];
    for spec in catalog::catalog().values() {
        if !out.iter().any(|p| p.id == spec.provider.id) {
            out.push(spec.provider.clone());
        }
    }
    out.sort_by_key(|p| {
        (
            ORDER.iter().position(|o| *o == p.id).unwrap_or(ORDER.len()),
            p.id,
        )
    });
    out
}

fn first_model(provider: &ProviderSpec) -> ModelSpec {
    catalog::catalog()
        .values()
        .find(|s| s.provider.id == provider.id)
        .cloned()
        .expect("every provider has a model")
}

/// Where the key for `spec` comes from, for display (None = nowhere).
pub fn api_key_source(spec: &ModelSpec, config: &Config) -> Option<String> {
    let p = spec.provider.id;
    if config.providers.get(p).is_some_and(|o| o.api_key.is_some()) {
        return Some(format!("config [providers.{p}]"));
    }
    if auth::get(p).is_some() {
        return Some("auth.json".into());
    }
    if config.llm.api_key.is_some() {
        return Some("config [llm]".into());
    }
    if let Some(env) = catalog::resolve_api_key_env(spec, config)
        && std::env::var(&env).is_ok_and(|v| !v.is_empty())
    {
        return Some(format!("${env}"));
    }
    spec.provider.default_api_key.map(|_| "默认 key".into())
}

fn context(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        let m = tokens as f64 / 1_000_000.0;
        format!("{}M", (m * 100.0).round() / 100.0)
    } else {
        format!("{}K", tokens / 1024)
    }
}

/// Read one line with terminal echo off (for keys).
fn read_secret(prompt: &str) -> std::io::Result<String> {
    print!("{prompt}");
    std::io::stdout().flush()?;
    let fd = libc::STDIN_FILENO;
    // SAFETY: plain termios calls on stdin; the old mode is restored below.
    let old = unsafe {
        let mut t: libc::termios = std::mem::zeroed();
        if libc::isatty(fd) == 1 && libc::tcgetattr(fd, &mut t) == 0 {
            let saved = t;
            t.c_lflag &= !libc::ECHO;
            libc::tcsetattr(fd, libc::TCSANOW, &t);
            Some(saved)
        } else {
            None
        }
    };
    let mut line = String::new();
    let r = std::io::stdin().lock().read_line(&mut line);
    if let Some(saved) = old {
        // SAFETY: restoring the mode saved above.
        unsafe { libc::tcsetattr(fd, libc::TCSANOW, &saved) };
        println!();
    }
    r?;
    Ok(line)
}

fn read_line(prompt: &str) -> std::io::Result<String> {
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    if std::io::stdin().lock().read_line(&mut line)? == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "EOF",
        ));
    }
    Ok(line.trim().to_string())
}

fn pick_provider() -> std::io::Result<Option<ProviderSpec>> {
    let options = providers();
    println!("选择提供商：");
    for (n, p) in options.iter().enumerate() {
        println!(
            "  {}. {}  ({})",
            n + 1,
            catalog::provider_display_name(p.id),
            p.id
        );
    }
    let answer = read_line("编号或 id：")?;
    let found = options
        .iter()
        .enumerate()
        .find(|(n, p)| answer == (n + 1).to_string() || answer == p.id)
        .map(|(_, p)| p.clone());
    if found.is_none() {
        println!("没有这个提供商：{answer:?}");
    }
    Ok(found)
}

#[derive(Debug, PartialEq)]
pub enum KeyCheck {
    Valid,
    Invalid(String),
    Unverified(String),
}

/// Cheap key check: list models on the endpoint. Only a definite 401/403
/// counts as a bad key, so offline logins still work.
pub async fn check_key(provider: &ProviderSpec, key: &str, base_url: Option<&str>) -> KeyCheck {
    let base = base_url.unwrap_or(provider.base_url).trim_end_matches('/');
    if base.is_empty() {
        return KeyCheck::Unverified("没有 base_url，跳过验证".into());
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .unwrap_or_default();
    let mut req;
    if provider.dialect == Dialect::AnthropicMessages {
        let url = if base.ends_with("/v1") {
            format!("{base}/models")
        } else {
            format!("{base}/v1/models")
        };
        req = client
            .get(url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01");
    } else {
        req = client.get(format!("{base}/models")).bearer_auth(key);
    }
    for (k, v) in &provider.headers {
        req = req.header(*k, *v);
    }
    match req.send().await {
        Err(e) => KeyCheck::Unverified(format!("无法连接：{e}")),
        Ok(r) if r.status().as_u16() == 401 || r.status().as_u16() == 403 => KeyCheck::Invalid(
            format!("服务端拒绝了这个 key（HTTP {}）", r.status().as_u16()),
        ),
        Ok(r) if r.status().is_success() => KeyCheck::Valid,
        Ok(r) => KeyCheck::Unverified(format!("HTTP {}，未能确认", r.status().as_u16())),
    }
}

/// `limbo auth login [provider]`.
pub async fn login(
    provider_id: Option<&str>,
    key: Option<String>,
    base_url: Option<&str>,
    verify: bool,
) -> i32 {
    let provider = match provider_id {
        None => match pick_provider() {
            Ok(Some(p)) => p,
            _ => return 1,
        },
        Some(id) => match providers().into_iter().find(|p| p.id == id) {
            Some(p) => p,
            None => {
                let known: Vec<_> = providers().iter().map(|p| p.id).collect();
                eprintln!("未知提供商 {id:?}（可选：{}）", known.join(", "));
                return 1;
            }
        },
    };
    let name = catalog::provider_display_name(provider.id);
    let key = match key {
        Some(k) => k,
        None => {
            let default = provider.default_api_key;
            let hint = default
                .map(|d| format!("，回车用默认 {d}"))
                .unwrap_or_default();
            match read_secret(&format!("{name} API key（输入不回显{hint}）：")) {
                Ok(k) if !k.trim().is_empty() => k,
                Ok(_) => default.unwrap_or("").to_string(),
                Err(_) => return 1,
            }
        }
    };
    let key = key.trim().to_string();
    if key.is_empty() {
        eprintln!("没有输入 key，已取消");
        return 1;
    }
    if verify {
        println!("验证中…");
        match check_key(&provider, &key, base_url).await {
            KeyCheck::Valid => println!("✓ key 有效"),
            KeyCheck::Unverified(d) => println!("✓ {d}"),
            KeyCheck::Invalid(d) => {
                println!("✗ {d}");
                return 1;
            }
        }
    }
    if let Err(e) = auth::save(provider.id, &key, base_url) {
        eprintln!("保存失败：{e}");
        return 1;
    }
    println!(
        "已保存 {name} 的 key {} → {}",
        auth::mask(&key),
        auth::auth_path().display()
    );
    0
}

/// `limbo auth list`.
pub fn list_logins(config: &Config) -> i32 {
    let stored = auth::load_all();
    for p in providers() {
        let spec = first_model(&p);
        let source = api_key_source(&spec, config);
        let env = catalog::resolve_api_key_env(&spec, config);
        let mark = if source.is_some() { "✓" } else { "·" };
        let name = catalog::provider_display_name(p.id);
        let mut line = format!("{mark} {name:<18} {:<12}", p.id);
        match source.as_deref() {
            Some("auth.json") => {
                line.push_str(&format!(" auth.json  {}", auth::mask(&stored[p.id].key)));
            }
            Some(s) => line.push_str(&format!(" {s}")),
            None => {
                let env = env.map(|e| format!(" 或 ${e}")).unwrap_or_default();
                line.push_str(&format!(" 未配置（limbo auth login {}{env}）", p.id));
            }
        }
        println!("{line}");
    }
    0
}

/// `limbo auth logout <provider>`.
pub fn logout(provider: &str) -> i32 {
    match auth::remove(provider) {
        Ok(true) => {
            println!(
                "已删除 {} 的已存 key",
                catalog::provider_display_name(provider)
            );
            0
        }
        Ok(false) => {
            println!("auth.json 里没有 {provider}");
            1
        }
        Err(e) => {
            eprintln!("删除失败：{e}");
            1
        }
    }
}

/// `limbo models [query]`.
pub fn list_models(config: &Config, query: Option<&str>) -> i32 {
    let current = &config.llm.model;
    for p in providers() {
        let specs: Vec<&ModelSpec> = catalog::catalog()
            .values()
            .filter(|s| s.provider.id == p.id)
            .filter(|s| query.is_none_or(|q| s.id.to_lowercase().contains(&q.to_lowercase())))
            .collect();
        let Some(first) = specs.first() else { continue };
        let ready = api_key_source(first, config).is_some();
        println!(
            "{}{}",
            catalog::provider_display_name(p.id),
            if ready { "" } else { "  （未配置 key）" }
        );
        for spec in specs {
            let levels: Vec<String> = catalog::thinking_choices(spec)
                .into_iter()
                .flatten()
                .collect();
            let mut parts = vec![context(spec.context_window)];
            if !levels.is_empty() {
                parts.push(format!("思考 {}", levels.join("/")));
            }
            if spec.vision {
                parts.push("图像".into());
            }
            let mark = if &spec.id == current { "●" } else { " " };
            println!("  {mark} {:<24} {}", spec.id, parts.join(" · "));
        }
    }
    0
}

/// No key anywhere and a real terminal: offer to log in instead of exiting.
pub async fn first_run_login(spec: &ModelSpec, config: &Config) -> bool {
    let name = catalog::provider_display_name(spec.provider.id);
    println!(
        "欢迎使用 limbo。还没有 {name} 的 API key（模型 {}）。",
        spec.id
    );
    println!("粘贴一个 key 就能开始，它会保存在 ~/.limbo/auth.json（仅自己可读）。");
    println!("想换提供商：Ctrl+C 退出后运行 limbo auth login，或用 -m 指定模型。\n");
    if login(Some(spec.provider.id), None, None, true).await != 0 {
        return false;
    }
    catalog::resolve_api_key(spec, config).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn providers_are_unique_and_ordered() {
        let ps = providers();
        let ids: Vec<_> = ps.iter().map(|p| p.id).collect();
        let mut dedup = ids.clone();
        dedup.dedup();
        assert_eq!(ids, dedup);
        assert!(!ids.is_empty());
    }

    #[test]
    fn context_labels() {
        assert_eq!(context(1_000_000), "1M");
        assert_eq!(context(131_072), "128K");
    }
}
