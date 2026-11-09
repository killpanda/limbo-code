//! `limbo channels login weixin`: show a QR code in the terminal, wait for
//! the phone to scan and confirm, and return the bot credentials. The
//! status values follow the reference plugin's `get_qrcode_status` loop.

use std::io::Write;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::api::{Api, DEFAULT_BASE_URL, Identity};
use crate::channels::SendError;
use crate::channels::store::WeixinCred;

/// `bot_type` of the QR code request.
const BOT_TYPE: &str = "3";
const MAX_REFRESH: u32 = 3;

/// Render `content` as a terminal QR code (two rows per character line).
pub fn qr_text(content: &str) -> Option<String> {
    let code = qrcode::QrCode::new(content.as_bytes()).ok()?;
    Some(
        code.render::<qrcode::render::unicode::Dense1x2>()
            .dark_color(qrcode::render::unicode::Dense1x2::Light)
            .light_color(qrcode::render::unicode::Dense1x2::Dark)
            .quiet_zone(true)
            .build(),
    )
}

async fn fetch_qr(api: &Api) -> anyhow::Result<(String, String)> {
    let r = api
        .post(
            &format!("ilink/bot/get_bot_qrcode?bot_type={BOT_TYPE}"),
            json!({"local_token_list": []}),
            Duration::from_secs(15),
        )
        .await?;
    let qrcode = r
        .get("qrcode")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("get_bot_qrcode returned no qrcode"))?;
    let content = r
        .get("qrcode_img_content")
        .and_then(Value::as_str)
        .unwrap_or("");
    Ok((qrcode.to_string(), content.to_string()))
}

fn show(content: &str) {
    println!();
    match qr_text(content) {
        Some(q) => println!("{q}"),
        None => println!("（无法在终端绘制二维码）"),
    }
    println!("用手机微信扫描上面的二维码；扫不了的话在微信里打开这个链接：\n{content}\n");
}

fn read_line(prompt: &str) -> String {
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let mut s = String::new();
    let _ = std::io::stdin().read_line(&mut s);
    s.trim().to_string()
}

pub async fn login(http: reqwest::Client, id: Identity) -> anyhow::Result<WeixinCred> {
    let mut api = Api {
        http,
        base_url: DEFAULT_BASE_URL.to_string(),
        token: String::new(),
        id,
    };
    let (mut qrcode, content) = fetch_qr(&api).await?;
    show(&content);
    let deadline = Instant::now() + Duration::from_secs(480);
    let mut refreshes = 0;
    let mut verify: Option<String> = None;
    let mut scanned = false;
    while Instant::now() < deadline {
        let mut q = format!("ilink/bot/get_qrcode_status?qrcode={}", urlencode(&qrcode));
        if let Some(v) = &verify {
            q.push_str(&format!("&verify_code={}", urlencode(v)));
        }
        let r = match api.get(&q, Duration::from_secs(35)).await {
            Ok(v) => v,
            Err(SendError::Transient(_)) => continue,
            Err(e) => anyhow::bail!("{e}"),
        };
        match r.get("status").and_then(Value::as_str).unwrap_or("wait") {
            "wait" => {}
            "scaned" => {
                verify = None;
                if !std::mem::replace(&mut scanned, true) {
                    println!("已扫码，请在手机上确认…");
                }
            }
            "need_verifycode" => {
                let p = if verify.is_some() {
                    "数字不匹配，请重新输入手机微信显示的数字："
                } else {
                    "输入手机微信显示的数字："
                };
                verify = Some(read_line(p));
            }
            "scaned_but_redirect" => {
                if let Some(h) = r.get("redirect_host").and_then(Value::as_str) {
                    api.base_url = format!("https://{h}");
                }
            }
            "expired" | "verify_code_blocked" => {
                refreshes += 1;
                if refreshes > MAX_REFRESH {
                    anyhow::bail!("二维码多次失效，请稍后再试");
                }
                println!("二维码已失效，正在刷新…");
                let (q2, c2) = fetch_qr(&api).await?;
                qrcode = q2;
                verify = None;
                scanned = false;
                show(&c2);
            }
            "binded_redirect" => {
                anyhow::bail!("这个微信已经绑定过一个 bot；如需重新登录请先在微信里解绑")
            }
            "confirmed" => {
                let token = r.get("bot_token").and_then(Value::as_str).unwrap_or("");
                let bot_id = r.get("ilink_bot_id").and_then(Value::as_str).unwrap_or("");
                if token.is_empty() || bot_id.is_empty() {
                    anyhow::bail!("登录失败：服务器没有返回 bot_token / ilink_bot_id");
                }
                let base = r
                    .get("baseurl")
                    .and_then(Value::as_str)
                    .filter(|b| !b.is_empty())
                    .map(String::from)
                    .unwrap_or_else(|| api.base_url.clone());
                let owner = r
                    .get("ilink_user_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                return Ok(WeixinCred {
                    token: token.to_string(),
                    bot_id: bot_id.to_string(),
                    base_url: base,
                    owners: if owner.is_empty() {
                        vec![]
                    } else {
                        vec![owner]
                    },
                });
            }
            other => eprintln!("未知的登录状态：{other}"),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!("登录超时（8 分钟）")
}

fn urlencode(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn qr_renders() {
        let q = super::qr_text("https://example.com/x").unwrap();
        assert!(q.lines().count() > 10);
    }
}
