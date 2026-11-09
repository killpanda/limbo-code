//! WeChat media lives on Tencent's CDN, encrypted with AES-128-ECB
//! (PKCS#7 padding) under a per-file random key: uploads encrypt before
//! `POST`, downloads decrypt after `GET`.
//!
//! 🦀 Rust notes
//! - The `aes` crate only provides the block cipher; ECB "mode" is just
//!   encrypting every 16-byte block on its own, done here by hand.
//! - `GenericArray::from_mut_slice` views a `[u8]` chunk as the fixed-size
//!   block type the cipher wants, without copying.

use aes::Aes128;
use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockDecrypt, BlockEncrypt, KeyInit};
use base64::Engine;
use md5::{Digest, Md5};
use serde_json::{Value, json};

use super::api::{Api, CDN_BASE_URL};

pub fn encrypt(plain: &[u8], key: &[u8; 16]) -> Vec<u8> {
    let cipher = Aes128::new(GenericArray::from_slice(key));
    let pad = 16 - plain.len() % 16;
    let mut buf = Vec::with_capacity(plain.len() + pad);
    buf.extend_from_slice(plain);
    buf.extend(std::iter::repeat_n(pad as u8, pad));
    for block in buf.chunks_mut(16) {
        cipher.encrypt_block(GenericArray::from_mut_slice(block));
    }
    buf
}

pub fn decrypt(data: &[u8], key: &[u8; 16]) -> anyhow::Result<Vec<u8>> {
    if data.is_empty() || !data.len().is_multiple_of(16) {
        anyhow::bail!("ciphertext length {} is not a multiple of 16", data.len());
    }
    let cipher = Aes128::new(GenericArray::from_slice(key));
    let mut buf = data.to_vec();
    for block in buf.chunks_mut(16) {
        cipher.decrypt_block(GenericArray::from_mut_slice(block));
    }
    let pad = *buf.last().unwrap_or(&0) as usize;
    if pad == 0
        || pad > 16
        || buf.len() < pad
        || !buf[buf.len() - pad..].iter().all(|b| *b as usize == pad)
    {
        anyhow::bail!("bad padding (wrong key?)");
    }
    buf.truncate(buf.len() - pad);
    Ok(buf)
}

/// Ciphertext size for `n` plaintext bytes.
#[cfg(test)]
pub fn padded_size(n: usize) -> usize {
    (n / 16 + 1) * 16
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

/// A media key as the API sends it: base64 of 16 raw bytes, base64 of a
/// 32-char hex string, or (image_item.aeskey) the hex string itself.
pub fn parse_key(s: &str) -> Option<[u8; 16]> {
    let raw = if s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        unhex(s)?
    } else {
        let d = base64::engine::general_purpose::STANDARD.decode(s).ok()?;
        if d.len() == 32 && d.iter().all(u8::is_ascii_hexdigit) {
            unhex(std::str::from_utf8(&d).ok()?)?
        } else {
            d
        }
    };
    raw.try_into().ok()
}

/// Download (and decrypt, with a key) one CDN media reference.
pub async fn download(
    http: &reqwest::Client,
    media: &Value,
    key: Option<[u8; 16]>,
    max_bytes: u64,
) -> anyhow::Result<Vec<u8>> {
    let url = match media.get("full_url").and_then(Value::as_str) {
        Some(u) if !u.is_empty() => u.to_string(),
        _ => {
            let q = media
                .get("encrypt_query_param")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("media has no download parameters"))?;
            let mut u = url::Url::parse(&format!("{CDN_BASE_URL}/download"))?;
            u.query_pairs_mut().append_pair("encrypted_query_param", q);
            u.to_string()
        }
    };
    let resp = http
        .get(&url)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("CDN download failed: HTTP {}", resp.status());
    }
    let bytes = resp.bytes().await?;
    if bytes.len() as u64 > max_bytes + 16 {
        anyhow::bail!("file too large ({} bytes)", bytes.len());
    }
    match key {
        Some(k) => decrypt(&bytes, &k),
        None => Ok(bytes.to_vec()),
    }
}

/// What a finished upload is referenced by in a message item.
pub struct Uploaded {
    pub download_param: String,
    pub aes_key: [u8; 16],
    pub raw_size: usize,
    pub cipher_size: usize,
}

impl Uploaded {
    /// The `media` object of an image / file / video item.
    pub fn media_json(&self) -> Value {
        json!({
            "encrypt_query_param": self.download_param,
            // The reference plugin sends base64 of the *hex* key.
            "aes_key": base64::engine::general_purpose::STANDARD.encode(hex(&self.aes_key)),
            "encrypt_type": 1,
        })
    }
}

/// Encrypt `plain` and upload it for `to_user`.
pub async fn upload(
    api: &Api,
    to_user: &str,
    media_type: i64,
    plain: &[u8],
) -> anyhow::Result<Uploaded> {
    let key: [u8; 16] = rand::random();
    let filekey = hex(&rand::random::<[u8; 16]>());
    let md5 = hex(&Md5::digest(plain));
    let cipher = encrypt(plain, &key);
    let req = json!({
        "filekey": filekey,
        "media_type": media_type,
        "to_user_id": to_user,
        "rawsize": plain.len(),
        "rawfilemd5": md5,
        "filesize": cipher.len(),
        "no_need_thumb": true,
        "aeskey": hex(&key),
    });
    let r = api.get_upload_url(req).await?;
    let url = match r
        .get("upload_full_url")
        .and_then(Value::as_str)
        .map(str::trim)
    {
        Some(u) if !u.is_empty() => u.to_string(),
        _ => {
            let p = r
                .get("upload_param")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("getuploadurl returned no upload URL"))?;
            let mut u = url::Url::parse(&format!("{CDN_BASE_URL}/upload"))?;
            u.query_pairs_mut()
                .append_pair("encrypted_query_param", p)
                .append_pair("filekey", &filekey);
            u.to_string()
        }
    };
    let mut last = String::new();
    for attempt in 1..=3 {
        let resp = api
            .http
            .post(&url)
            .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
            .timeout(std::time::Duration::from_secs(300))
            .body(cipher.clone())
            .send()
            .await;
        match resp {
            Ok(r) if r.status().is_success() => {
                let param = r
                    .headers()
                    .get("x-encrypted-param")
                    .and_then(|h| h.to_str().ok())
                    .map(String::from)
                    .ok_or_else(|| anyhow::anyhow!("CDN upload: no x-encrypted-param header"))?;
                return Ok(Uploaded {
                    download_param: param,
                    aes_key: key,
                    raw_size: plain.len(),
                    cipher_size: cipher.len(),
                });
            }
            Ok(r) if r.status().is_client_error() => {
                anyhow::bail!("CDN upload rejected: HTTP {}", r.status())
            }
            Ok(r) => last = format!("HTTP {}", r.status()),
            Err(e) => last = e.to_string(),
        }
        tokio::time::sleep(std::time::Duration::from_secs(attempt)).await;
    }
    anyhow::bail!("CDN upload failed: {last}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aes_ecb_round_trip_with_pkcs7() {
        let key = [7u8; 16];
        for n in [0usize, 1, 15, 16, 17, 100] {
            let plain: Vec<u8> = (0..n as u8).collect();
            let c = encrypt(&plain, &key);
            assert_eq!(c.len(), padded_size(n));
            assert_eq!(decrypt(&c, &key).unwrap(), plain);
        }
        assert!(decrypt(&encrypt(b"x", &key), &[8u8; 16]).is_err());
    }

    #[test]
    fn aes_known_vector() {
        // FIPS-197 appendix C.1 (one block, no padding check: compare prefix).
        let key: [u8; 16] = unhex("000102030405060708090a0b0c0d0e0f")
            .unwrap()
            .try_into()
            .unwrap();
        let plain = unhex("00112233445566778899aabbccddeeff").unwrap();
        let c = encrypt(&plain, &key);
        assert_eq!(hex(&c[..16]), "69c4e0d86a7b0430d8cdb78070b4c55a");
    }

    #[test]
    fn keys_in_every_encoding() {
        let raw = [0xabu8; 16];
        let b64_raw = base64::engine::general_purpose::STANDARD.encode(raw);
        let b64_hex = base64::engine::general_purpose::STANDARD.encode(hex(&raw));
        assert_eq!(parse_key(&b64_raw), Some(raw));
        assert_eq!(parse_key(&b64_hex), Some(raw));
        assert_eq!(parse_key(&hex(&raw)), Some(raw));
        assert_eq!(parse_key("nope"), None);
    }
}
