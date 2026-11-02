//! `web_fetch` (daily mode): open a URL and return its readable text.
//!
//! HTML becomes Markdown-ish text with a tiny hand-written tag scanner (no
//! DOM): scripts, navigation and forms are skipped, and when the page has
//! an `<article>`/`<main>` only that part is kept. PDFs go through
//! `pdftotext` (poppler) or Python's `pypdf`, whichever is installed.
//!
//! 🦀 Rust notes: the scanner walks `&str` by byte index but only ever
//! slices at ASCII bytes (`<`, `>`, quotes), so every slice lands on a
//! UTF-8 boundary and can never panic.

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use super::{Tool, ToolContext, arg_i64, arg_str};
use crate::models::ToolResult;

const MAX_DOWNLOAD_BYTES: usize = 5 * 1024 * 1024;
const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 \
Safari/537.36 limbo";
const TEXT_TYPES: [&str; 6] = [
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/xml",
    "text/xml",
];
const SKIP: [&str; 14] = [
    "script", "style", "noscript", "svg", "template", "iframe", "nav", "footer", "header", "aside",
    "form", "button", "select", "head",
];
const BLOCK: [&str; 18] = [
    "p",
    "div",
    "section",
    "article",
    "main",
    "ul",
    "ol",
    "table",
    "tr",
    "blockquote",
    "pre",
    "figure",
    "figcaption",
    "dl",
    "dt",
    "dd",
    "br",
    "hr",
];

pub struct WebFetchTool {
    pub max_chars: usize,
    pub timeout: f64,
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &'static str {
        "web_fetch"
    }
    fn description(&self) -> String {
        format!(
            "Open a web page and return its readable text as Markdown (title, headings, lists, links; navigation \
             and scripts stripped). Use it to read search results or links the user gives you; PDF links work too \
             (text only). Long pages come in chunks of {} characters and their full text is saved to a file you \
             can read; pass offset to continue.",
            self.max_chars
        )
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http(s) URL to open."},
                "offset": {"type": "integer", "description": "Character offset to continue a long page from (default 0)."},
            },
            "required": ["url"],
        })
    }
    fn replay_safe(&self) -> bool {
        true
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let url = arg_str(&args, "url").unwrap_or("").trim().to_string();
        if url.is_empty() {
            return ToolResult::err("url must be a non-empty string");
        }
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return ToolResult::err("url must start with http:// or https://");
        }
        let offset = match args.get("offset") {
            None | Some(Value::Null) => 0,
            Some(_) => match arg_i64(&args, "offset") {
                Some(n) if n >= 0 => n as usize,
                _ => return ToolResult::err("offset must be a non-negative integer"),
            },
        };
        let (final_url, ctype, body) = match self.download(&url).await {
            Ok(v) => v,
            Err(e) => return ToolResult::err(e),
        };
        let mut title = String::new();
        let text = if ctype.starts_with("application/pdf") || body.starts_with(b"%PDF-") {
            match pdf_text(&body).await {
                Ok(t) => t,
                Err(e) => return ToolResult::err(e),
            }
        } else {
            let raw = String::from_utf8_lossy(&body).into_owned();
            let looks_html = raw.chars().take(2000).collect::<String>().to_lowercase().contains("<html");
            if ctype.contains("html") || (ctype.is_empty() && looks_html) {
                let (t, x) = html_to_text(&raw, &final_url);
                title = t;
                x
            } else if !ctype.is_empty() && !TEXT_TYPES.iter().any(|t| ctype.starts_with(t)) {
                return ToolResult::err(format!("Unsupported content type: {ctype}"));
            } else {
                raw
            }
        };
        match self.page(&ctx.workdir, &final_url, &title, &text, offset) {
            Ok(out) => ToolResult::ok(out),
            Err(e) => ToolResult::err(e),
        }
    }
}

impl WebFetchTool {
    async fn download(&self, url: &str) -> Result<(String, String, Vec<u8>), String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(self.timeout))
            .user_agent(USER_AGENT)
            .build()
            .map_err(|e| e.to_string())?;
        let resp = client
            .get(url)
            .header("Accept", "text/html,*/*;q=0.8")
            .send()
            .await
            .map_err(|e| format!("Could not fetch {url}: {e}"))?;
        let status = resp.status();
        let final_url = resp.url().to_string();
        if status.as_u16() >= 400 {
            return Err(format!("HTTP {} for {final_url}", status.as_u16()));
        }
        let ctype = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let mut stream = resp.bytes_stream();
        let mut body = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Could not fetch {url}: {e}"))?;
            body.extend_from_slice(&chunk);
            if body.len() >= MAX_DOWNLOAD_BYTES {
                break;
            }
        }
        Ok((final_url, ctype, body))
    }

    fn page(&self, workdir: &Path, url: &str, title: &str, text: &str, offset: usize) -> Result<String, String> {
        let chars: Vec<char> = text.chars().collect();
        let total = chars.len();
        if offset > 0 && offset >= total {
            return Err(format!(
                "offset {offset} is past the end of the page ({total} characters)"
            ));
        }
        let end = (offset + self.max_chars).min(total);
        let chunk: String = chars[offset..end].iter().collect();
        let mut out = format!("URL: {url}");
        if !title.is_empty() {
            out.push_str(&format!("\nTitle: {title}"));
        }
        out.push_str("\n\n");
        out.push_str(if chunk.is_empty() {
            "(no readable text)"
        } else {
            &chunk
        });
        if offset > 0 || end < total {
            out.push_str(&format!("\n\n[Characters {offset}-{end} of {total}."));
            if end < total {
                out.push_str(&format!(" Call again with offset={end} for more.]"));
            } else {
                out.push(']');
            }
        }
        if offset == 0
            && end < total
            && let Some(saved) = save_full(workdir, url, title, text)
        {
            out.push_str(&format!(
                "\n[Full text saved to {}; read it with the read tool.]",
                saved.display()
            ));
        }
        Ok(out)
    }
}

/// Keep a long page's full text in `<workdir>/fetched/` (best effort).
fn save_full(workdir: &Path, url: &str, title: &str, text: &str) -> Option<PathBuf> {
    let host = url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(String::from))
        .unwrap_or_else(|| "page".into());
    let host: String = host
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let digest = sha1_smol::Sha1::from(url).digest().to_string();
    let path = workdir
        .join("fetched")
        .join(format!("{host}-{}.md", &digest[..8]));
    let header = if title.is_empty() {
        format!("<!-- {url} -->\n\n")
    } else {
        format!("<!-- {url} -->\n# {title}\n\n")
    };
    std::fs::create_dir_all(path.parent()?).ok()?;
    std::fs::write(&path, header + text).ok()?;
    Some(path)
}

async fn pdf_text(body: &[u8]) -> Result<String, String> {
    let dir = tempfile_dir();
    let pdf = dir.join(format!("{}.pdf", crate::models::new_id()));
    std::fs::write(&pdf, body).map_err(|e| e.to_string())?;
    let result = async {
        if let Ok(out) = tokio::process::Command::new("pdftotext")
            .args(["-layout", "-enc", "UTF-8"])
            .arg(&pdf)
            .arg("-")
            .output()
            .await
            && out.status.success()
        {
            return Ok(String::from_utf8_lossy(&out.stdout).into_owned());
        }
        let script = "import sys\nfrom pypdf import PdfReader\nr=PdfReader(sys.argv[1])\n\
                      print('\\n\\n'.join(f'## Page {i}\\n\\n'+(p.extract_text() or '').strip() for i,p in enumerate(r.pages,1)))";
        match tokio::process::Command::new("python3")
            .args(["-c", script])
            .arg(&pdf)
            .output()
            .await
        {
            Ok(out) if out.status.success() => Ok(String::from_utf8_lossy(&out.stdout).into_owned()),
            _ => Err("This URL is a PDF and neither pdftotext (poppler) nor Python's pypdf is installed. Or \
                      download it with bash (curl -L -o file.pdf URL) and use the pdf skill."
                .to_string()),
        }
    }
    .await;
    let _ = std::fs::remove_file(&pdf);
    result
}

fn tempfile_dir() -> PathBuf {
    std::env::temp_dir()
}

// -- HTML → text ------------------------------------------------------------------

/// (title, readable text) of an HTML page.
pub fn html_to_text(html: &str, base_url: &str) -> (String, String) {
    let base = url::Url::parse(base_url).ok();
    let mut ex = Extractor {
        base,
        ..Default::default()
    };
    let bytes = html.as_bytes();
    let mut i = 0;
    let mut text_start = 0;
    while i < bytes.len() {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        // flush text before the tag
        if text_start < i {
            ex.data(&decode_entities(&html[text_start..i]));
        }
        let rest = &html[i..];
        if rest.starts_with("<!--") {
            let end = rest.find("-->").map(|e| i + e + 3).unwrap_or(bytes.len());
            i = end;
            text_start = i;
            continue;
        }
        let Some(close) = find_tag_end(bytes, i + 1) else {
            // a stray '<': treat as text
            i += 1;
            continue;
        };
        let inner = &html[i + 1..close];
        i = close + 1;
        text_start = i;
        if let Some(name) = inner.strip_prefix('/') {
            ex.end(&tag_name(name));
        } else if inner.starts_with('!') || inner.starts_with('?') {
            // doctype / processing instruction
        } else {
            let name = tag_name(inner);
            let attrs = parse_attrs(inner);
            let self_closing = inner.trim_end().ends_with('/');
            ex.start(&name, &attrs);
            // raw-text elements: skip to their end tag without parsing
            if (name == "script" || name == "style") && !self_closing {
                let lower = html[i..].to_ascii_lowercase();
                let end_tag = format!("</{name}");
                let end = lower.find(&end_tag).map(|e| i + e).unwrap_or(bytes.len());
                if name == "script" || name == "style" {
                    i = end;
                    text_start = i;
                }
            } else if self_closing {
                ex.end(&name);
            }
        }
    }
    if text_start < bytes.len() {
        ex.data(&decode_entities(&html[text_start..]));
    }
    let title = ex.title.split_whitespace().collect::<Vec<_>>().join(" ");
    (title, ex.text())
}

/// The `>` that ends a tag starting at `from`, skipping quoted attribute values.
fn find_tag_end(bytes: &[u8], from: usize) -> Option<usize> {
    let mut quote: Option<u8> = None;
    let first = *bytes.get(from)?;
    if !(first.is_ascii_alphabetic() || first == b'/' || first == b'!' || first == b'?') {
        return None;
    }
    for (j, &b) in bytes.iter().enumerate().skip(from) {
        match quote {
            Some(q) if b == q => quote = None,
            Some(_) => {}
            None if b == b'"' || b == b'\'' => quote = Some(b),
            None if b == b'>' => return Some(j),
            None => {}
        }
    }
    None
}

fn tag_name(inner: &str) -> String {
    inner
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn parse_attrs(inner: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let s = inner.trim_start();
    let s = &s[s.find(|c: char| c.is_whitespace()).unwrap_or(s.len())..];
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        while i < chars.len() && (chars[i].is_whitespace() || chars[i] == '/') {
            i += 1;
        }
        let start = i;
        while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '=' && chars[i] != '/' {
            i += 1;
        }
        let key: String = chars[start..i].iter().collect::<String>().to_ascii_lowercase();
        while i < chars.len() && chars[i].is_whitespace() {
            i += 1;
        }
        let mut value = String::new();
        if i < chars.len() && chars[i] == '=' {
            i += 1;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            if i < chars.len() && (chars[i] == '"' || chars[i] == '\'') {
                let q = chars[i];
                i += 1;
                let vs = i;
                while i < chars.len() && chars[i] != q {
                    i += 1;
                }
                value = chars[vs..i].iter().collect();
                i += 1;
            } else {
                let vs = i;
                while i < chars.len() && !chars[i].is_whitespace() {
                    i += 1;
                }
                value = chars[vs..i].iter().collect();
            }
        }
        if !key.is_empty() {
            out.push((key, decode_entities(&value)));
        }
    }
    out
}

pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('&') {
        out.push_str(&rest[..p]);
        rest = &rest[p..];
        let end = rest[1..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '#'))
            .map(|e| e + 1);
        let (ent, consumed) = match end {
            Some(e) if rest[e..].starts_with(';') => (&rest[1..e], e + 1),
            Some(e) => (&rest[1..e], e),
            None => (&rest[1..], rest.len()),
        };
        let decoded = if let Some(num) = ent.strip_prefix('#') {
            let n = if let Some(h) = num.strip_prefix(['x', 'X']) {
                u32::from_str_radix(h, 16).ok()
            } else {
                num.parse().ok()
            };
            n.and_then(char::from_u32).map(String::from)
        } else {
            match ent {
                "amp" => Some("&".into()),
                "lt" => Some("<".into()),
                "gt" => Some(">".into()),
                "quot" => Some("\"".into()),
                "apos" | "#39" => Some("'".into()),
                "nbsp" => Some(" ".into()),
                "mdash" => Some("—".into()),
                "ndash" => Some("–".into()),
                "hellip" => Some("…".into()),
                "copy" => Some("©".into()),
                "reg" => Some("®".into()),
                "laquo" => Some("«".into()),
                "raquo" => Some("»".into()),
                "ldquo" => Some("“".into()),
                "rdquo" => Some("”".into()),
                "lsquo" => Some("‘".into()),
                "rsquo" => Some("’".into()),
                "middot" => Some("·".into()),
                _ => None,
            }
        };
        match decoded {
            Some(d) if consumed > 1 => {
                out.push_str(&d);
                rest = &rest[consumed..];
            }
            _ => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[derive(Default)]
struct Extractor {
    base: Option<url::Url>,
    title: String,
    in_title: bool,
    skip_tag: Option<String>,
    skip_depth: usize,
    root_depth: usize,
    pre: usize,
    links: Vec<Option<String>>,
    parts: Vec<(String, bool)>,
}

impl Extractor {
    fn emit(&mut self, t: impl Into<String>) {
        self.parts.push((t.into(), self.root_depth > 0));
    }
    fn start(&mut self, tag: &str, attrs: &[(String, String)]) {
        if tag == "title" {
            self.in_title = true;
        }
        if let Some(skip) = &self.skip_tag {
            if tag == skip {
                self.skip_depth += 1;
            }
            return;
        }
        if SKIP.contains(&tag) {
            self.skip_tag = Some(tag.to_string());
            self.skip_depth = 1;
            return;
        }
        if tag == "article" || tag == "main" {
            self.root_depth += 1;
        }
        let attr = |k: &str| attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        if tag == "pre" {
            self.pre += 1;
            self.emit("\n```\n");
        } else if is_heading(tag) {
            let n: usize = tag[1..].parse().unwrap_or(1);
            self.emit(format!("\n\n{} ", "#".repeat(n)));
        } else if tag == "li" {
            self.emit("\n- ");
        } else if tag == "td" || tag == "th" {
            self.emit(" | ");
        } else if BLOCK.contains(&tag) {
            self.emit(if tag == "p" || tag == "blockquote" {
                "\n\n"
            } else {
                "\n"
            });
        } else if tag == "a" {
            match attr("href").filter(|h| !h.starts_with('#') && !h.starts_with("javascript:")) {
                Some(href) => {
                    let abs = self
                        .base
                        .as_ref()
                        .and_then(|b| b.join(&href).ok())
                        .map(|u| u.to_string())
                        .unwrap_or(href);
                    self.links.push(Some(abs));
                    self.emit("[");
                }
                None => self.links.push(None),
            }
        } else if tag == "img" {
            let alt = attr("alt").unwrap_or_default();
            if !alt.trim().is_empty() {
                self.emit(format!("[image: {}]", alt.trim()));
            }
        }
    }
    fn end(&mut self, tag: &str) {
        if tag == "title" {
            self.in_title = false;
        }
        if let Some(skip) = &self.skip_tag {
            if tag == skip {
                self.skip_depth -= 1;
                if self.skip_depth == 0 {
                    self.skip_tag = None;
                }
            }
            return;
        }
        if tag == "a" && !self.links.is_empty() {
            if let Some(Some(href)) = self.links.pop() {
                self.emit(format!("]({href})"));
            }
        } else if tag == "pre" && self.pre > 0 {
            self.pre -= 1;
            self.emit("\n```\n");
        } else if is_heading(tag) || BLOCK.contains(&tag) {
            self.emit("\n");
        }
        if (tag == "article" || tag == "main") && self.root_depth > 0 {
            self.root_depth -= 1;
        }
    }
    fn data(&mut self, d: &str) {
        if self.in_title {
            self.title.push_str(d);
        }
        if self.skip_tag.is_some() {
            return;
        }
        if self.pre > 0 {
            self.emit(d);
        } else {
            let collapsed = collapse_ws(d);
            self.emit(collapsed);
        }
    }
    fn text(&self) -> String {
        let rooted: String = self
            .parts
            .iter()
            .filter(|(_, inside)| *inside)
            .map(|(t, _)| t.as_str())
            .collect();
        let chosen = if rooted.trim().is_empty() {
            self.parts.iter().map(|(t, _)| t.as_str()).collect()
        } else {
            rooted
        };
        let empty_links = regex::Regex::new(r"\[\s*\]\([^)]*\)").unwrap();
        let t = empty_links.replace_all(&chosen, "");
        let t = regex::Regex::new(r"[ \t]+\n").unwrap().replace_all(&t, "\n");
        let t = regex::Regex::new(r"\n[ \t]+").unwrap().replace_all(&t, "\n");
        let t = regex::Regex::new(r"\n{3,}").unwrap().replace_all(&t, "\n\n");
        t.trim().to_string()
    }
}

fn is_heading(tag: &str) -> bool {
    tag.len() == 2 && tag.starts_with('h') && matches!(tag.as_bytes()[1], b'1'..=b'6')
}

fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !ws {
                out.push(' ');
            }
            ws = true;
        } else {
            out.push(c);
            ws = false;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_article_and_links() {
        let html = r#"<html><head><title> Hello &amp; bye </title><script>var x="<p>";</script></head>
        <body><nav><a href="/home">Home</a></nav>
        <article><h1>Big</h1><p>Some <b>bold</b> text with a <a href="/x?a=1&amp;b=2">link</a>.</p>
        <ul><li>one</li><li>two</li></ul><pre>  code
  here</pre></article><footer>foot</footer></body></html>"#;
        let (title, text) = html_to_text(html, "https://ex.com/page");
        assert_eq!(title, "Hello & bye");
        assert!(text.starts_with("# Big"), "{text}");
        assert!(text.contains("[link](https://ex.com/x?a=1&b=2)"), "{text}");
        assert!(text.contains("- one"));
        assert!(text.contains("```\ncode\nhere"), "{text}");
        assert!(!text.contains("Home"));
        assert!(!text.contains("foot"));
        assert!(!text.contains("var x"));
    }

    #[test]
    fn entities() {
        assert_eq!(decode_entities("a&lt;b&#x4e2d;&#20013;&bogus;"), "a<b中中&bogus;");
    }
}
