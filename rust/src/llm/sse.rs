//! A tiny Server-Sent Events parser.
//!
//! Every provider streams its response as SSE: lines of `event: x` /
//! `data: {...}` separated by a blank line. HTTP chunks can split a line
//! anywhere, so the parser buffers bytes until it sees a full event.
//!
//! 🦀 We turn a stream of byte chunks into a stream of events with
//! `futures::stream::unfold`, which threads a piece of state (the buffer and
//! the byte stream) through each step.

use futures::{Stream, StreamExt};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Incremental parser: feed bytes, take out complete events.
#[derive(Default)]
pub struct SseParser {
    buf: String,
    pending: Vec<u8>,
}

impl SseParser {
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        // A multi-byte UTF-8 character may be split across chunks: keep the
        // incomplete tail bytes for the next call.
        self.pending.extend_from_slice(chunk);
        let valid_up_to = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) => e.valid_up_to(),
        };
        let text = String::from_utf8_lossy(&self.pending[..valid_up_to]).into_owned();
        self.pending.drain(..valid_up_to);
        self.buf.push_str(&text.replace("\r\n", "\n"));

        let mut out = Vec::new();
        while let Some(pos) = self.buf.find("\n\n") {
            let block: String = self.buf.drain(..pos + 2).collect();
            if let Some(ev) = parse_block(&block) {
                out.push(ev);
            }
        }
        out
    }

    /// Whatever is left when the stream ends without a trailing blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        let rest = std::mem::take(&mut self.buf);
        parse_block(&rest)
    }
}

fn parse_block(block: &str) -> Option<SseEvent> {
    let mut ev = SseEvent::default();
    let mut data_lines = Vec::new();
    for line in block.lines() {
        if line.starts_with(':') {
            continue; // comment / keep-alive
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "event" => ev.event = Some(value.to_string()),
            "data" => data_lines.push(value.to_string()),
            _ => {}
        }
    }
    if data_lines.is_empty() && ev.event.is_none() {
        return None;
    }
    ev.data = data_lines.join("\n");
    Some(ev)
}

/// Adapt a `reqwest` byte stream into a stream of SSE events.
pub fn sse_stream(
    bytes: impl Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin + Send + 'static,
) -> impl Stream<Item = anyhow::Result<SseEvent>> + Send {
    futures::stream::unfold(
        (
            bytes,
            SseParser::default(),
            std::collections::VecDeque::new(),
            false,
        ),
        |(mut bytes, mut parser, mut queue, mut done)| async move {
            loop {
                if let Some(ev) = queue.pop_front() {
                    return Some((Ok(ev), (bytes, parser, queue, done)));
                }
                if done {
                    return None;
                }
                match bytes.next().await {
                    Some(Ok(chunk)) => queue.extend(parser.feed(&chunk)),
                    Some(Err(e)) => {
                        done = true;
                        return Some((Err(anyhow::Error::new(e)), (bytes, parser, queue, done)));
                    }
                    None => {
                        done = true;
                        if let Some(ev) = parser.finish() {
                            queue.push_back(ev);
                        }
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_events_across_chunks() {
        let mut p = SseParser::default();
        assert!(p.feed(b"data: {\"a\"").is_empty());
        let evs = p.feed(b":1}\n\nevent: done\ndata: x\n\n");
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "{\"a\":1}");
        assert_eq!(evs[1].event.as_deref(), Some("done"));
    }

    #[test]
    fn handles_split_utf8() {
        let mut p = SseParser::default();
        let bytes = "data: 你好\n\n".as_bytes();
        assert!(p.feed(&bytes[..8]).is_empty());
        let evs = p.feed(&bytes[8..]);
        assert_eq!(evs[0].data, "你好");
    }
}
