//! Anthropic Messages dialect (Kimi For Coding, Magpie's /v1/messages).

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::catalog::ThinkingFormat;
use super::responses::SIGNATURE_PREFIX;
use super::sse::SseEvent;
use super::{Decoder, LlmClient, LlmError, Request, encode_image, parse_arguments};
use crate::models::{CompletionMeta, LlmEvent, Message, Role};

pub fn build_request(client: &LlmClient, messages: &[Message], tools: &[Value]) -> Request {
    let spec = &client.spec;
    let (system, msgs) = messages_to_anthropic(messages);
    let mut body = json!({
        "model": spec.wire_id(),
        "max_tokens": client.max_tokens,
        "messages": msgs,
        "stream": true,
    });
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    if spec.reasoning && spec.thinking_format == Some(ThinkingFormat::AnthropicAdaptive) {
        body["thinking"] = json!({"type": "adaptive", "display": "summarized"});
        if let Some(v) = client
            .thinking_effort
            .as_deref()
            .and_then(|e| spec.map_effort(e))
        {
            body["output_config"] = json!({"effort": v});
        }
    } else {
        // Temperature is incompatible with extended thinking.
        body["temperature"] = json!(client.temperature);
    }
    if !tools.is_empty() {
        body["tools"] = tools
            .iter()
            .map(|t| {
                let f = t.get("function").unwrap_or(t);
                json!({
                    "name": f["name"],
                    "description": f.get("description").cloned().unwrap_or(json!("")),
                    "input_schema": f.get("parameters").cloned().unwrap_or(json!({"type": "object"})),
                })
            })
            .collect();
    }
    Request {
        path: "/v1/messages",
        body,
    }
}

fn image_blocks(m: &Message) -> Vec<Value> {
    m.attachments
        .iter()
        .flatten()
        .filter_map(encode_image)
        .map(|(data, mime)| json!({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}))
        .collect()
}

fn text_or_blocks(m: &Message) -> Value {
    let mut blocks = image_blocks(m);
    if blocks.is_empty() {
        return json!(m.text());
    }
    blocks.push(json!({"type": "text", "text": m.text()}));
    Value::Array(blocks)
}

/// Split out the system prompt; merge consecutive tool results into one
/// user turn (the API wants `tool_result` blocks inside a user message).
pub fn messages_to_anthropic(messages: &[Message]) -> (String, Vec<Value>) {
    let mut system = Vec::new();
    let mut out: Vec<Value> = Vec::new();
    for m in messages {
        match m.role {
            Role::System => {
                if !m.text().is_empty() {
                    system.push(m.text().to_string());
                }
            }
            Role::User => out.push(json!({"role": "user", "content": text_or_blocks(m)})),
            Role::Assistant => {
                out.push(json!({"role": "assistant", "content": assistant_blocks(m)}))
            }
            Role::Tool => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                    "content": text_or_blocks(m),
                });
                match out.last_mut() {
                    Some(prev) if prev["role"] == "user" && prev["content"].is_array() => {
                        prev["content"].as_array_mut().unwrap().push(block);
                    }
                    _ => out.push(json!({"role": "user", "content": [block]})),
                }
            }
        }
    }
    (system.join("\n\n"), out)
}

fn assistant_blocks(m: &Message) -> Vec<Value> {
    let mut blocks = Vec::new();
    if let Some(r) = m.reasoning.as_ref().filter(|r| !r.is_empty()) {
        let mut sig = m.reasoning_signature.clone().unwrap_or_default();
        if sig.starts_with(SIGNATURE_PREFIX) {
            sig.clear(); // foreign-dialect signature after a model switch
        }
        blocks.push(json!({"type": "thinking", "thinking": r, "signature": sig}));
    }
    if !m.text().is_empty() {
        blocks.push(json!({"type": "text", "text": m.text()}));
    }
    for c in m.tool_calls.iter().flatten() {
        let input = match &c.function.arguments {
            Value::String(s) => serde_json::from_str(s).unwrap_or(json!({})),
            other => other.clone(),
        };
        blocks
            .push(json!({"type": "tool_use", "id": c.id, "name": c.function.name, "input": input}));
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }
    blocks
}

#[derive(Default)]
pub struct AnthropicDecoder {
    tools: BTreeMap<u64, (String, String, String)>,
    usage: serde_json::Map<String, Value>,
    finish_reason: Option<String>,
}

impl Decoder for AnthropicDecoder {
    fn on_event(&mut self, ev: &SseEvent, out: &mut Vec<LlmEvent>) -> Result<bool, LlmError> {
        let Ok(e) = serde_json::from_str::<Value>(&ev.data) else {
            return Ok(false);
        };
        let idx = e.get("index").and_then(Value::as_u64).unwrap_or(0);
        match e.get("type").and_then(Value::as_str).unwrap_or("") {
            "message_start" => {
                if let Some(Value::Object(u)) = e.pointer("/message/usage") {
                    self.usage.extend(u.clone());
                }
            }
            "message_delta" => {
                if let Some(Value::Object(u)) = e.get("usage") {
                    self.usage.extend(u.clone());
                }
                if let Some(s) = e.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.finish_reason = Some(s.to_string());
                }
            }
            "content_block_start" => {
                let b = &e["content_block"];
                if b["type"] == "tool_use" {
                    let id = b["id"].as_str().unwrap_or("").to_string();
                    let name = b["name"].as_str().unwrap_or("").to_string();
                    self.tools.insert(idx, (id, name, String::new()));
                }
            }
            "content_block_delta" => {
                let d = &e["delta"];
                match d["type"].as_str().unwrap_or("") {
                    "thinking_delta" => out.push(LlmEvent::Thinking {
                        text: d["thinking"].as_str().unwrap_or("").to_string(),
                        signature: None,
                    }),
                    "signature_delta" => out.push(LlmEvent::Thinking {
                        text: String::new(),
                        signature: Some(d["signature"].as_str().unwrap_or("").to_string()),
                    }),
                    "text_delta" => {
                        out.push(LlmEvent::Text(d["text"].as_str().unwrap_or("").to_string()))
                    }
                    "input_json_delta" => {
                        if let Some(t) = self.tools.get_mut(&idx) {
                            t.2.push_str(d["partial_json"].as_str().unwrap_or(""));
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                if let Some((id, name, json)) = self.tools.remove(&idx) {
                    out.push(LlmEvent::ToolCall {
                        id,
                        name,
                        arguments: parse_arguments(&json),
                    });
                }
            }
            "message_stop" => return Ok(true),
            "error" => {
                let kind = e
                    .pointer("/error/type")
                    .and_then(Value::as_str)
                    .unwrap_or("error");
                let msg = e
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown stream error");
                let text = format!("{kind}: {msg}");
                return Err(if kind == "overloaded_error" {
                    LlmError::Overloaded(text)
                } else {
                    LlmError::Other(text)
                });
            }
            _ => {}
        }
        Ok(false)
    }

    fn finish(&mut self, out: &mut Vec<LlmEvent>) {
        for (_, (id, name, json)) in std::mem::take(&mut self.tools) {
            out.push(LlmEvent::ToolCall {
                id,
                name,
                arguments: parse_arguments(&json),
            });
        }
        let usage = std::mem::take(&mut self.usage);
        out.push(LlmEvent::Done(CompletionMeta {
            usage: (!usage.is_empty()).then_some(Value::Object(usage)),
            finish_reason: self.finish_reason.take(),
            ..Default::default()
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ToolCall;

    #[test]
    fn merges_tool_results_into_one_user_turn() {
        let mut a = Message::assistant("");
        a.tool_calls = Some(vec![
            ToolCall::new("t1", "ls", json!({})),
            ToolCall::new("t2", "ls", json!({})),
        ]);
        let msgs = vec![
            Message::system("sys"),
            Message::user("hi"),
            a,
            Message::tool("t1", "x"),
            Message::tool("t2", "y"),
        ];
        let (system, out) = messages_to_anthropic(&msgs);
        assert_eq!(system, "sys");
        assert_eq!(out.len(), 3);
        assert_eq!(out[2]["content"].as_array().unwrap().len(), 2);
        assert_eq!(out[1]["content"][0]["type"], "tool_use");
    }

    #[test]
    fn decodes_tool_use_blocks() {
        let mut d = AnthropicDecoder::default();
        let mut out = vec![];
        for data in [
            r#"{"type":"message_start","message":{"usage":{"input_tokens":5}}}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu","name":"read"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"x\"}"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}"#,
        ] {
            d.on_event(
                &SseEvent {
                    event: None,
                    data: data.into(),
                },
                &mut out,
            )
            .unwrap();
        }
        d.finish(&mut out);
        assert!(matches!(&out[0], LlmEvent::ToolCall { name, .. } if name == "read"));
        match &out[1] {
            LlmEvent::Done(m) => {
                assert_eq!(m.finish_reason.as_deref(), Some("tool_use"));
                assert_eq!(m.usage.as_ref().unwrap()["output_tokens"], 3);
            }
            other => panic!("{other:?}"),
        }
    }
}
