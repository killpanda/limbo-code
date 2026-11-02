//! OpenAI Responses API dialect (Codex models, Magpie's /v1/responses).
//!
//! Requests use `store: false`, so reasoning items come back encrypted and
//! must be replayed verbatim on the next request. We keep them in
//! `Message::reasoning_signature`, prefixed with `responses:`.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::catalog::ThinkingFormat;
use super::sse::SseEvent;
use super::{
    Decoder, LlmClient, LlmError, Request, arguments_string, encode_image, parse_arguments,
};
use crate::models::{CompletionMeta, LlmEvent, Message, Role};

pub const SIGNATURE_PREFIX: &str = "responses:";

pub fn build_request(client: &LlmClient, messages: &[Message], tools: &[Value]) -> Request {
    let spec = &client.spec;
    let (instructions, input) = messages_to_responses(messages);
    let mut body = json!({
        "model": spec.wire_id(),
        "store": false,
        "stream": true,
        "input": input,
        "max_output_tokens": client.max_tokens,
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });
    if !instructions.is_empty() {
        body["instructions"] = json!(instructions);
    }
    if let Some(key) = &client.prompt_cache_key {
        body["prompt_cache_key"] = json!(key);
    }
    if spec.reasoning {
        body["include"] = json!(["reasoning.encrypted_content"]);
        if spec.thinking_format == Some(ThinkingFormat::OpenAiResponses)
            && let Some(v) = client
                .thinking_effort
                .as_deref()
                .and_then(|e| spec.map_effort(e))
        {
            body["reasoning"] = json!({"effort": v, "summary": "auto"});
        }
    } else {
        body["temperature"] = json!(client.temperature);
    }
    if !tools.is_empty() {
        body["tools"] = tools
            .iter()
            .map(|t| {
                let f = t.get("function").unwrap_or(t);
                json!({
                    "type": "function",
                    "name": f["name"],
                    "description": f.get("description").cloned().unwrap_or(json!("")),
                    "parameters": f.get("parameters").cloned().unwrap_or(json!({"type": "object"})),
                })
            })
            .collect();
    }
    Request {
        path: "/responses",
        body,
    }
}

fn image_blocks(m: &Message) -> Vec<Value> {
    m.attachments
        .iter()
        .flatten()
        .filter_map(encode_image)
        .map(|(data, mime)| json!({"type": "input_image", "detail": "auto", "image_url": format!("data:{mime};base64,{data}")}))
        .collect()
}

pub fn messages_to_responses(messages: &[Message]) -> (String, Vec<Value>) {
    let mut system = Vec::new();
    let mut items = Vec::new();
    for m in messages {
        match m.role {
            Role::System => {
                if !m.text().is_empty() {
                    system.push(m.text().to_string());
                }
            }
            Role::User => {
                let mut blocks = image_blocks(m);
                blocks.push(json!({"type": "input_text", "text": m.text()}));
                items.push(json!({"role": "user", "content": blocks}));
            }
            Role::Assistant => {
                if let Some(sig) = m.reasoning_signature.as_deref()
                    && let Some(raw) = sig.strip_prefix(SIGNATURE_PREFIX)
                    && let Ok(item) = serde_json::from_str::<Value>(raw)
                {
                    items.push(item);
                }
                if !m.text().is_empty() {
                    items.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": m.text(), "annotations": []}],
                        "status": "completed",
                    }));
                }
                for c in m.tool_calls.iter().flatten() {
                    items.push(json!({
                        "type": "function_call",
                        "call_id": c.id,
                        "name": c.function.name,
                        "arguments": arguments_string(&c.function.arguments),
                    }));
                }
            }
            Role::Tool => {
                let blocks = image_blocks(m);
                let output = if blocks.is_empty() {
                    json!(m.text())
                } else {
                    let mut b = blocks;
                    b.push(json!({"type": "input_text", "text": m.text()}));
                    Value::Array(b)
                };
                items.push(json!({
                    "type": "function_call_output",
                    "call_id": m.tool_call_id.clone().unwrap_or_default(),
                    "output": output,
                }));
            }
        }
    }
    (system.join("\n\n"), items)
}

fn signature_of(item: &Value) -> Option<String> {
    item.get("encrypted_content").filter(|v| !v.is_null())?;
    Some(format!("{SIGNATURE_PREFIX}{item}"))
}

#[derive(Default)]
pub struct ResponsesDecoder {
    calls: BTreeMap<String, (String, String, String)>,
    usage: Option<Value>,
    finish_reason: Option<String>,
    have_signature: bool,
}

impl Decoder for ResponsesDecoder {
    fn on_event(&mut self, ev: &SseEvent, out: &mut Vec<LlmEvent>) -> Result<bool, LlmError> {
        let Ok(e) = serde_json::from_str::<Value>(&ev.data) else {
            return Ok(false);
        };
        let s = |v: &Value| v.as_str().unwrap_or("").to_string();
        match e["type"].as_str().unwrap_or("") {
            "response.output_text.delta" => out.push(LlmEvent::Text(s(&e["delta"]))),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                out.push(LlmEvent::Thinking {
                    text: s(&e["delta"]),
                    signature: None,
                })
            }
            "response.output_item.added" if e["item"]["type"] == "function_call" => {
                let it = &e["item"];
                self.calls.insert(
                    s(&it["id"]),
                    (s(&it["call_id"]), s(&it["name"]), String::new()),
                );
            }
            "response.function_call_arguments.delta" => {
                if let Some(c) = self.calls.get_mut(&s(&e["item_id"])) {
                    c.2.push_str(e["delta"].as_str().unwrap_or(""));
                }
            }
            "response.function_call_arguments.done" => {
                if let Some(c) = self.calls.get_mut(&s(&e["item_id"]))
                    && let Some(a) = e["arguments"].as_str()
                {
                    c.2 = a.to_string();
                }
            }
            "response.output_item.done" => {
                let it = &e["item"];
                match it["type"].as_str().unwrap_or("") {
                    "function_call" => {
                        let (id, name, args) =
                            self.calls.remove(&s(&it["id"])).unwrap_or_else(|| {
                                (s(&it["call_id"]), s(&it["name"]), s(&it["arguments"]))
                            });
                        out.push(LlmEvent::ToolCall {
                            id,
                            name,
                            arguments: parse_arguments(&args),
                        });
                    }
                    "reasoning" => {
                        if let Some(sig) = signature_of(it) {
                            self.have_signature = true;
                            out.push(LlmEvent::Thinking {
                                text: String::new(),
                                signature: Some(sig),
                            });
                        }
                    }
                    _ => {}
                }
            }
            "response.completed" | "response.incomplete" => {
                let r = &e["response"];
                if r["usage"].is_object() {
                    self.usage = Some(r["usage"].clone());
                }
                let mut status = s(&r["status"]);
                if status == "incomplete" {
                    status = r
                        .pointer("/incomplete_details/reason")
                        .and_then(Value::as_str)
                        .unwrap_or("incomplete")
                        .to_string();
                }
                self.finish_reason = Some(status);
                if !self.have_signature
                    && let Some(item) = r["output"]
                        .as_array()
                        .and_then(|o| o.iter().find(|i| i["type"] == "reasoning"))
                    && let Some(sig) = signature_of(item)
                {
                    out.push(LlmEvent::Thinking {
                        text: String::new(),
                        signature: Some(sig),
                    });
                }
                return Ok(true);
            }
            "response.failed" | "error" => {
                let err = if e["type"] == "error" {
                    e.clone()
                } else {
                    e["response"]["error"].clone()
                };
                let code = err["code"].as_str().unwrap_or("error").to_string();
                let msg = err["message"].as_str().unwrap_or("unknown stream error");
                let text = format!("{code}: {msg}");
                return Err(
                    if code.contains("overload") || code.contains("rate_limit") {
                        LlmError::Overloaded(text)
                    } else {
                        LlmError::Other(text)
                    },
                );
            }
            _ => {}
        }
        Ok(false)
    }

    fn finish(&mut self, out: &mut Vec<LlmEvent>) {
        for (_, (id, name, args)) in std::mem::take(&mut self.calls) {
            out.push(LlmEvent::ToolCall {
                id,
                name,
                arguments: parse_arguments(&args),
            });
        }
        out.push(LlmEvent::Done(CompletionMeta {
            usage: self.usage.take(),
            finish_reason: self.finish_reason.take(),
            ..Default::default()
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replays_reasoning_items() {
        let mut a = Message::assistant("ok");
        a.reasoning_signature = Some(format!(
            "{SIGNATURE_PREFIX}{}",
            json!({"type": "reasoning", "encrypted_content": "x"})
        ));
        let (_, items) = messages_to_responses(&[Message::system("s"), a]);
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[1]["type"], "message");
    }
}
