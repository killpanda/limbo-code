//! OpenAI Chat Completions dialect (DeepSeek, Moonshot, GLM, Magpie, and any
//! OpenAI-compatible endpoint).

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::catalog::ThinkingFormat;
use super::sse::SseEvent;
use super::{
    Decoder, LlmClient, LlmError, Request, arguments_string, encode_image, parse_arguments,
};
use crate::models::{CompletionMeta, LlmEvent, Message, Role};

pub fn build_request(client: &LlmClient, messages: &[Message], tools: &[Value]) -> Request {
    let spec = &client.spec;
    let mut body = json!({
        "model": spec.wire_id(),
        "messages": messages_to_openai(messages, spec.requires_reasoning_content, spec.vision),
        "temperature": client.temperature,
        "max_tokens": client.max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let obj = body.as_object_mut().expect("object");
    apply_thinking(client, obj);
    if !tools.is_empty() {
        obj.insert("tools".into(), Value::Array(tools.to_vec()));
        if let Some(Value::Object(extra)) = &spec.provider.tool_extra_body {
            for (k, v) in extra {
                obj.insert(k.clone(), v.clone());
            }
        }
    }
    Request {
        path: "/chat/completions",
        body,
    }
}

fn apply_thinking(client: &LlmClient, obj: &mut Map<String, Value>) {
    let spec = &client.spec;
    let (Some(fmt), Some(effort)) = (spec.thinking_format, client.thinking_effort.as_deref())
    else {
        return;
    };
    if !spec.reasoning {
        return;
    }
    match fmt {
        ThinkingFormat::OpenAi => {
            if let Some(v) = spec.map_effort(effort) {
                obj.insert("reasoning_effort".into(), json!(v));
            }
        }
        ThinkingFormat::DeepSeek | ThinkingFormat::Zai => {
            if effort == "off" {
                if spec.thinking_can_disable {
                    obj.insert("thinking".into(), json!({"type": "disabled"}));
                }
                return;
            }
            let mut thinking = json!({"type": "enabled"});
            if fmt == ThinkingFormat::Zai {
                thinking["clear_thinking"] = json!(false);
                if let Some(v) = spec.map_effort(effort) {
                    obj.insert("reasoning_effort".into(), json!(v));
                }
            }
            obj.insert("thinking".into(), thinking);
        }
        _ => {}
    }
}

/// Convert messages to chat-completions params. Tool results stay
/// text-only (the API rejects images there); images a tool returned are sent
/// afterwards as one synthetic user message.
pub fn messages_to_openai(
    messages: &[Message],
    include_reasoning: bool,
    vision: bool,
) -> Vec<Value> {
    let mut out = Vec::new();
    let mut pending_images: Vec<Value> = Vec::new();
    let flush = |out: &mut Vec<Value>, imgs: &mut Vec<Value>| {
        if !imgs.is_empty() {
            let mut content =
                vec![json!({"type": "text", "text": "Attached image(s) from tool result:"})];
            content.append(imgs);
            out.push(json!({"role": "user", "content": content}));
        }
    };
    for m in messages {
        if m.role == Role::Tool {
            let mut has_images = false;
            if vision {
                for att in m.attachments.iter().flatten() {
                    if let Some((data, mime)) = encode_image(att) {
                        has_images = true;
                        pending_images.push(json!({"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{data}")}}));
                    }
                }
            }
            let mut content = m.text().to_string();
            if content.is_empty() {
                content = if has_images {
                    "(see attached image)"
                } else {
                    "(no tool output)"
                }
                .into();
            }
            out.push(json!({"role": "tool", "content": content, "tool_call_id": m.tool_call_id.clone().unwrap_or_default()}));
            continue;
        }
        flush(&mut out, &mut pending_images);
        out.push(message_to_openai(m, include_reasoning));
    }
    flush(&mut out, &mut pending_images);
    out
}

fn message_to_openai(m: &Message, include_reasoning: bool) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    let mut v = json!({"role": role, "content": m.text()});
    if m.role == Role::User {
        let blocks: Vec<Value> = m
            .attachments
            .iter()
            .flatten()
            .filter_map(encode_image)
            .map(|(data, mime)| json!({"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{data}")}}))
            .collect();
        if !blocks.is_empty() {
            let mut blocks = blocks;
            blocks.push(json!({"type": "text", "text": m.text()}));
            v["content"] = Value::Array(blocks);
        }
    }
    if include_reasoning && m.role == Role::Assistant {
        v["reasoning_content"] = json!(m.reasoning.clone().unwrap_or_default());
    }
    if let Some(calls) = m.tool_calls.as_ref().filter(|c| !c.is_empty()) {
        v["tool_calls"] = calls
            .iter()
            .map(|c| {
                json!({
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": arguments_string(&c.function.arguments)},
                })
            })
            .collect();
    }
    v
}

#[derive(Default)]
struct PartialCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Default)]
pub struct OpenAiDecoder {
    calls: BTreeMap<u64, PartialCall>,
    usage: Option<Value>,
    finish_reason: Option<String>,
}

impl Decoder for OpenAiDecoder {
    fn on_event(&mut self, ev: &SseEvent, out: &mut Vec<LlmEvent>) -> Result<bool, LlmError> {
        if ev.data.trim() == "[DONE]" {
            return Ok(true);
        }
        let Ok(chunk) = serde_json::from_str::<Value>(&ev.data) else {
            return Ok(false);
        };
        if let Some(err) = chunk.get("error") {
            return Err(LlmError::Other(format!("stream error: {err}")));
        }
        if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(u.clone());
        }
        let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) else {
            return Ok(false);
        };
        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(fr.to_string());
        }
        let Some(delta) = choice.get("delta") else {
            return Ok(false);
        };
        for key in ["reasoning_content", "reasoning"] {
            if let Some(r) = delta
                .get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                out.push(LlmEvent::Thinking {
                    text: r.to_string(),
                    signature: None,
                });
                break;
            }
        }
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            out.push(LlmEvent::Text(text.to_string()));
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tc in calls {
                let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                let entry = self.calls.entry(idx).or_default();
                if let Some(id) = tc
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    entry.id = id.to_string();
                }
                if let Some(f) = tc.get("function") {
                    if let Some(n) = f.get("name").and_then(Value::as_str) {
                        entry.name.push_str(n);
                    }
                    if let Some(a) = f.get("arguments").and_then(Value::as_str) {
                        entry.arguments.push_str(a);
                    }
                }
            }
        }
        Ok(false)
    }

    fn finish(&mut self, out: &mut Vec<LlmEvent>) {
        for (idx, call) in std::mem::take(&mut self.calls) {
            out.push(LlmEvent::ToolCall {
                id: if call.id.is_empty() {
                    format!("call_{idx}")
                } else {
                    call.id
                },
                name: call.name,
                arguments: parse_arguments(&call.arguments),
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
    use crate::models::ToolCall;

    #[test]
    fn decodes_streamed_tool_calls() {
        let mut d = OpenAiDecoder::default();
        let mut out = vec![];
        for data in [
            r#"{"choices":[{"delta":{"content":"hi"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{\"pa"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":10,"total_tokens":12}}"#,
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
        assert!(
            d.on_event(
                &SseEvent {
                    event: None,
                    data: "[DONE]".into()
                },
                &mut out
            )
            .unwrap()
        );
        d.finish(&mut out);
        assert!(matches!(&out[0], LlmEvent::Text(t) if t == "hi"));
        match &out[1] {
            LlmEvent::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "c1");
                assert_eq!(name, "read");
                assert_eq!(arguments["path"], "a");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(
            matches!(&out[2], LlmEvent::Done(m) if m.finish_reason.as_deref() == Some("tool_calls"))
        );
    }

    #[test]
    fn serializes_tool_call_arguments_as_string() {
        let mut m = Message::assistant("");
        m.tool_calls = Some(vec![ToolCall::new("c1", "ls", json!({"path": "."}))]);
        let v = messages_to_openai(&[m], true, false);
        assert_eq!(
            v[0]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\".\"}"
        );
        assert_eq!(v[0]["reasoning_content"], "");
    }
}
