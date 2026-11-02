//! Token accounting: normalize provider usage counters, estimate prompt size.

use serde_json::Value;

use crate::models::Message;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UsageTotals {
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

fn int(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}

fn cache_traffic(u: &Value) -> u64 {
    int(u, "cache_read_input_tokens").unwrap_or(0)
        + int(u, "cache_creation_input_tokens").unwrap_or(0)
}

/// Normalize OpenAI / Anthropic / Responses / DeepSeek usage shapes.
pub fn normalize_usage(usage: Option<&Value>) -> UsageTotals {
    let Some(u) = usage.filter(|u| u.is_object()) else {
        return UsageTotals::default();
    };
    let prompt =
        int(u, "prompt_tokens").or_else(|| int(u, "input_tokens").map(|i| i + cache_traffic(u)));
    let cached = int(u, "prompt_cache_hit_tokens")
        .or_else(|| {
            u.pointer("/prompt_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
        })
        .or_else(|| {
            u.pointer("/input_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
        })
        .or_else(|| int(u, "cache_read_input_tokens"));
    let total = int(u, "total_tokens").or_else(|| {
        let i = int(u, "input_tokens");
        let o = int(u, "output_tokens");
        (i.is_some() || o.is_some()).then(|| i.unwrap_or(0) + o.unwrap_or(0) + cache_traffic(u))
    });
    UsageTotals {
        prompt_tokens: prompt,
        cached_tokens: cached,
        total_tokens: total,
    }
}

/// chars/4 + per-message overhead — only used when no real figure exists.
pub fn estimate_tokens(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(|m| {
            let mut chars =
                m.text().chars().count() + m.reasoning.as_deref().map_or(0, |r| r.chars().count());
            if let Some(calls) = &m.tool_calls {
                chars += serde_json::to_string(calls).map_or(0, |s| s.chars().count());
            }
            (chars / 4 + 4) as u64
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn anthropic_adds_cache_traffic() {
        let t = normalize_usage(Some(
            &json!({"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100}),
        ));
        assert_eq!(t.prompt_tokens, Some(110));
        assert_eq!(t.cached_tokens, Some(100));
        assert_eq!(t.total_tokens, Some(115));
    }

    #[test]
    fn deepseek_cache_hits() {
        let t = normalize_usage(Some(
            &json!({"prompt_tokens": 50, "total_tokens": 60, "prompt_cache_hit_tokens": 40}),
        ));
        assert_eq!(t.cached_tokens, Some(40));
    }
}
