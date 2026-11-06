//! `run_code`: Code Mode's transport.
//!
//! With Code Mode on (`/code-mode on`) the model sees exactly one tool —
//! `run_code` — and every other tool becomes an SDK it calls from *inside*
//! a Python program: `await tools.read(path=...)`. Several tool calls then
//! cost one LLM round trip, and only what the program prints or returns
//! enters the conversation.
//!
//! How it runs here: Rust starts `python3` with a small bootstrap that
//! wraps the program body in an `async def`. Tool calls travel back to
//! Rust as JSON lines over the child's stdout and results return over its
//! stdin — so the tools themselves still run in Rust, through the same
//! registry (plan mode, hooks and all). Calls carry ids, so programs may
//! overlap independent reads with `asyncio.gather`.
//!
//! 🦀 Rust notes: `tokio::select!` drives three things at once — lines
//! from the child, finished tool calls (a `FuturesUnordered`), and the
//! timeout/cancel — without threads or locks.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::FuturesUnordered;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};

use super::{Tool, ToolContext, ToolRegistry, arg_str};
use crate::models::ToolResult;
use crate::prompt::first_sentence;

const DEFAULT_TIMEOUT_SECONDS: f64 = 120.0;
const MAX_RESULT_CHARS: usize = 64 * 1024;

pub const CODE_ONLY_INSTRUCTION: &str = "`run_code` is the only tool you can call directly — a tool call naming any \
other tool fails. Reach every tool the SDK declares below from inside the program.";

pub const RUN_CODE_INSTRUCTIONS: &str = "`run_code` takes two required arguments: `code` — the body of an async \
Python function (top-level `await` and `return` both work) — and `description`, a short summary of what the program \
does. At run time exactly two names are bound: `tools` and `ToolCallError`. Inside the program:\n\n\
- Batch, don't drip-feed: one program should complete a whole step of the task — combine as many `tools.*` calls as \
it takes (loop over files, branch on results, tolerate failures with `try/except`). That is the point of `run_code`: \
several tool calls in one LLM round trip. A program that merely wraps a single tool call without processing its \
output wastes the round trip.\n\
- Call tools as `await tools.name(keyword=value)` (subscript access `await tools[\"name\"](...)` also works). \
Positional arguments map to the tool's declared parameter order (`await tools.read(\"src/a.py\")` == \
`path=\"src/a.py\"`), and a lone dict is spread as keyword arguments (`await tools.read({\"path\": \"src/a.py\"})`). \
Arguments are forwarded to the tool verbatim; every tool returns its text output as a `str`.\n\
- A FAILED tool call raises `ToolCallError`, whose `toolName` identifies the failed tool and whose message is \
human-readable — wrap in `try/except ToolCallError` to handle and continue.\n\
- Independent read-only calls MAY overlap under `asyncio.gather`; sequence dependent work with `await`.\n\
- Emit the run's answer with `print(...)` and/or a top-level `return <value>`. ONLY what you print and the returned \
value come back — intermediate tool results never enter the conversation, so extract just what you need.";

fn py_type(schema: &Value) -> &'static str {
    if schema.get("anyOf").is_some() || schema.get("oneOf").is_some() {
        return "Any";
    }
    match schema.get("type").and_then(Value::as_str).unwrap_or("") {
        "string" => "str",
        "integer" => "int",
        "number" => "float",
        "boolean" => "bool",
        "array" => "list",
        "object" => "dict",
        "null" => "None",
        _ => "Any",
    }
}

/// Property names in declaration order (serde_json keeps insertion order
/// with the `preserve_order` feature; we read the schema's own order).
fn param_names(parameters: &Value) -> Vec<String> {
    parameters
        .get("properties")
        .and_then(Value::as_object)
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}

fn parameter_list(parameters: &Value) -> String {
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let props = parameters.get("properties").and_then(Value::as_object);
    let Some(props) = props else {
        return String::new();
    };
    props
        .iter()
        .map(|(name, prop)| {
            if required.contains(&name.as_str()) {
                format!("{name}: {}", py_type(prop))
            } else {
                format!("{name}: {} = None", py_type(prop))
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// One program sub-call: admitted and run through the registry like any
/// call, and journaled as its own action (`parent` = the `run_code` call).
async fn run_sub_call(
    registry: &ToolRegistry,
    name: &str,
    args: Value,
    sub: &ToolContext,
) -> ToolResult {
    use crate::journal::{ActionKind, ActionPhase, ActionRecord, append};
    let parent = sub.call_id.rsplit_once('.').map_or("", |(p, _)| p);
    let record = |phase| {
        ActionRecord::new(ActionKind::Tool, &sub.call_id, name, phase)
            .with_capability(registry.capability(name))
            .with_parent(parent)
    };
    let admitted = registry.admit(name, &args) == crate::admission::Verdict::Allow;
    if let Some(path) = &sub.journal {
        let phase = if admitted {
            ActionPhase::Started
        } else {
            ActionPhase::Denied
        };
        append(path, &[record(phase)]);
    }
    let r = registry.execute(name, args, sub).await;
    if let (true, Some(path)) = (admitted, &sub.journal) {
        let phase = if r.success {
            ActionPhase::Completed
        } else {
            ActionPhase::Failed
        };
        append(path, &[record(phase)]);
    }
    r
}

/// The SDK section of the system prompt, derived from the registry.
pub fn render_tools_sdk(registry: &ToolRegistry) -> String {
    let mut lines = vec![
        "## Writing code for run_code".to_string(),
        String::new(),
        RUN_CODE_INSTRUCTIONS.into(),
        String::new(),
        "The available tools (all return text as `str`, failure raises `ToolCallError`):".into(),
        String::new(),
    ];
    for def in registry.program_definitions() {
        let f = &def["function"];
        lines.push(format!(
            "- `tools.{}({})` — {}",
            f["name"].as_str().unwrap_or(""),
            parameter_list(&f["parameters"]),
            first_sentence(f["description"].as_str().unwrap_or(""))
        ));
    }
    lines.join("\n")
}

const BOOTSTRAP: &str = r#"
import ast, asyncio, io, json, sys, threading, traceback, contextlib
_rpc_out = sys.stdout
_rpc_in = sys.stdin
_init = json.loads(_rpc_in.readline())
_params = _init["params"]

class ToolCallError(Exception):
    def __init__(self, tool_name, message):
        super().__init__(message)
        self.tool_name = tool_name
        self.toolName = tool_name

_pending = {}
_next = [0]
_loop = None
_write_lock = threading.Lock()

def _send(obj):
    with _write_lock:
        _rpc_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        _rpc_out.flush()

def _reader():
    for line in _rpc_in:
        try:
            msg = json.loads(line)
        except Exception:
            continue
        fut = _pending.pop(msg.get("id"), None)
        if fut is not None:
            _loop.call_soon_threadsafe(lambda f=fut, m=msg: f.done() or f.set_result(m))

class _Tools:
    def _binding(self, name):
        async def _call(*args, **kwargs):
            if name not in _params:
                raise ToolCallError(name, f"unknown tool: {name}")
            if args:
                params = _params[name]
                if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                    kwargs = dict(args[0])
                else:
                    if len(args) > len(params):
                        raise ToolCallError(name, f"too many positional arguments ({len(args)}) for {name} (expects at most {len(params)}: {', '.join(params) or 'none'})")
                    for i, a in enumerate(args):
                        key = params[i]
                        if key in kwargs:
                            raise ToolCallError(name, f"got multiple values for argument '{key}'")
                        kwargs[key] = a
            _next[0] += 1
            cid = _next[0]
            fut = _loop.create_future()
            _pending[cid] = fut
            _send({"type": "call", "id": cid, "name": name, "args": kwargs})
            msg = await fut
            if not msg.get("ok"):
                raise ToolCallError(name, msg.get("error") or f"tool {name} failed")
            return msg.get("output") or ""
        return _call
    def __getattr__(self, name):
        if name.startswith("_") or name == "run_code":
            raise AttributeError(name)
        return self._binding(name)
    def __getitem__(self, name):
        return self._binding(name)

def _graft(code):
    body = ast.parse(code).body
    func = ast.AsyncFunctionDef(name="_limbo_run_code",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body or [ast.Pass()], decorator_list=[], type_params=[])
    mod = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(mod)
    return mod

def _render(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return repr(v)

async def _main():
    global _loop
    _loop = asyncio.get_running_loop()
    threading.Thread(target=_reader, daemon=True).start()
    ns = {"tools": _Tools(), "ToolCallError": ToolCallError, "__name__": "__limbo_run_code__"}
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(_graft(_init["code"]), "<run_code>", "exec"), ns)
            value = await ns["_limbo_run_code"]()
        logs = [s.rstrip() for s in (out.getvalue(), err.getvalue()) if s]
        _send({"type": "done", "ok": True, "logs": logs, "value": _render(value)})
    except BaseException as e:
        captured = [s.rstrip() for s in (out.getvalue(), err.getvalue()) if s]
        msg = f"code run failed ({type(e).__name__}): {e}"
        if captured:
            msg += "\nCaptured output:\n" + "\n".join(captured)
        msg += "\nTraceback:\n" + traceback.format_exc()
        _send({"type": "done", "ok": False, "error": msg})

asyncio.run(_main())
"#;

pub struct RunCodeTool;

#[async_trait]
impl Tool for RunCodeTool {
    fn name(&self) -> &'static str {
        "run_code"
    }
    fn description(&self) -> String {
        "Execute a Python program that orchestrates the available tools: batch every `tools.*` call a task step \
         needs into ONE program (loops, conditionals, error handling, `asyncio.gather` for independent reads) \
         instead of one tool call per program — that is what saves LLM round trips. Takes two required arguments: \
         `code`, the BODY of an async Python function (top-level `await` and `return` work), and `description`, a \
         short summary of what the program does. Call tools as `await tools.name(kwarg=value)` (positional args \
         follow the SDK's declared parameter order; a lone dict also works) per the SDK declarations in the system \
         prompt. Only what you print or return comes back — intermediate tool results never enter the \
         conversation, so extract just what you need."
            .into()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The program: the body of an async Python function. Use `await tools.<name>(...)` to call tools, `print(...)` for logs, and a top-level `return <value>` for the final answer."},
                "description": {"type": "string", "description": "Clear, concise description of what this program does in active voice, 5-10 words (shown in the UI)."},
                "timeout": {"type": "number", "description": format!("Optional wall-clock budget in seconds (default {DEFAULT_TIMEOUT_SECONDS}).")},
            },
            "required": ["code", "description"],
        })
    }
    /// Its sub-calls go through the registry, which applies plan mode to
    /// each of them; the transport itself changes nothing.
    fn effect(&self) -> crate::effect::Effect {
        crate::effect::Effect::Session
    }
    fn provider(&self) -> crate::capability::Provider {
        crate::capability::Provider::Program
    }
    fn program_callable(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let Some(code) = arg_str(&args, "code").filter(|c| !c.trim().is_empty()) else {
            return ToolResult::err("run_code: missing required argument 'code'");
        };
        if arg_str(&args, "description").is_none_or(|d| d.trim().is_empty()) {
            return ToolResult::err("run_code: missing required argument 'description'");
        }
        let timeout = match args.get("timeout") {
            None | Some(Value::Null) => DEFAULT_TIMEOUT_SECONDS,
            Some(v) => match v
                .as_f64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            {
                Some(t) if t > 0.0 => t,
                Some(_) => return ToolResult::err("run_code: timeout must be > 0"),
                None => return ToolResult::err("run_code: invalid timeout"),
            },
        };
        let Some(registry) = ctx.registry.clone() else {
            return ToolResult::err("run_code: registry not attached");
        };
        run_program(code, timeout, &registry, ctx).await
    }
}

async fn run_program(
    code: &str,
    timeout: f64,
    registry: &ToolRegistry,
    ctx: &ToolContext,
) -> ToolResult {
    let params: serde_json::Map<String, Value> = registry
        .program_definitions()
        .iter()
        .map(|d| {
            let f = &d["function"];
            (
                f["name"].as_str().unwrap_or("").to_string(),
                json!(param_names(&f["parameters"])),
            )
        })
        .collect();
    let mut child = match tokio::process::Command::new("python3")
        .args(["-I", "-u", "-c", BOOTSTRAP])
        .current_dir(&ctx.workdir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return ToolResult::err(format!(
                "run_code: could not start python3 ({e}). Code Mode needs Python 3 on PATH; turn it off with \
                 /code-mode off."
            ));
        }
    };
    let mut stdin = child.stdin.take().expect("piped");
    let stdout = child.stdout.take().expect("piped");
    let mut stderr = child.stderr.take().expect("piped");
    let init = json!({"code": code, "params": params}).to_string() + "\n";
    if stdin.write_all(init.as_bytes()).await.is_err() {
        return ToolResult::err("run_code: python3 exited before the program started");
    }
    let mut lines = BufReader::new(stdout).lines();
    let mut running = FuturesUnordered::new();
    let deadline = tokio::time::sleep(Duration::from_secs_f64(timeout));
    tokio::pin!(deadline);
    let done: Value = loop {
        tokio::select! {
            line = lines.next_line() => {
                let Ok(Some(line)) = line else {
                    let mut err = String::new();
                    let _ = stderr.read_to_string(&mut err).await;
                    break json!({"ok": false, "error": format!("run_code: python3 exited unexpectedly\n{}", err.trim())});
                };
                let Ok(msg) = serde_json::from_str::<Value>(&line) else { continue };
                match msg.get("type").and_then(Value::as_str) {
                    Some("done") => break msg,
                    Some("call") => {
                        let id = msg.get("id").cloned().unwrap_or(Value::Null);
                        let name = msg.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                        let call_args = msg.get("args").cloned().unwrap_or_else(|| json!({}));
                        let registry = registry.clone();
                        let mut sub = ctx.clone();
                        sub.call_id = format!("{}.{}", ctx.call_id, &crate::models::new_id()[..6]);
                        let provider = registry.capability(&name).map(|c| c.provider).unwrap_or_default();
                        ctx.progress(format!("tools.{name}(…) · {}", provider.label()));
                        running.push(async move {
                            let r = if registry.get(&name).is_some_and(|t| !t.program_callable()) {
                                ToolResult::err(format!("{name} cannot be called from a program; call it as a tool"))
                            } else {
                                run_sub_call(&registry, &name, call_args, &sub).await
                            };
                            (id, r)
                        });
                    }
                    _ => {}
                }
            }
            Some((id, r)) = running.next() => {
                let reply = if r.success {
                    json!({"id": id, "ok": true, "output": r.output.unwrap_or_default()})
                } else {
                    json!({"id": id, "ok": false, "error": r.error.unwrap_or_default()})
                };
                let _ = stdin.write_all((reply.to_string() + "\n").as_bytes()).await;
            }
            _ = &mut deadline => {
                let _ = child.kill().await;
                return ToolResult::err(format!("code run failed (timeout): program exceeded {timeout}s"));
            }
            _ = ctx.cancel.cancelled() => {
                let _ = child.kill().await;
                return ToolResult::err("run_code: program interrupted (ESC)");
            }
        }
    };
    let _ = child.kill().await;
    if done.get("ok").and_then(Value::as_bool) != Some(true) {
        return ToolResult::err(
            done.get("error")
                .and_then(Value::as_str)
                .unwrap_or("run_code failed")
                .to_string(),
        );
    }
    let mut parts: Vec<String> = done
        .get("logs")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();
    if let Some(v) = done
        .get("value")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
    {
        parts.push(v.to_string());
    }
    let mut output = if parts.is_empty() {
        "(run_code completed with no output)".to_string()
    } else {
        parts.join("\n")
    };
    if output.chars().count() > MAX_RESULT_CHARS {
        output =
            output.chars().take(MAX_RESULT_CHARS).collect::<String>() + "\n[Output truncated.]";
    }
    ToolResult::ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_python() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    #[tokio::test]
    async fn program_calls_tools_and_returns() {
        if !has_python() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "alpha\nbeta\n").unwrap();
        let reg = ToolRegistry::core(true);
        let ctx = ToolContext::simple(dir.path());
        let code = "txt = await tools.read('a.txt')\nprint('lines:', len(txt.splitlines()))\n\
                    try:\n    await tools.read(path='missing.txt')\nexcept ToolCallError as e:\n    print('failed', e.toolName)\n\
                    a, b = await asyncio.gather(tools.ls(path='.'), tools.read({'path': 'a.txt'}))\nreturn {'n': 2}";
        let code = format!("import asyncio\n{code}");
        let r = reg
            .execute(
                "run_code",
                json!({"code": code, "description": "test"}),
                &ctx,
            )
            .await;
        let out = r.output.clone().unwrap_or_else(|| r.error.clone().unwrap());
        assert!(r.success, "{out}");
        assert!(out.contains("failed read"), "{out}");
        assert!(out.contains("{\"n\": 2}"), "{out}");
    }

    #[tokio::test]
    async fn sub_calls_are_journaled() {
        if !has_python() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let journal = dir.path().join("s.jsonl");
        std::fs::write(&journal, "").unwrap();
        struct NoWrite;
        impl crate::admission::Policy for NoWrite {
            fn admit(&self, req: &crate::admission::ActionRequest) -> crate::admission::Verdict {
                match req.name() {
                    "write" => crate::admission::Verdict::Deny("no".into()),
                    _ => crate::admission::Verdict::Allow,
                }
            }
        }
        let mut reg = ToolRegistry::core(true);
        reg.policies.push(std::sync::Arc::new(NoWrite));
        let mut ctx = ToolContext::simple(dir.path());
        ctx.call_id = "rc1".into();
        ctx.journal = Some(journal.clone());
        let code = "await tools.ls(path='.')\ntry:\n    await tools.write(path='x', content='y')\nexcept ToolCallError:\n    pass";
        let r = reg
            .execute("run_code", json!({"code": code, "description": "t"}), &ctx)
            .await;
        assert!(r.success, "{:?}", r.error);
        let recs: Vec<Value> = std::fs::read_to_string(&journal)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let phases: Vec<(&str, &str)> = recs
            .iter()
            .map(|r| (r["name"].as_str().unwrap(), r["phase"].as_str().unwrap()))
            .collect();
        assert_eq!(
            phases,
            [("ls", "started"), ("ls", "completed"), ("write", "denied")]
        );
        assert!(
            recs.iter()
                .all(|r| r["parent"] == "rc1" && r["provider"] == "tool")
        );
    }

    #[tokio::test]
    async fn exceptions_and_timeouts() {
        if !has_python() {
            return;
        }
        let reg = ToolRegistry::core(true);
        let ctx = ToolContext::simple(std::env::temp_dir());
        let r = reg
            .execute(
                "run_code",
                json!({"code": "print('x')\nraise ValueError('boom')", "description": "t"}),
                &ctx,
            )
            .await;
        let e = r.error.unwrap();
        assert!(
            e.contains("ValueError") && e.contains("boom") && e.contains("Captured output"),
            "{e}"
        );
        let r = reg
            .execute(
                "run_code",
                json!({"code": "while True: pass", "description": "t", "timeout": 0.5}),
                &ctx,
            )
            .await;
        assert!(r.error.unwrap().contains("timeout"));
    }

    #[test]
    fn sdk_lists_program_tools() {
        let reg = ToolRegistry::core(true);
        let sdk = render_tools_sdk(&reg);
        assert!(sdk.contains("`tools.read(path: str"), "{sdk}");
        assert!(!sdk.contains("tools.run_code"));
    }
}
