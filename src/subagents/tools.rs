//! Task tools: what the main agent and the subagents call. They drive the
//! live board, so none of them is replay-safe except `task_status` and
//! `checkpoint`, and none is callable from a `run_code` program.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::supervisor::{Supervisor, str_list};
use crate::models::ToolResult;
use crate::tools::{Tool, ToolContext};

/// Task tools every subagent may call whatever its profile.
pub const SUBAGENT_TOOL_NAMES: [&str; 5] =
    ["checkpoint", "need", "ask_main", "finish", "propose_task"];

fn ok(text: impl Into<String>, end_turn: bool) -> ToolResult {
    let mut r = ToolResult::ok(text);
    r.end_turn = end_turn;
    r
}

fn s(args: &Value, key: &str) -> String {
    args.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

macro_rules! task_tool {
    ($ty:ident, $name:literal, $safe:expr, $desc:expr, $params:expr, |$sup:ident, $id:ident, $args:ident| $body:expr) => {
        pub struct $ty {
            sup: Arc<Supervisor>,
            #[allow(dead_code)]
            task_id: String,
        }

        #[async_trait]
        impl Tool for $ty {
            fn name(&self) -> &'static str {
                $name
            }
            fn description(&self) -> String {
                $desc.into()
            }
            fn parameters(&self) -> Value {
                $params
            }
            fn replay_safe(&self) -> bool {
                $safe
            }
            fn mutates(&self) -> bool {
                false // plan mode may still manage the board
            }
            fn program_callable(&self) -> bool {
                false
            }
            async fn run(&self, $args: Value, _ctx: &ToolContext) -> ToolResult {
                let $sup = &self.sup;
                #[allow(unused_variables)]
                let $id = self.task_id.as_str();
                $body
            }
        }
    };
}

fn task_spec() -> Value {
    json!({
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Short local key other tasks in this plan use in depends_on"},
            "title": {"type": "string"},
            "brief": {"type": "string", "description": "Everything the subagent needs: goal, context, constraints, where to look"},
            "profile": {"type": "string", "description": "Agent profile name from <available_agents> (default worker)"},
            "depends_on": {"type": "array", "items": {"type": "string"}, "description": "Keys from this plan or existing task ids that must finish first"},
            "acceptance": {"type": "string", "description": "How the result is verified (a command or a criterion)"},
            "scope_paths": {"type": "array", "items": {"type": "string"}, "description": "Glob paths this task will write (claims them)"},
            "priority": {"type": "integer"},
            "isolation": {"type": "string", "enum": ["shared", "worktree"]},
            "context": {"type": "string", "enum": ["brief", "fork"], "description": "fork = the subagent starts with a copy of this conversation"},
        },
        "required": ["title", "brief"],
    })
}

// -- main agent ----------------------------------------------------------------

task_tool!(
    PlanTasksTool,
    "plan_tasks",
    false,
    "Delegate work to background subagents by submitting a task DAG. Use it when the work splits into \
     independent parts, needs broad exploration whose details you do not need, or benefits from an independent \
     review/verification. Subagents run in parallel while you keep talking to the user; their results come back \
     to you as <subagent_updates> messages. Submit the whole plan in one call.",
    json!({
        "type": "object",
        "properties": {
            "tasks": {"type": "array", "items": task_spec()},
            "rationale": {"type": "string", "description": "One sentence: why delegate"},
        },
        "required": ["tasks"],
    }),
    |sup, _id, args| {
        let mut specs = args.get("tasks").cloned().unwrap_or(Value::Null);
        if let Value::String(text) = &specs {
            match serde_json::from_str(text) {
                Ok(v) => specs = v,
                Err(_) => return ToolResult::err("tasks must be a JSON array of task objects"),
            }
        }
        let list: Vec<Value> = specs
            .as_array()
            .map(|a| a.iter().filter(|s| s.is_object()).cloned().collect())
            .unwrap_or_default();
        if list.is_empty() {
            return ToolResult::err("tasks must be a non-empty array");
        }
        let r = sup.plan(&list, "main");
        ok(r.message, r.proposed)
    }
);

task_tool!(
    TaskStatusTool,
    "task_status",
    true,
    "Show the task board: every task's state, last checkpoint and open questions; with an id, that task in \
     detail (brief, checkpoints, outcome, transcript path).",
    json!({"type": "object", "properties": {"id": {"type": "string", "description": "Task id or agent name"}}, "required": []}),
    |sup, _id, args| {
        let r = s(&args, "id");
        ok(
            sup.status_text(Some(r.as_str()).filter(|x| !x.is_empty())),
            false,
        )
    }
);

task_tool!(
    TaskSendTool,
    "task_send",
    false,
    "Send a message to a subagent. mode=steer reaches it at its next step; follow_up waits until its current \
     turn ends. A message to a task that is not running is delivered when it (re)starts.",
    json!({
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Task id or agent name"},
            "text": {"type": "string"},
            "mode": {"type": "string", "enum": ["steer", "follow_up"]},
        },
        "required": ["id", "text"],
    }),
    |sup, _id, args| {
        let mode = if s(&args, "mode") == "follow_up" {
            "follow_up"
        } else {
            "steer"
        };
        ok(
            sup.send(&s(&args, "id"), &s(&args, "text"), mode, "main"),
            false,
        )
    }
);

task_tool!(
    AnswerTool,
    "answer",
    false,
    "Answer a subagent's question (the id from its <question> update). It resumes with the answer.",
    json!({"type": "object", "properties": {"question_id": {"type": "string"}, "text": {"type": "string"}}, "required": ["question_id", "text"]}),
    |sup, _id, args| {
        ok(
            sup.answer(&s(&args, "question_id"), &s(&args, "text")),
            false,
        )
    }
);

task_tool!(
    TaskControlTool,
    "task_control",
    false,
    "Control tasks on the user's behalf: pause, resume, cancel, retry (fresh=true starts a new transcript), \
     reprioritize (priority), edit (brief/depends_on of a task that has not started, or a correction steered \
     into a running one), approve / reject proposed tasks, merge (a finished worktree task's branch). id may be \
     'all' for pause/resume/cancel/approve.",
    json!({
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Task id, agent name, or 'all'"},
            "action": {"type": "string", "enum": ["pause", "resume", "cancel", "retry", "reprioritize", "edit", "approve", "reject", "merge"]},
            "note": {"type": "string", "description": "Reason, or what the subagent is told on resume/retry"},
            "priority": {"type": "integer"},
            "fresh": {"type": "boolean"},
            "brief": {"type": "string"},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "action"],
    }),
    |sup, _id, args| {
        let deps = args.get("depends_on").map(|d| str_list(Some(d)));
        let brief = args.get("brief").and_then(Value::as_str).map(String::from);
        ok(
            sup.control(
                &s(&args, "id"),
                &s(&args, "action"),
                &s(&args, "note"),
                args.get("priority").and_then(Value::as_i64),
                args.get("fresh").and_then(Value::as_bool).unwrap_or(false),
                brief,
                deps,
                "main",
            ),
            false,
        )
    }
);

pub fn main_tools(sup: Arc<Supervisor>) -> Vec<Arc<dyn Tool>> {
    let t = String::new;
    vec![
        Arc::new(PlanTasksTool {
            sup: sup.clone(),
            task_id: t(),
        }),
        Arc::new(TaskStatusTool {
            sup: sup.clone(),
            task_id: t(),
        }),
        Arc::new(TaskSendTool {
            sup: sup.clone(),
            task_id: t(),
        }),
        Arc::new(AnswerTool {
            sup: sup.clone(),
            task_id: t(),
        }),
        Arc::new(TaskControlTool { sup, task_id: t() }),
    ]
}

// -- subagent --------------------------------------------------------------------

task_tool!(
    CheckpointTool,
    "checkpoint",
    true,
    "Record progress on your task at each phase change (what is done, what is next). The main agent and the \
     user see it on the task board.",
    json!({
        "type": "object",
        "properties": {
            "phase": {"type": "string", "description": "Short phase name"},
            "summary": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "progress": {"type": "number", "description": "0..1"},
        },
        "required": ["phase"],
    }),
    |sup, id, args| {
        sup.add_checkpoint(
            id,
            &s(&args, "phase"),
            &s(&args, "summary"),
            str_list(args.get("files")),
            args.get("progress").and_then(Value::as_f64),
            false,
        );
        ok("已打点", false)
    }
);

task_tool!(
    NeedTool,
    "need",
    false,
    "You discovered that your task depends on other work. on=[task ids] waits for existing tasks; \
     new={title, brief, profile} requests work that does not exist yet. policy=wait suspends you until it is \
     done (you resume with its result); cancel_self drops your task as obsolete; ask_main asks the main agent \
     to decide. Ends your turn.",
    json!({
        "type": "object",
        "properties": {
            "on": {"type": "array", "items": {"type": "string"}},
            "new": {"type": "object", "properties": {
                "title": {"type": "string"}, "brief": {"type": "string"}, "profile": {"type": "string"},
                "scope_paths": {"type": "array", "items": {"type": "string"}},
            }},
            "reason": {"type": "string"},
            "policy": {"type": "string", "enum": ["wait", "cancel_self", "ask_main"]},
            "join": {"type": "string", "enum": ["all_settled", "fail_fast"]},
        },
        "required": ["reason"],
    }),
    |sup, id, args| {
        let new = match args.get("new") {
            Some(Value::String(t)) => Some(
                serde_json::from_str::<Value>(t)
                    .unwrap_or_else(|_| json!({"title": t, "brief": t})),
            ),
            Some(v) if v.is_object() => Some(v.clone()),
            _ => None,
        };
        let policy = Some(s(&args, "policy"))
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| "wait".into());
        ok(
            sup.child_need(
                id,
                str_list(args.get("on")),
                new.as_ref(),
                &s(&args, "reason"),
                &policy,
            ),
            true,
        )
    }
);

task_tool!(
    AskMainTool,
    "ask_main",
    false,
    "Ask the main agent (and through it the user) a question you cannot decide yourself. You are suspended \
     until it is answered. Ends your turn.",
    json!({"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}),
    |sup, id, args| ok(sup.child_ask(id, &s(&args, "question")), true)
);

task_tool!(
    FinishTool,
    "finish",
    false,
    "Deliver your task and stop. summary: what you did and how you verified it (the main agent reads only \
     this). status=failed when you could not complete it (say why). Ends your turn.",
    json!({
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "files_changed": {"type": "array", "items": {"type": "string"}},
            "follow_ups": {"type": "array", "items": {"type": "string"}},
            "status": {"type": "string", "enum": ["done", "failed"]},
        },
        "required": ["summary"],
    }),
    |sup, id, args| {
        ok(
            sup.finish(
                id,
                &s(&args, "summary"),
                str_list(args.get("files_changed")),
                str_list(args.get("follow_ups")),
                s(&args, "status") == "failed",
                "",
            ),
            true,
        )
    }
);

task_tool!(
    ProposeTaskTool,
    "propose_task",
    false,
    "Propose a new task you noticed but that is outside your scope (you cannot start subagents yourself). The \
     main agent decides. Does not suspend you.",
    json!({
        "type": "object",
        "properties": {"title": {"type": "string"}, "brief": {"type": "string"}, "profile": {"type": "string"}},
        "required": ["title", "brief"],
    }),
    |sup, id, args| {
        let profile = Some(s(&args, "profile"))
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| "worker".into());
        ok(
            sup.child_propose(id, &s(&args, "title"), &s(&args, "brief"), &profile),
            false,
        )
    }
);

pub fn subagent_tools(sup: Arc<Supervisor>, task_id: &str) -> Vec<Arc<dyn Tool>> {
    let t = || task_id.to_string();
    vec![
        Arc::new(CheckpointTool {
            sup: sup.clone(),
            task_id: t(),
        }),
        Arc::new(NeedTool {
            sup: sup.clone(),
            task_id: t(),
        }),
        Arc::new(AskMainTool {
            sup: sup.clone(),
            task_id: t(),
        }),
        Arc::new(FinishTool {
            sup: sup.clone(),
            task_id: t(),
        }),
        Arc::new(ProposeTaskTool { sup, task_id: t() }),
    ]
}
