"""Code Mode presentation helpers (deepseek-harness parity).

When Code Mode is on, the model-facing tool catalog collapses to a single
``run_code`` tool, and this module renders the SDK section that tells the
model how to reach every other tool from inside a program. The SDK is
derived from the registry — never hand-maintained — so the declared
methods always match what the model can actually call (a tool disabled by
``bash_enabled = false`` never appears).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from limbo.tools.registry import ToolRegistry

# The model-facing statement of the collapse: names the consequence (a
# direct call to any other tool fails) and the route (inside the program),
# because a rule the model can only discover by being denied is one it
# corrects too late.
CODE_ONLY_INSTRUCTION = (
    "`run_code` is the only tool you can call directly — a tool call naming "
    "any other tool fails. Reach every tool the SDK declares below from "
    "inside the program."
)

# The run_code usage instructions, kept in one place so the SDK section and
# the tool description cannot drift.
RUN_CODE_INSTRUCTIONS = (
    "`run_code` takes two required arguments: `code` — the body of an async "
    "Python function (top-level `await` and `return` both work) — and "
    "`description`, a short summary of what the program does. At run time "
    "exactly two names are bound: `tools` and `ToolCallError`. Inside the "
    "program:\n\n"
    "- Call tools as `await tools.name(keyword=value)` (subscript access "
    "`await tools[\"name\"](...)` also works). Positional arguments map "
    "to the tool's declared parameter order (`await tools.read(\"src/a.py\")` "
    "== `path=\"src/a.py\"`), and a lone dict is spread as keyword "
    "arguments (`await tools.read({\"path\": \"src/a.py\"})`). Arguments "
    "are forwarded to the tool verbatim; every tool returns its text "
    "output as a `str`.\n"
    "- A FAILED tool call raises `ToolCallError`, whose `toolName` identifies "
    "the failed tool and whose message is human-readable — wrap in "
    "`try/except ToolCallError` to handle and continue.\n"
    "- Independent read-only calls MAY overlap under `asyncio.gather`; "
    "sequence dependent work with `await`.\n"
    "- Emit the run's answer with `print(...)` and/or a top-level "
    "`return <value>`. ONLY what you print and the returned value come "
    "back — intermediate tool results never enter the conversation, so "
    "extract just what you need."
)

# JSON-schema type name → Python annotation.
_TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}


def _py_type(schema: dict) -> str:
    """Best-effort JSON schema → Python annotation (unknown → ``Any``)."""
    if "anyOf" in schema or "oneOf" in schema:
        return "Any"
    return _TYPE_MAP.get(schema.get("type", ""), "Any")


def _parameter_list(parameters: dict) -> str:
    """Render a tool's JSON-schema parameters as a keyword-argument list.

    Required parameters come first without a default; optional ones get
    ``= None`` so the SDK reads like a callable signature.
    """
    props = parameters.get("properties", {}) or {}
    required = set(parameters.get("required", []) or [])
    parts = []
    for name, prop in props.items():
        annotation = _py_type(prop)
        if name in required:
            parts.append(f"{name}: {annotation}")
        else:
            parts.append(f"{name}: {annotation} = None")
    return ", ".join(parts)


def _first_sentence(description: str) -> str:
    """The summary line for a tool: its description's first sentence."""
    return description.split(". ")[0].rstrip(".")


def render_tools_sdk(registry: ToolRegistry) -> str:
    """Render the SDK section for the registry's visible tools.

    Deterministic: tools render in registration order (stable), excluding
    ``run_code`` itself — the program must not be able to nest runs.
    """
    lines = [
        "## Writing code for run_code",
        "",
        RUN_CODE_INSTRUCTIONS,
        "",
        "The available tools (all return text as `str`, failure raises "
        "`ToolCallError`):",
        "",
    ]
    for definition in registry.all_definitions():
        function = definition["function"]
        name = function["name"]
        params = _parameter_list(function["parameters"])
        lines.append(f"- `tools.{name}({params})` — {_first_sentence(function['description'])}")
    return "\n".join(lines)
