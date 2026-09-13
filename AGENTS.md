# AGENTS.md

## Project

Aculptoi is a local-first autonomous 3D agent for Blender.

Its core loop is deliberately explicit:

```text
goal → construction plan → construction items → validated actions → Blender worker → multi-view renders → issue discovery → focused issue analysis → assembled critique → refinement
```

The project is early-stage. Do not document functionality as implemented until it has been implemented and verified.

## Architectural invariants

- Aculptoi is CLI-first and must remain usable without MCP.
- MCP may be added only as an optional adapter; core architecture must not depend on it.
- Actor and Vision Critic are separate application roles with independent prompts, request construction, schemas, responsibilities, and permissions. They may select the same provider, endpoint, and multimodal model, or separate providers.
- The vision critic is read-only. It must never mutate Blender or invoke scene operations.
- The Actor is the planning/reasoning role. It may use bounded model reasoning, but it emits only typed structured actions—never shell commands.
- Blender mutations happen only through the Blender worker.
- Validate actions at both the harness and worker boundaries.
- Do not add arbitrary `exec`, shell, Python, or `bpy` execution paths.
- The Blender worker listens on localhost by default and must reject non-loopback binding.
- Keep orchestration explicit, bounded, inspectable, and deterministic where practical.
- Do not introduce a heavy agent framework unless its benefit clearly outweighs the new complexity.

## Repository responsibilities

- `src/aculptoi/agent/`
  Actor planning, vision-critic integration, and the refinement state machine.

- `src/aculptoi/blender/`
  Versioned worker transport, client behavior, and Blender-worker error handling.

- `src/aculptoi/schemas/`
  Typed action-plan and critique contracts. These are the primary harness validation boundary.

- `blender/aculptoi_worker.py`
  Standalone Blender-side HTTP worker. Keep it dependency-light, route-based, and allowlisted.

- `src/aculptoi/models/`
  Model-provider interfaces and OpenAI-compatible local HTTP implementation.

- `src/aculptoi/runtime/`
  Optional, user-invoked local runtime helpers. These must use fixed argument
  vectors and never accept model-generated process instructions. The llama.cpp
  launcher must require `HF_HOME`; it must not silently select a model-cache path.

- `src/aculptoi/config/`
  Local-first TOML configuration and safe defaults.

- `src/aculptoi/checkpoints/`
  Inspectable run artifacts and checkpoint metadata persistence.

- `tests/`
  Fast unit tests for schemas, providers, orchestration, and persistence. Add Blender integration tests as worker behavior grows.

## Blender worker rules

`blender/aculptoi_worker.py` runs inside Blender's Python environment. Blender APIs must execute on Blender's main thread; do not reintroduce threaded request handling for `bpy` operations.

- Avoid normal project dependencies unless absolutely necessary.
- Expose explicit, versioned routes and a small action allowlist.
- Reject unknown actions and invalid arguments independently of client validation.
- Constrain worker-originated file writes to the project `.aculptoi/` artifact directory.
- Never expose arbitrary shell, Python, `bpy`, or filesystem execution by default.
- Preserve useful error messages without returning internal traces to clients.
- Keep checkpoint/recovery behavior explicit when introducing mutations with meaningful side effects.

## Model rules

Models are untrusted inputs. Never assume an actor response is valid.

Actor output must:

1. Conform to a typed schema.
2. Use only allowlisted actions.
3. Pass harness validation.
4. Pass Blender-worker validation.

The vision critic may return observations, scores, issues, and suggested changes only. It does not gain execution capability merely because an actor consumes its output later. It is read-only even when it shares weights or an HTTP client with the Actor. Its complete-view discovery pass creates immutable issue summaries; focused issue analysis may enrich exactly one known summary but must not replace its ID, title, region, severity, or evidence views or discover unrelated issues.

Keep role prompts in `src/aculptoi/agent/prompt_templates/` as versioned Markdown files; `src/aculptoi/agent/prompts.py` only loads and validates their version markers. The Actor receives structured text state, never renders by default. The Vision Critic receives prepared render images and never receives a Blender client or action executor. Maintain separate `vision_issue_discovery.md` and `vision_issue_analysis.md` templates. The harness—not a model prompt—owns bounded issue-analysis selection, preserves skipped summaries, and isolates malformed issue responses.

The first Actor response in every visual-refinement iteration is a typed,
planning-only, descriptive **construction plan**. It contains items, objectives, and
dependencies, but no completion criteria or actions. Persist it immutably before
requesting actions. Process its ordered **construction items** one at a time. The
first work-item Actor response must target the active item and create a non-empty,
immutable completion-criteria list. Each response may contain no more than 25 typed
actions and must explicitly report `continue` or `complete`; later responses must use,
not replace, the established criteria. Persist every prompt, response, worker result,
and checkpoint under that item. Subsequent item requests must receive a fresh scene
inspection plus compact completed-item context that traces the object names earlier
items created or affected. Invoke the read-only Vision Critic only after all plan items
complete.

Do not use a normal per-item or per-iteration batch cap to drive completion. Completion
is semantic and comes from work-item status. The harness must still enforce global,
operator-configured Actor-request, action-count, and wall-clock safety budgets, stopping
before further mutation when a budget is exhausted.

Preserve raw failed model responses as local run artifacts for debugging, but never parse, replay, or execute them outside the normal typed validation path. Treat them as potentially sensitive diagnostic data.

Do not add model-specific behavior to core contracts. llama.cpp's OpenAI-compatible API is a first-class target, not a mandatory runtime or model family.

## CLI rules

The public CLI is a stable interface. Prefer noun–verb forms such as:

```text
aculptoi scene inspect
aculptoi object list
aculptoi render views
```

- Operational commands should support `--json` where appropriate.
- Do not silently break existing JSON output schemas.
- Keep user-facing errors actionable and safe to show in structured output.
- Avoid making a daemon mandatory for future batch/headless modes.

## New action checklist

Do not expand the action surface merely because Blender supports an operation. A new Blender action requires:

- a clear typed schema;
- validation at both boundaries;
- bounded, documented side effects;
- unit tests and, where feasible, Blender integration coverage;
- checkpoint/undo implications considered; and
- README, architecture, and security updates when its capability changes the trust boundary.

Prefer a narrow typed operation over a generic code-execution escape hatch.

## Development workflow

Install development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Before finishing a change, run:

```bash
ruff check .
ruff format --check .
pytest
mypy src
```

Run an appropriate headless Blender-worker smoke test when changing worker routes, actions, rendering, checkpointing, or the transport. Add or update tests for every behavioral change.

## Change philosophy

Prefer:

- small modules;
- typed interfaces;
- explicit state;
- deterministic behavior;
- inspectable artifacts;
- clear failure modes; and
- minimal dependencies.

Avoid:

- hidden global state;
- giant agent prompts containing application logic;
- model-specific assumptions in core code;
- opaque autonomous behavior; and
- scope expansion without a corresponding safety design.

## Documentation

Use [docs/ubiquitous-language.md](docs/ubiquitous-language.md) as the canonical project vocabulary. In particular, preserve the distinction between model providers (Python integration code), model endpoints (running services), and model weights (user-supplied local files).

If an architectural invariant changes, update the relevant documents:

- `README.md`
- `docs/architecture.md`
- `docs/ubiquitous-language.md` when project terminology changes
- `SECURITY.md` when a trust boundary changes
- this `AGENTS.md`
