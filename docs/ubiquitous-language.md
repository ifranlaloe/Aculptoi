# Ubiquitous language

This glossary gives contributors, users, and agents one shared vocabulary. Use these terms consistently in code, CLI output, issues, and documentation.

## Core terms

| Term | Meaning | Not the same as |
| --- | --- | --- |
| **Goal** | The human-readable outcome a user asks Aculptoi to work toward, such as “create a simple creature.” | An action plan or a Blender object name. |
| **Actor** | The planning role. It first turns a goal, scene inspection, and prior critique into a construction plan, then proposes actions for one active construction item at a time. | The Blender worker or the vision critic. |
| **Vision critic** | The read-only role that evaluates rendered views and returns a score, observations, issues, and suggestions. | An executor; it cannot change Blender. |
| **Model provider** | A reusable software adapter that talks to one named model endpoint. One or both application roles may select it. | A model weight file or a particular model family. |
| **Model endpoint** | A running HTTP service that accepts model requests, for example a local llama.cpp server at `http://localhost:8080/v1`. | The Aculptoi Blender worker. |
| **Local runtime launcher** | The optional user-invoked `aculptoi model serve` helper that starts llama.cpp from user-supplied artifacts under `HF_HOME`. | A model provider or an actor capability. |
| **Model weights** | The large learned files used by a model, often `.gguf` files. They are supplied and run by the user, not included in this repository. | The Python files in `src/aculptoi/models/`. |
| **Role prompt template** | A versioned Markdown instruction file for one application role, loaded from `src/aculptoi/agent/prompt_templates/`. | A shared conversation history or executable skill. |
| **Construction plan** | The immutable, ordered Actor response created at the start of every visual-refinement iteration. It contains construction items and dependencies but no executable actions. | An action batch or hidden model task list. |
| **Construction item** | One named, bounded component or feature in a descriptive construction plan, with an objective and earlier dependencies. | A Blender object; one item may affect several objects or take several action batches. |
| **Completion criteria** | A non-empty, immutable list of independently checkable conditions created by the work-item Actor in its first response for an item. | A construction-plan field or a visual-critic score. |
| **Work-item status** | The Actor's explicit `continue` or `complete` decision for the active construction item after its proposed actions, evaluated against the item's completion criteria. | A visual-critic score or an implicit guess by the harness. |
| **Action batch** | One validated Actor response for the active construction item, containing its status, reason, and up to 25 typed actions. The first batch also creates the item's completion criteria. | A complete construction item or visual-refinement iteration. |
| **Action** | One allowlisted, typed scene mutation such as `object.create` or `object.scale`. | A raw `bpy` expression. |
| **Harness** | The orchestration layer that runs the bounded actor → worker → render → critic loop and persists artifacts. | A heavy agent framework. |
| **Blender worker** | The persistent local Blender process and its narrow HTTP interface. It is the only component allowed to mutate the Blender scene. | The CLI process. |
| **Scene inspection** | A structured description of the current Blender scene or one object. | A render or a visual critique. |
| **Inspection render** | A PNG rendered from a known viewpoint to provide visual evidence to a person or vision critic. | A final production render. |
| **Run** | One bounded refinement session. It owns an incrementing directory under `.aculptoi/runs/`. | A checkpoint. |
| **Execution batch** | The non-empty action list from one action batch sent to the Blender worker and counted in the stable CLI result. | The surrounding Actor response, a construction item, or a visual-refinement iteration. |
| **Inspection milestone** | The point after all construction-plan items complete where Aculptoi renders multiple views and asks the Vision Critic for feedback. | Every individual Blender mutation. |
| **Visual-refinement iteration** | The complete feedback cycle: immutable construction planning, ordered work-item execution, multi-view renders, Vision Critic analysis, and checkpointing. This is the meaning of `iteration-XXX` in run artifacts. | A construction item, action batch, or execution batch. |
| **Checkpoint** | A recoverable Blender `.blend` snapshot plus associated metadata. | A render or an autosave file. |
| **Artifact** | An inspectable output from a run: exact user prompt, role prompt, plan JSON, action record, critique JSON, render PNG, iteration `.blend` copy, raw failed model response, log, or checkpoint metadata. | A tracked source file. |

## The `models` directory

`src/aculptoi/models/` contains **Python integration code**, not AI models or model weights:

| File | Responsibility |
| --- | --- |
| `base.py` | Defines the small `ModelProvider` interface and model-provider error type. |
| `openai_compatible.py` | Implements that interface for local OpenAI-compatible HTTP servers, including llama.cpp. |
| `__init__.py` | Exposes the provider types used elsewhere in Aculptoi. |

This separation lets Aculptoi support multiple local runtimes without coupling its actor or vision logic to one model family. The Actor and Vision Critic select named provider configurations. They can share one provider instance and local HTTP client while keeping their role-specific prompts, requests, schemas, and permissions separate.

Model weights are intentionally outside the repository. By default, a top-level `models/` directory and common weight formats (`.gguf`, `.ggml`) are ignored by Git, so users can keep local weights beside a checkout without accidentally committing them.

## Example: following one request

For this command:

```bash
aculptoi run "create a simple creature"
```

the intended language is:

1. The quoted text is the **goal**.
2. The **actor**, through its configured **model provider**, proposes a descriptive **construction plan**.
3. The harness persists that immutable plan and activates its first **construction item**.
4. The actor's first **action batch** creates the item's immutable **completion criteria** and reports a **work-item status**.
5. The **harness** validates its typed **actions** and sends any non-empty list to the **Blender worker** as an **execution batch**.
6. The harness records the response, worker result, and checkpoint under the active item, then continues that item or advances to the next.
7. At the **inspection milestone**, the worker creates **inspection renders**.
8. The **vision critic**, through its separate provider, returns a read-only **critique**.
9. The harness records the **visual-refinement iteration's** summary and final **checkpoint** before deciding whether another iteration is needed.

## Naming rules

- Say **actor** and **vision critic** for roles; say **model** only when referring to the underlying AI model or its configured identifier. A shared **provider** does not make the two roles one role.
- Say **model provider** for code that calls an endpoint; never call it “the model” when that distinction matters.
- Say **action** for a schema-validated, allowlisted mutation; never use “command” to imply arbitrary shell execution.
- Say **worker** for the persistent Blender-side service and **CLI** for the user-facing process.
- Say **checkpoint** for recovery state and **artifact** for a recorded output.
- Say **construction plan** for the immutable descriptive iteration-level breakdown, **construction item** for one semantic unit, **completion criteria** for the item-level artifact first created by its Actor, **action batch** for one item-scoped Actor response, and **execution batch** only when a non-empty action list is submitted to the worker.
- Use **visual-refinement iteration** only for a completed plan → work items → visual critique feedback cycle. Do not use “iteration” by itself when the distinction matters.
