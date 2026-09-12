# Ubiquitous language

This glossary gives contributors, users, and agents one shared vocabulary. Use these terms consistently in code, CLI output, issues, and documentation.

## Core terms

| Term | Meaning | Not the same as |
| --- | --- | --- |
| **Goal** | The human-readable outcome a user asks Aculptoi to work toward, such as “create a simple creature.” | An action plan or a Blender object name. |
| **Actor** | The planning role. It turns a goal, scene inspection, and prior critique into a proposed structured action plan. | The Blender worker or the vision critic. |
| **Vision critic** | The read-only role that evaluates rendered views and returns a score, observations, issues, and suggestions. | An executor; it cannot change Blender. |
| **Model provider** | A reusable software adapter that talks to one named model endpoint. One or both application roles may select it. | A model weight file or a particular model family. |
| **Model endpoint** | A running HTTP service that accepts model requests, for example a local llama.cpp server at `http://localhost:8080/v1`. | The Aculptoi Blender worker. |
| **Local runtime launcher** | The optional user-invoked `aculptoi model serve` helper that starts llama.cpp from user-supplied artifacts under `HF_HOME`. | A model provider or an actor capability. |
| **Model weights** | The large learned files used by a model, often `.gguf` files. They are supplied and run by the user, not included in this repository. | The Python files in `src/aculptoi/models/`. |
| **Action plan** | A validated actor response containing a reason and one or more typed actions. | Arbitrary code or shell commands. |
| **Action** | One allowlisted, typed scene mutation such as `object.create` or `object.scale`. | A raw `bpy` expression. |
| **Harness** | The orchestration layer that runs the bounded actor → worker → render → critic loop and persists artifacts. | A heavy agent framework. |
| **Blender worker** | The persistent local Blender process and its narrow HTTP interface. It is the only component allowed to mutate the Blender scene. | The CLI process. |
| **Scene inspection** | A structured description of the current Blender scene or one object. | A render or a visual critique. |
| **Inspection render** | A PNG rendered from a known viewpoint to provide visual evidence to a person or vision critic. | A final production render. |
| **Run** | One bounded refinement session. It owns an incrementing directory under `.aculptoi/runs/`. | A checkpoint. |
| **Iteration** | One pass through the refinement loop within a run. | A Blender undo step. |
| **Checkpoint** | A recoverable Blender `.blend` snapshot plus associated metadata. | A render or an autosave file. |
| **Artifact** | An inspectable output from a run: plan JSON, action record, critique JSON, render PNG, log, or checkpoint metadata. | A tracked source file. |

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
2. The **actor**, through its configured **model provider**, proposes an **action plan**.
3. The **harness** validates the plan and sends the typed **actions** to the **Blender worker**.
4. The worker changes the scene and creates **inspection renders**.
5. The **vision critic**, through its separate provider, returns a read-only **critique**.
6. The harness records the iteration's **artifacts** and a **checkpoint** before deciding whether another iteration is needed.

## Naming rules

- Say **actor** and **vision critic** for roles; say **model** only when referring to the underlying AI model or its configured identifier. A shared **provider** does not make the two roles one role.
- Say **model provider** for code that calls an endpoint; never call it “the model” when that distinction matters.
- Say **action** for a schema-validated, allowlisted mutation; never use “command” to imply arbitrary shell execution.
- Say **worker** for the persistent Blender-side service and **CLI** for the user-facing process.
- Say **checkpoint** for recovery state and **artifact** for a recorded output.
