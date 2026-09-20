# Ubiquitous language

This glossary gives contributors, users, and agents one shared vocabulary. Use these terms consistently in code, CLI output, issues, and documentation.

## Core terms

| Term | Meaning | Not the same as |
| --- | --- | --- |
| **Goal** | The human-readable outcome a user asks Aculptoi to work toward, such as “create a simple creature.” | An action plan or a Blender object name. |
| **Target Brief** | The bounded, durable interpretation aid derived once for a new run from its authoritative Goal. It records transferable visual priorities, explicit constraints, non-goals, and form traits. | A replacement for the Goal, a subject recipe, or an executable plan. |
| **Actor** | The planning role. On a new run it derives the Target Brief, then turns the Goal, scene inspection, prior critique, and (when available) one current viewport observation into a construction plan and Modeling Steps for one active construction item at a time. | The Blender worker or the vision critic. |
| **Inspection subsystem** | The bounded observational system that selects cameras, renders isolated evidence, composes an atlas, and obtains technical acceptance before critique. | A modeling or artistic-lighting tool. |
| **Camera Manager** | The deterministic part of the Inspection subsystem that generates candidates, retains canonical anchors, selects dynamic views, and derives framing from geometry bounds. | A model-driven camera controller. |
| **Inspection Reviewer** | The read-only Vision role that judges technical evidence quality and returns `accept`, `augment`, or `retry`. | The Critic; it does not judge whether the scene meets the goal. |
| **Inspection atlas** | One labeled combined image of selected views of the unchanged scene. | An unrelated collection of independent images. |
| **Atlas manifest** | Typed metadata mapping tile IDs to source shots, pixel bounds, orientation, projection, selection rationale, framing, and sensor versions. | Model-generated Blender coordinates. |
| **Canonical anchor view** | A retained world-space reference view: front (-Y), right (+X), rear (+Y), or front-upper. | A semantic claim about object anatomy or orientation. |
| **Dynamic inspection view** | A non-anchor camera selected deterministically for estimated novel coverage or silhouette information. | A camera proposed by a model. |
| **Inspection lighting rig** | The isolated neutral-studio setup used only for evidence acquisition. | User-authored scene lighting. |
| **Sensor version** | A versioned description of interpretation-relevant camera, atlas, framing, and lighting behavior. | A model-provider or model-weight version. |
| **Vision critic** | The read-only role that discovers visible issues from an accepted atlas and analyzes selected issues in depth before the harness assembles feedback. | An executor; it cannot change Blender. |
| **Model provider** | A reusable software adapter that talks to one named model endpoint. The Actor, Inspection Reviewer, Critic Discovery, and Critic Issue Analysis may all select it without becoming one logical role. | A model weight file or a particular model family. |
| **Model endpoint** | A running HTTP service that accepts model requests, for example a local llama.cpp server at `http://localhost:8080/v1`. | The Aculptoi Blender worker. |
| **Local runtime launcher** | The optional user-invoked `aculptoi model serve` helper that starts llama.cpp from user-supplied artifacts under `HF_HOME`. | A model provider or an actor capability. |
| **Model weights** | The large learned files used by a model, often `.gguf` files. They are supplied and run by the user, not included in this repository. | The Python files in `src/aculptoi/models/`. |
| **Reasoning effort** | A role-specific semantic setting (`low`, `medium`, `high`, or `xhigh`) forwarded by a provider using that endpoint's native mechanism. | A fixed count of hidden reasoning tokens or the output-token ceiling. |
| **Inference profile** | The thinking enablement, optional reasoning-effort, and maximum-output-token settings applied to one logical model stage. | A provider endpoint, an image-dimension setting, or a shared Vision-wide budget. |
| **Role prompt template** | A versioned Markdown instruction file for one application role, loaded from `src/aculptoi/agent/prompt_templates/`. | A shared conversation history or executable skill. |
| **Construction plan** | The immutable, ordered Actor response created at the start of every visual-refinement iteration. It contains construction items and dependencies but no executable actions. | An action batch or hidden model task list. |
| **Construction item** | One named, bounded component or feature in a descriptive construction plan, with an objective and earlier dependencies. | A Blender object; one item may affect several objects or take several Modeling Steps. |
| **Completion criteria** | A non-empty, immutable list of independently checkable conditions created by the work-item Actor in its first response for an item. | A construction-plan field or a visual-critic score. |
| **Modeling Step** | One bounded, transactional, human-meaningful Actor mutation toward an active item's completion criteria. It has one semantic intent and one to 25 typed actions. | A whole construction item, arbitrary “do everything” response, or persistent model session. |
| **Modeling Step budget** | The per-work-item ceiling on accepted Modeling Step attempts. A step counts when it reaches the execution path, including a recoverably rolled-back worker failure. | A wall-clock deadline, an Actor request budget, or an action-count budget. |
| **Observation request** | A bounded, typed Actor request for another semantic view from Aculptoi's observation viewport. It has no Blender mutations. | A scene action, Critic camera plan, or generic Blender UI command. |
| **Actor viewport observation** | One current, transient image plus metadata from Aculptoi's reserved `VIEW_3D` sensor. | A screenshot archive, user desktop capture, or Inspection atlas tile. |
| **Work-item response** | One Actor response of kind `modeling_step`, `observation_request`, or `complete`. The first response also creates immutable completion criteria. | A complete construction item or visual-refinement iteration. |
| **Action** | One allowlisted, typed scene mutation such as `object.create`, `object.join`, or a bounded mesh-region operation. | A raw `bpy` expression. |
| **Normalized mesh region** | An inclusive local-space mesh AABB whose `min` and `max` coordinates are normalized to `[-1, 1]` against current bounds at the start of an action. | Raw vertex or face identifiers supplied by a model. |
| **Harness** | The orchestration layer that runs the bounded actor → worker → render → critic loop and persists artifacts. | A heavy agent framework. |
| **Blender worker** | The persistent local Blender process and its narrow HTTP interface. It is the only component allowed to mutate the Blender scene. | The CLI process. |
| **Scene inspection** | A structured description of the current Blender scene or one object. | A render or a visual critique. |
| **Inspection render** | A durable full-quality source PNG for one selected camera in an inspection round. | A final production render or a temporary candidate diagnostic. |
| **Run** | One bounded refinement session. It owns an incrementing directory under `.aculptoi/runs/`. | A checkpoint. |
| **Run telemetry** | The append-only `run-events.jsonl` record of lifecycle and expensive-stage observations, timing, profile, errors, and provider usage when available. | Recovery state; telemetry is never read to decide resume behavior. |
| **Canonical scene** | The mutable `scene.blend` at a run root, continuously saved by that run's attached Blender worker after every successful Modeling Step. | An immutable checkpoint or an iteration artifact. |
| **Active work item** | The one item currently executing. Its partial changes may appear in the canonical scene, but are disposable after interruption. | A durable work item. |
| **Durable work item** | A completed work item whose saved canonical scene was successfully copied to an immutable run-local checkpoint and recorded in `run-state.json`. | An Actor response that merely says `complete`. |
| **Checkpoint** | An immutable run-local `.blend` copy created only at a completed-work-item boundary. | The mutable canonical scene, an autosave, or an arbitrary manual snapshot. |
| **Observer Mode** | The visible, worker-controlled Blender UI for viewing and navigating the live canonical scene while normal selection/editing is restricted to prevent accidents. | A second Blender viewer or a security boundary. |
| **Headless Mode** | The same worker and persistence model running in Blender background mode. | A second orchestration path. |
| **Execution batch** | The non-empty action list from one Modeling Step sent to the Blender worker and counted in the stable CLI result. | The surrounding Actor response, a construction item, or a visual-refinement iteration. |
| **Inspection round** | One independently persisted camera plan, source-shot set, atlas, and technical review inside the bounded Inspection subsystem. | A failed attempt; later rounds may add valid evidence. |
| **Inspection milestone** | The point after all construction-plan items complete where Aculptoi prepares and technically validates an inspection atlas. | Every individual Blender mutation. |
| **Issue discovery** | The complete-atlas Critic pass: the model returns compact tuples, then the harness validates them and creates an immutable domain inventory with identity, severity, confidence, and evidence tiles. | A detailed correction plan or an Actor action. |
| **Issue analysis** | One focused, read-only Critic pass that enriches exactly one discovered issue while retaining the complete atlas for comparison. | A new discovery pass; it must not create unrelated issues. |
| **Critic wire format** | The compact JSON contract used only between the model provider and Critic validator: discovery tuples or short detail keys with integer percentages and coded enums. | The rich domain model, an Actor input, or a persisted normal artifact. |
| **Critic domain model** | The expanded typed issue and detail representation with descriptive names, deterministic issue IDs, decimal confidence, and readable summaries. | Raw model output. |
| **Issue summary** | One immutable observation from issue discovery. It remains available even when detailed analysis is skipped or fails. | A detailed issue analysis. |
| **Assembled critique** | The final structured critique that combines discovery summaries with available focused details and failure markers; it is the only critique passed to the Actor. | A raw Critic conversation or executable scene plan. |
| **Visual-refinement iteration** | The complete feedback cycle: immutable construction planning, ordered work-item execution, inspection rounds, Critic analysis, and checkpointing. This is the meaning of `iteration-XXX` in run artifacts. | A construction item, action batch, or execution batch. |
| **Artifact** | An inspectable output from a run: exact user prompt, role prompt, plan JSON, action record, critique JSON, render PNG, raw failed model response, log, or checkpoint metadata. | A tracked source file. |

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
4. The actor's first **work-item response** creates the item's immutable **completion criteria**, then returns a **Modeling Step**, an **observation request**, or completion.
5. The **harness** validates a Modeling Step's typed **actions** and sends its non-empty list to the **Blender worker** as an **execution batch**; it supplies a fresh transient viewport observation after successful steps in UI mode.
6. The harness records the response and worker result, then saves the run's canonical scene. On item completion it creates the immutable checkpoint and only then marks the item durable.
7. At the **inspection milestone**, the **Camera Manager** creates an **inspection atlas** from selected **inspection renders**, then the **Inspection Reviewer** accepts, augments, or retries it within the inspection budget.
8. The **vision critic** completes atlas-based **issue discovery**, then performs bounded, focused **issue analysis** requests selected by the harness.
9. The harness creates the read-only **assembled critique** from immutable summaries and any available details.
10. The harness records the **visual-refinement iteration's** summary before deciding whether another iteration is needed; durable checkpoints remain tied to item boundaries.

## Naming rules

- Say **Actor**, **Inspection Reviewer**, **Critic Discovery**, and **Critic Issue Analysis** for logical roles; say **model** only when referring to the underlying AI model or its configured identifier. A shared **provider** does not make those roles one role.
- Say **model provider** for code that calls an endpoint; never call it “the model” when that distinction matters.
- Say **action** for a schema-validated, allowlisted mutation; never use “command” to imply arbitrary shell execution.
- Say **worker** for the persistent Blender-side service and **CLI** for the user-facing process.
- Say **canonical scene** for mutable current run state, **checkpoint** for immutable completed-item recovery state, and **artifact** for a recorded explanation/output.
- Say **active work item** for disposable partial progress and **durable work item** only after its checkpoint exists and is recorded.
- Say **Observer Mode** for the worker's visible Blender UI and **Headless Mode** for the same worker without UI. Observer viewport navigation never determines Critic camera views.
- Say **construction plan** for the immutable descriptive iteration-level breakdown, **construction item** for one semantic unit, **completion criteria** for the item-level artifact first created by its Actor, **Modeling Step** for one item-scoped mutation, and **execution batch** only when a non-empty action list is submitted to the worker. Use **observation request** for a view-only Actor turn. A **Modeling Step budget** limits accepted item-scoped mutation attempts rather than elapsed wall-clock time.
- Use **inspection atlas** for the labeled combined visual survey and **atlas manifest** for its machine-readable provenance. A **canonical anchor view** has a stable world-space convention; a **dynamic inspection view** is selected by Aculptoi, never by a model.
- Use **issue discovery** for the compact whole-atlas visual scan and **issue analysis** for one focused follow-up. An **issue summary** does not become a new issue when analysis disagrees; record the disagreement as analysis conflict.
- Use **Critic wire format** only for compact model input/output at the response boundary. Use **Critic domain model** for the Actor, harness state, and normal run artifacts.
- Use **visual-refinement iteration** only for a completed plan → work items → visual critique feedback cycle. Do not use “iteration” by itself when the distinction matters.
