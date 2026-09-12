# Architecture

Aculptoi keeps orchestration separate from scene execution. The Actor and Vision Critic are separate roles even when they share a single multimodal provider, model endpoint, and set of weights.

```mermaid
flowchart LR
    M[llama.cpp\nmultimodal model] --> A[Actor\ntext-only request]
    M --> V[Vision Critic\nmultimodal request]
    A -->|typed construction plan| H[Aculptoi harness]
    H -->|one current work item| A
    A -->|typed item action batch| H
    H --> W[Persistent Blender worker]
    W --> B[Blender]
    B --> R[Multi-view PNG renders]
    R --> V
    V -->|read-only structured critique| H
```

| Component | Responsibility | May mutate Blender? |
| --- | --- | --- |
| CLI | User intent, configuration, stable JSON output | No |
| Harness | Plan-first visual-refinement state machine, work-item scheduling, safety-budget enforcement, and artifact recording | Via worker only |
| Actor | First creates an ordered construction plan, then proposes typed actions for one active item at a time; requests are text-only by default | No |
| Blender client | Versioned local transport abstraction | Sends validated actions |
| Blender worker | Independently validates and performs V1 operations | Yes |
| Vision critic | Evaluates prepared PNG renders into structured, read-only feedback | No |
| Model provider | Reusable OpenAI-compatible endpoint client; may be selected by one or both roles | No |
| Checkpoint store | Writes inspectable metadata, captures failed model responses, and copies validated snapshots into run iterations | No scene mutation |

The HTTP transport is purposefully small and local-only in V1. A Unix socket or a headless batch transport can replace the client implementation without changing actor, critic, schemas, or loop semantics.

`[providers.<name>]` defines an endpoint once. `[actor]` and `[vision]` each select it by name. The default `local` provider is selected by both roles, so one llama.cpp multimodal server is enough. Selecting different provider names preserves the two-endpoint topology. Sharing a provider only shares its reusable HTTP client; it does not create shared conversation history, grant the critic mutation authority, or merge the role prompts or schemas.

The critic prepares PNG copies in memory and constrains their longest dimension before placing them in OpenAI-compatible image data URLs. Stored inspection renders remain the original worker outputs.

Each run begins with `user-prompt.txt`, containing the exact human goal. Each
visual-refinement iteration has its own `iteration-XXX/` directory. Its first model
response is a planning-only, typed construction plan, persisted immutably as
`construction-plan.json`. It is descriptive: it names ordered items, objectives, and
dependencies but does not define actions or completion criteria. The harness processes
that plan in order. Each construction item has an `items/NNN-id/` directory containing
its immutable definition, the work-item Actor's immutable `completion-criteria.json`,
every stateless Actor prompt, validated action batch, worker result, checkpoint
metadata, and `.blend` copy. The Actor creates the completion criteria in its first
response for the item, then explicitly returns `continue` or `complete` against those
criteria. Every item request also includes a fresh full scene inspection, the immutable
construction plan, and compact summaries of completed items. The scene is the source of
truth for current object state; summaries preserve semantic lineage and trace the object
names created or affected by earlier items. Only after every item is complete does the
harness render and invoke the read-only Vision Critic.

Normal progress has no fixed per-item or per-iteration action-batch count. Operator
configured Actor-request, action-count, and wall-clock budgets remain global safety
backstops. If one is exhausted, the harness records `budget-exhausted.json` and stops
before another scene mutation. The central `.aculptoi/checkpoints/` location remains
the recovery source used by `checkpoint restore`. On model-response parsing or schema
failure, the corresponding plan or item retains an error JSON artifact and raw response
text for local debugging; that diagnostic data never gains execution authority.

Role instructions are versioned Markdown templates under
`src/aculptoi/agent/prompt_templates/`. Construction planning and work-item execution
have separate Actor templates and schemas, but remain one planning role with no direct
mutation authority. The small Python loader validates template version markers and
supplies the content to requests; it does not contain role-instruction prose.

`blender/aculptoi_worker.py` is standalone so it can execute inside Blender's Python environment without requiring the project's normal Python dependencies. It must remain an explicit-route, allowlisted server.
