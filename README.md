# Aculptoi

**A local-first autonomous 3D agent for Blender.**

> Local AI that sees, builds, and refines in Blender.

Aculptoi can build 3D objects, look at its own work from multiple viewpoints, critique the result using a local vision model, and iteratively refine the Blender scene. It is an early-stage, experimental project: V1 establishes the safe, inspectable architecture rather than attempting fully autonomous sculpting.

## Why Aculptoi?

Most 3D agents stop at generating code or issue opaque operations. Aculptoi is designed to close a local feedback loop: a planner proposes constrained scene changes, a persistent Blender worker performs them, renders provide evidence, and a separate vision model critiques the result. Every iteration produces files that a person can inspect.

It is CLI-first, Blender-native, model-agnostic, and local-first. [llama.cpp](https://github.com/ggml-org/llama.cpp)'s OpenAI-compatible server is a first-class target, but any compatible local endpoint can be configured. MCP is not required and is deliberately not in the core architecture.

## Status

V1 is a usable foundation. It includes:

- a typed, allowlisted action language and independent validation at the harness and worker boundaries;
- a persistent localhost-only Blender HTTP worker;
- scene/object inspection, four inspection renders, `.blend` checkpoints, and a bounded actor → Blender → vision loop;
- independent OpenAI-compatible actor and vision providers; and
- structured JSON output for operational commands.

It does **not** yet autonomously make sophisticated creatures, offer broad sculpt tooling, or provide an add-on UI. See [the roadmap](#roadmap).

## Architecture

```mermaid
flowchart TD
    U[User goal / Aculptoi CLI] --> H[Aculptoi harness\nsmall Python state machine]
    H --> A[Actor / planner\nOpenAI-compatible local endpoint]
    A -->|validated structured actions| H
    H --> W[Persistent Blender worker\n127.0.0.1 only]
    W --> B[Blender scene]
    B --> R[Multi-view PNG renders]
    R --> V[Vision critic\nindependent local endpoint]
    V -->|validated read-only critique| H
    H --> C[.aculptoi runs & checkpoints]
```

The harness is deliberately a small explicit state machine, not a heavy agent framework:

```text
inspect scene → actor plan → validate → execute → render views → vision critique → checkpoint
```

This separation makes an eventual LangGraph integration, REST service, or MCP adapter additive rather than foundational.

## Quick start

Requirements: Python 3.12+, Blender, and (for `run`) two OpenAI-compatible local model endpoints. No cloud keys are required.

```bash
git clone https://github.com/ifranlaloe/Aculptoi.git
cd Aculptoi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Checks Python, Blender, project access, worker, actor, and vision endpoints.
aculptoi doctor

# Starts one background Blender process; it stays up between commands.
aculptoi blender start
aculptoi scene inspect --json
aculptoi object list --json
```

On macOS the default Blender path is `/Applications/Blender.app/Contents/MacOS/Blender`. On Linux and Windows-compatible shells, set `blender.executable` in `aculptoi.toml` if `blender` is not on `PATH`.

### Local llama.cpp configuration

Start compatible local servers independently, for example one text/coder model for planning and one vision-language model for critique. Then create `aculptoi.toml` in the project root:

```toml
[actor]
base_url = "http://localhost:8080/v1"
model = "local-actor"

[vision]
base_url = "http://localhost:8081/v1"
model = "local-vision"

[blender]
host = "127.0.0.1"
port = 9876

max_iterations = 5
score_target = 0.9
```

Models and endpoint URLs are examples only—Aculptoi does not hard-code Qwen, llama.cpp, or any cloud provider. The actor and critic are separate configuration objects on purpose. The current vision client sends PNGs through OpenAI-compatible `image_url` data URLs, which is supported by vision-capable llama.cpp server builds.

## CLI

All commands that return operational data accept `--json`.

```bash
aculptoi doctor
aculptoi config show --json

aculptoi blender start
aculptoi blender status --json
aculptoi blender stop

aculptoi scene inspect --json
aculptoi object list --json
aculptoi object inspect Cube --json

aculptoi render views Dragon --views front,right,top,perspective --json

aculptoi checkpoint save iteration-12
aculptoi checkpoint list --json
aculptoi checkpoint restore iteration-11

aculptoi run "create a simple creature" --json
# `create` is a convenience alias for `run`.
aculptoi create "a western dragon"
# Restore the latest recorded checkpoint and continue its goal.
aculptoi refine
```

`run` requires the Blender worker and both local models to be running. It limits the number of iterations via `max_iterations`, uses deterministic (`temperature: 0`) model requests where supported, and creates artifacts under:

```text
.aculptoi/
├── checkpoints/
│   └── run-000001-iteration-001.blend
└── runs/
    └── 000001/
        ├── actor-plan-001.json
        ├── actions-001.json
        ├── critique-001.json
        ├── checkpoint-001.json
        └── iteration-001/
            ├── front.png
            ├── right.png
            ├── top.png
            └── perspective.png
```

The checkpoint metadata records iteration, timestamp, goal, plan, executed actions, scene snapshot reference, render paths, critique, and score across its associated JSON artifacts. These files are intentionally ignored by Git.

## V1 action boundary

The actor never receives shell access. Its responses must match Pydantic schemas, and the Blender worker repeats its own validation before it changes the scene. V1 accepts only:

| Family | Commands |
| --- | --- |
| Objects | `object.create`, `object.delete`, `object.translate`, `object.rotate`, `object.scale` |
| Sculpt | `sculpt.voxel_remesh` |

Scene inspection, object inspection, multi-view rendering, and checkpoint operations use dedicated worker routes—not arbitrary `bpy` strings. Unsupported commands, arbitrary file paths, shell commands, and Python/`bpy` snippets are rejected. `execute_bpy` is intentionally absent.

## Safety

Blender automation is code execution adjacent and should be treated carefully. The worker listens only on localhost by default, validates every request, constrains artifact paths to the current project's `.aculptoi/` directory, bounds request size, and uses checkpoint recovery. Keep the worker on a trusted machine and do not expose its port with a tunnel or port-forwarding rule.

Read the full [threat model](SECURITY.md) before connecting an untrusted model or project.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
mypy src
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance and [docs/architecture.md](docs/architecture.md) for component boundaries.

For the project’s shared terminology—including the distinction between model-provider code and local model weights—see [docs/ubiquitous-language.md](docs/ubiquitous-language.md).

## Roadmap

- [ ] Additional safe sculpt operations (smooth, inflate, grab) and modifiers
- [ ] Automatic creature construction and semantic scene planning
- [ ] Topology analysis, UV/material generation, rigging, and animation
- [ ] Visual progress scoring and adaptive camera selection
- [ ] Multiple actor models and model-routing policies
- [ ] Blender add-on UI
- [ ] Optional MCP adapter (not a core dependency)
- [ ] REST API
- [ ] Benchmark and evaluation suite
- [ ] Optional headless `--no-daemon` batch mode

## Contributing

Contributions are welcome, especially small improvements to safety, reproducibility, Blender testing, and documentation. Please open an issue before a large architectural change and keep the action boundary explicit. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
