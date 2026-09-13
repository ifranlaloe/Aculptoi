# Security policy and threat model

## Security posture

Aculptoi controls Blender, so it is designed to make the model-to-scene boundary narrow and inspectable. It is experimental software and is not a sandbox for malicious files, models, or Blender extensions.

The persistent worker binds only to `127.0.0.1` by default. Configuration rejects non-loopback worker hosts. Do not expose its port with a proxy, tunnel, container mapping, or firewall rule.

## Trust boundaries

1. **Actor model → harness.** The actor returns JSON only. Its first response in each iteration must match the planning-only construction-plan schema. Later responses are scoped to one known work-item ID, explicitly report `continue` or `complete`, and may contain at most 25 actions from the discriminated allowlist. Global Actor-request, action-count, and wall-clock budgets stop runaway iterations before further mutation. Arbitrary shell commands, Python, `bpy` expressions, filesystem paths, and unknown action names are not part of either schema.
2. **Harness → Blender worker.** The worker validates the allowlist again. It does not trust an HTTP client merely because it is local.
3. **Vision model → harness.** The vision critic has separate compact wire schemas for discovery and focused issue analysis. They are strictly validated and converted to rich domain models before use. The harness assigns discovery issue identity; analysis can enrich only one supplied issue and cannot alter its identity fields. The harness bounds discovery and focused-analysis request counts, preserves a failed issue as an untrusted summary-only observation, and never lets either Critic response invoke Blender operations.
4. **Worker → filesystem.** Render outputs and checkpoints must resolve under the current project's `.aculptoi/` directory. Model actions do not carry paths.
5. **Operator → local model runtime.** `aculptoi model serve` is an optional foreground launcher for an operator-selected `llama` executable and local model files. It requires an explicit `HF_HOME` cache root, uses a fixed argument vector, never invokes a shell, and binds only to `127.0.0.1`. It is not reachable by actor or critic output.
6. **Failed model response → run artifact.** Raw malformed model responses are written only under the project-local `.aculptoi/runs/` directory for debugging. They remain untrusted data, are ignored by Git, and must never be replayed as actions or executed as code. They may contain model reasoning or user-derived context.
7. **Construction plan → run artifacts.** The accepted construction plan is written once per iteration. Every later Actor prompt, action batch, worker result, and checkpoint is linked to its plan and active work item rather than relying on hidden model conversation state.

## Known limitations

- Localhost is not authentication. Other local processes running as the same user can connect to the worker port. Use a trusted workstation and avoid multi-user environments.
- Blender itself can execute embedded scripts, add-ons, drivers, and untrusted `.blend` content. Do not open untrusted project files in the worker.
- A local model can still request allowed destructive scene actions such as deletion. Inspect runs, use `.blend` checkpoints, and work on a copy of important files.
- Resource exhaustion from complex geometry or rendering is not fully mitigated in V1. Run Blender with OS-level limits where this matters.
- The OpenAI-compatible endpoints are configured as local by default but may be changed by the operator. Treat endpoint selection as a privacy decision.

## Reporting a vulnerability

Please report potential security issues privately to the repository owner through GitHub's private vulnerability reporting, if enabled. Do not file public issues for vulnerabilities that could permit remote access, arbitrary code execution, or unsafe filesystem writes.
