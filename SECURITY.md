# Security policy and threat model

## Security posture

Aculptoi controls Blender, so it is designed to make the model-to-scene boundary narrow and inspectable. It is experimental software and is not a sandbox for malicious files, models, or Blender extensions.

The persistent worker binds only to `127.0.0.1` by default. Configuration rejects non-loopback worker hosts. Do not expose its port with a proxy, tunnel, container mapping, or firewall rule.

## Trust boundaries

1. **Actor model → harness.** The actor returns JSON only. Pydantic validates a discriminated, allowlisted action union. Arbitrary shell commands, Python, `bpy` expressions, filesystem paths, and unknown action names are not part of the schema.
2. **Harness → Blender worker.** The worker validates the allowlist again. It does not trust an HTTP client merely because it is local.
3. **Vision model → harness.** The vision critic has a read-only schema (`score`, summary, issues). It cannot invoke Blender operations.
4. **Worker → filesystem.** Render outputs and checkpoints must resolve under the current project's `.aculptoi/` directory. Model actions do not carry paths.

## Known limitations

- Localhost is not authentication. Other local processes running as the same user can connect to the worker port. Use a trusted workstation and avoid multi-user environments.
- Blender itself can execute embedded scripts, add-ons, drivers, and untrusted `.blend` content. Do not open untrusted project files in the worker.
- A local model can still request allowed destructive scene actions such as deletion. Inspect runs, use `.blend` checkpoints, and work on a copy of important files.
- Resource exhaustion from complex geometry or rendering is not fully mitigated in V1. Run Blender with OS-level limits where this matters.
- The OpenAI-compatible endpoints are configured as local by default but may be changed by the operator. Treat endpoint selection as a privacy decision.

## Reporting a vulnerability

Please report potential security issues privately to the repository owner through GitHub's private vulnerability reporting, if enabled. Do not file public issues for vulnerabilities that could permit remote access, arbitrary code execution, or unsafe filesystem writes.
