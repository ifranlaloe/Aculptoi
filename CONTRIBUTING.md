# Contributing to Aculptoi

Thank you for helping make local 3D agents easier to inspect and safer to run.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
mypy src
```

## Guidelines

- Keep modules focused: model providers do not know Blender internals, and vision does not execute actions.
- Do not add arbitrary shell or `bpy` execution to model-produced action payloads.
- Add schema and validation tests with every new action.
- Keep local transports loopback-only unless an explicit security design is approved.
- Update README status and security documentation when behavior changes.
- Prefer deterministic defaults and persist enough run metadata to reproduce a decision.

For a substantial feature, open an issue first with the proposed action schema, worker behavior, safety implications, and test plan.
