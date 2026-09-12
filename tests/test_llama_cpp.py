from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aculptoi.cli.app import app
from aculptoi.runtime import LlamaServeConfig


def test_llama_serve_config_builds_the_documented_local_command() -> None:
    config = LlamaServeConfig(
        model_path=Path("/models/Qwen3.8-27B-Q4_K_M.gguf"),
        mmproj_path=Path("/models/mmproj-F16.gguf"),
    )

    assert config.command() == [
        "llama",
        "serve",
        "-m",
        "/models/Qwen3.8-27B-Q4_K_M.gguf",
        "--mmproj",
        "/models/mmproj-F16.gguf",
        "-c",
        "32768",
        "-np",
        "1",
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "-a",
        "aculptoi",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]


def test_model_serve_dry_run_requires_hf_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)

    result = CliRunner().invoke(
        app,
        [
            "model",
            "serve",
            "--model",
            "/does/not/matter.gguf",
            "--mmproj",
            "/does/not/matter-mmproj.gguf",
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert "HF_HOME is not set" in result.stdout


def test_model_serve_dry_run_reports_the_validated_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj-F16.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"projection")
    arguments = [
        "serve",
        "--model",
        str(model),
        "--mmproj",
        str(mmproj),
        "--dry-run",
        "--json",
    ]

    result = CliRunner().invoke(app, ["model", *arguments])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["command"] == [
        "llama",
        "serve",
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "-c",
        "32768",
        "-np",
        "1",
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "-a",
        "aculptoi",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]
