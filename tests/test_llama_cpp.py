from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aculptoi.cli.app import app
from aculptoi.runtime import (
    DAVIDAU_MMPROJ_FILENAME,
    DAVIDAU_MODEL_FILENAME,
    DAVIDAU_REPOSITORY_DIRECTORY,
    LlamaServeConfig,
)


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
        "--reasoning",
        "on",
        "--reasoning-budget",
        "512",
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


def test_model_serve_uses_the_davidau_default_under_hf_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "hub" / DAVIDAU_REPOSITORY_DIRECTORY / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    model = snapshot / DAVIDAU_MODEL_FILENAME
    mmproj = snapshot / DAVIDAU_MMPROJ_FILENAME
    model.write_bytes(b"model")
    mmproj.write_bytes(b"projection")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    result = CliRunner().invoke(app, ["model", "serve", "--dry-run", "--json"])

    body = json.loads(result.stdout)
    assert result.exit_code == 0
    assert body["artifact_source"] == "default-davidau"
    assert body["reasoning_budget"] == 512
    assert body["command"][3] == str(model)
    assert body["command"][5] == str(mmproj)


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
        "--reasoning",
        "on",
        "--reasoning-budget",
        "512",
        "-a",
        "aculptoi",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]
