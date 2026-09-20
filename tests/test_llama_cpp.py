from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from aculptoi.cli.app import app
from aculptoi.runtime import (
    DAVIDAU_MMPROJ_FILENAME,
    DAVIDAU_MODEL_FILENAME,
    DAVIDAU_REPOSITORY_DIRECTORY,
    LlamaServeConfig,
    default_davidau_artifacts,
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
        "65536",
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


def test_llama_serve_config_adds_bounded_mtp_speculation_only_when_enabled() -> None:
    config = LlamaServeConfig(
        model_path=Path("/models/Qwen3.8-27B-TWIN-TURBO-NEO-MTP-Q4_K_M.gguf"),
        mmproj_path=Path("/models/mmproj-BF16.gguf"),
        mtp_enabled=True,
        mtp_draft_tokens=2,
    )

    assert config.command() == [
        "llama",
        "serve",
        "-m",
        "/models/Qwen3.8-27B-TWIN-TURBO-NEO-MTP-Q4_K_M.gguf",
        "--mmproj",
        "/models/mmproj-BF16.gguf",
        "-c",
        "65536",
        "-np",
        "1",
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "2",
        "-a",
        "aculptoi",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]

    with pytest.raises(ValidationError):
        LlamaServeConfig(
            model_path=Path("/models/model.gguf"),
            mmproj_path=Path("/models/mmproj.gguf"),
            mtp_draft_tokens=0,
        )


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
    assert body["context_size"] == 65_536
    assert body["mtp_enabled"] is True
    assert body["mtp_draft_tokens"] == 2
    assert "reasoning_budget" not in body
    assert "--reasoning-budget" not in body["command"]
    assert "--spec-type" in body["command"]
    assert "draft-mtp" in body["command"]
    assert "--spec-draft-n-max" in body["command"]
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
        "65536",
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


def test_default_model_serve_can_disable_mtp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "hub" / DAVIDAU_REPOSITORY_DIRECTORY / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / DAVIDAU_MODEL_FILENAME).write_bytes(b"model")
    (snapshot / DAVIDAU_MMPROJ_FILENAME).write_bytes(b"projection")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    result = CliRunner().invoke(app, ["model", "serve", "--no-mtp", "--dry-run", "--json"])

    body = json.loads(result.stdout)
    assert result.exit_code == 0
    assert body["mtp_enabled"] is False
    assert "--spec-type" not in body["command"]
    assert "--spec-draft-n-max" not in body["command"]


def test_explicit_model_requires_explicit_mtp_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    model = tmp_path / "custom-mtp.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"model")
    mmproj.write_bytes(b"projection")
    base = ["model", "serve", "-m", str(model), "--mmproj", str(mmproj), "--dry-run", "--json"]

    without_mtp = CliRunner().invoke(app, base)
    with_mtp = CliRunner().invoke(app, [*base[:-2], "--mtp", *base[-2:]])

    without_mtp_body = json.loads(without_mtp.stdout)
    with_mtp_body = json.loads(with_mtp.stdout)
    assert without_mtp.exit_code == with_mtp.exit_code == 0
    assert without_mtp_body["mtp_enabled"] is False
    assert "--spec-type" not in without_mtp_body["command"]
    assert with_mtp_body["mtp_enabled"] is True
    assert with_mtp_body["command"].count("--spec-type") == 1


def test_model_serve_rejects_contradictory_mtp_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    result = CliRunner().invoke(app, ["model", "serve", "--mtp", "--no-mtp", "--dry-run"])

    assert result.exit_code == 1
    assert "cannot be used together" in result.output


def test_default_artifact_discovery_requires_a_complete_new_model_pair(tmp_path: Path) -> None:
    snapshots = tmp_path / "hub" / DAVIDAU_REPOSITORY_DIRECTORY / "snapshots"
    missing_model = snapshots / "a-missing-model"
    missing_model.mkdir(parents=True)
    (missing_model / DAVIDAU_MMPROJ_FILENAME).write_bytes(b"projection")
    missing_projection = snapshots / "b-missing-projection"
    missing_projection.mkdir()
    (missing_projection / DAVIDAU_MODEL_FILENAME).write_bytes(b"model")

    with pytest.raises(FileNotFoundError, match="TWIN-TURBO"):
        default_davidau_artifacts(tmp_path)

    complete = snapshots / "z-complete"
    complete.mkdir()
    model = complete / DAVIDAU_MODEL_FILENAME
    mmproj = complete / DAVIDAU_MMPROJ_FILENAME
    model.write_bytes(b"model")
    mmproj.write_bytes(b"projection")

    assert default_davidau_artifacts(tmp_path) == (model, mmproj)
