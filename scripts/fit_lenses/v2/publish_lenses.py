#!/usr/bin/env python3
"""Validate, stage, clean-cache test, and optionally publish v2 Lens artifacts.

Artifact files are uploaded first. The Hub index is deliberately updated last so
readers can never discover a partially uploaded release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HUB_REPO = "sida/llmdbg"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--stage", type=Path, default=PROJECT_ROOT / "build/hf-upload")
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_artifact(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = {
        "format_version", "artifact", "sha256", "model_id", "model_revision",
        "tokenizer_revision", "method", "sites", "hidden_size", "dtype",
        "provenance",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"{manifest_path} is missing {', '.join(missing)}")
    if manifest["format_version"] != 2:
        raise ValueError(f"{manifest_path} is not format version 2")
    if manifest["method"] not in {"j", "r", "jpp", "s"}:
        raise ValueError(f"invalid method {manifest['method']!r}")
    tensor_path = directory / manifest["artifact"]
    if digest(tensor_path) != manifest["sha256"]:
        raise ValueError(f"checksum mismatch for {tensor_path}")
    expected = set()
    if manifest["sites"].get("embed"):
        expected.add("embed")
    expected.update(f"mid.{layer}" for layer in manifest["sites"].get("mid", []))
    expected.update(f"post.{layer}" for layer in manifest["sites"].get("post", []))
    hidden = int(manifest["hidden_size"])
    with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        if keys != expected:
            raise ValueError(f"site keys differ: manifest={sorted(expected)}, file={sorted(keys)}")
        for key in keys:
            if tuple(tensors.get_slice(key).get_shape()) != (hidden, hidden):
                raise ValueError(f"{key} is not a {hidden}x{hidden} map")
    return manifest


def artifact_path(manifest: dict[str, Any]) -> str:
    model = str(manifest["model_id"]).replace("/", "--")
    return f"lenses/v2/{model}/{manifest['method']}"


def stage_artifacts(artifacts: list[Path], stage: Path) -> list[dict[str, Any]]:
    entries = []
    for source in artifacts:
        manifest = validate_artifact(source)
        relative = artifact_path(manifest)
        destination = stage / relative
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", manifest["artifact"]):
            shutil.copy2(source / name, destination / name)
        validate_artifact(destination)
        entries.append({
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "method": manifest["method"],
            "format_version": 2,
            "path": relative,
        })
    return entries


def clean_cache_test(stage: Path, entries: list[dict[str, Any]]) -> None:
    from llmdbg import workspace_lens

    with tempfile.TemporaryDirectory(prefix="llmdbg-workspace-lens-cache-"):
        old = workspace_lens.os.environ.get("LLMDBG_WORKSPACE_LENS_PATH")
        try:
            workspace_lens.os.environ["LLMDBG_WORKSPACE_LENS_PATH"] = str(
                stage / "lenses" / "v2"
            )
            for entry in entries:
                workspace_lens.clear_cache()
                loaded = workspace_lens.load_v2(entry["model_id"], entry["method"])
                if loaded is None:
                    raise RuntimeError(f"clean-cache load failed for {entry}")
        finally:
            workspace_lens.clear_cache()
            if old is None:
                workspace_lens.os.environ.pop("LLMDBG_WORKSPACE_LENS_PATH", None)
            else:
                workspace_lens.os.environ["LLMDBG_WORKSPACE_LENS_PATH"] = old


def update_index(stage: Path, additions: list[dict[str, Any]]) -> Path:
    index_path = stage / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    current = list(index.get("workspace_lenses", []))
    keys = {(entry["model_id"].lower(), entry["method"]) for entry in additions}
    current = [
        entry for entry in current
        if (str(entry.get("model_id", "")).lower(), entry.get("method")) not in keys
    ]
    index["workspace_lenses"] = sorted(
        current + additions, key=lambda entry: (entry["model_id"].lower(), entry["method"])
    )
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index_path


def seed_index_from_hub(stage: Path) -> Path:
    """Start an upload from the live index so unrelated entries survive."""
    from huggingface_hub import hf_hub_download

    remote = Path(hf_hub_download(
        repo_id=HUB_REPO,
        repo_type="dataset",
        filename="index.json",
        force_download=True,
    ))
    index = json.loads(remote.read_text())
    index_path = stage / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index_path


def upload(stage: Path, entries: list[dict[str, Any]], index_path: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for entry in entries:
        api.upload_folder(
            repo_id=HUB_REPO, repo_type="dataset",
            folder_path=stage / entry["path"], path_in_repo=entry["path"],
            commit_message=f"Add {entry['model_id']} {entry['method']} Lens v2",
        )
    api.upload_file(
        repo_id=HUB_REPO, repo_type="dataset", path_or_fileobj=index_path,
        path_in_repo="index.json", commit_message="Update Lens v2 index",
    )


def main() -> int:
    args = arguments()
    entries = stage_artifacts(args.artifacts, args.stage)
    clean_cache_test(args.stage, entries)
    if args.upload:
        seed_index_from_hub(args.stage)
    index_path = update_index(args.stage, entries)
    if args.upload:
        upload(args.stage, entries, index_path)
    print(json.dumps({"stage": str(args.stage), "entries": entries, "uploaded": args.upload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
