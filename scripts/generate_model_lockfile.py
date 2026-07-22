#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate artifacts.lock.yaml for hermetic model prefetching.

Discovers all ML models required by NeMo Guardrails and generates a lockfile
with download URLs and checksums for the Konflux generic fetcher.

Usage:
    uv run --with pyyaml --with fastembed scripts/generate_model_lockfile.py
"""

import hashlib
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")

HUGGINGFACE_API = "https://huggingface.co/api/models"
HUGGINGFACE_RESOLVE = "https://huggingface.co"
NLTK_DATA_BASE = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages"

SKIP_FILES = {
    ".gitattributes",
    "README.md",
    "train_script.py",
}

SKIP_EXTENSIONS = {
    ".bin",  # pytorch_model.bin (duplicate of safetensors)
    ".h5",  # tf_model.h5
    ".ot",  # rust_model.ot
}

SKIP_PREFIXES = [
    "onnx/",
    "openvino/",
]


def hf_api_get(url: str) -> Any:
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def download_and_hash(url: str) -> str:
    req = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    sha = hashlib.sha256()
    with urllib.request.urlopen(req) as resp:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def flat_filename(prefix: str, org: str, repo: str, filepath: str) -> str:
    safe_path = filepath.replace("/", "--")
    return f"{prefix}--{org}--{repo}--{safe_path}"


def get_hf_model_commit(model_id: str) -> str:
    info = hf_api_get(f"{HUGGINGFACE_API}/{model_id}")
    return info["sha"]


def list_hf_files(model_id: str, path: str = "") -> list[dict]:
    url = f"{HUGGINGFACE_API}/{model_id}/tree/main"
    if path:
        url += f"/{path}"
    return hf_api_get(url)


def collect_hf_files(
    model_id: str,
    skip_files: set[str] | None = None,
    skip_extensions: set[str] | None = None,
    skip_prefixes: list[str] | None = None,
) -> list[dict]:
    skip_files = skip_files or set()
    skip_extensions = skip_extensions or set()
    skip_prefixes = skip_prefixes or []

    entries = list_hf_files(model_id)
    result = []
    dirs_to_visit = []

    for entry in entries:
        if entry["type"] == "directory":
            dirname = entry["path"]
            if not any(dirname.startswith(p.rstrip("/")) for p in skip_prefixes):
                dirs_to_visit.append(dirname)
            continue
        result.append(entry)

    for d in dirs_to_visit:
        for entry in list_hf_files(model_id, d):
            if entry["type"] == "file":
                result.append(entry)

    filtered = []
    for f in result:
        path = f["path"]
        if path in skip_files:
            continue
        if any(path.endswith(ext) for ext in skip_extensions):
            continue
        if any(path.startswith(p) for p in skip_prefixes):
            continue
        filtered.append(f)

    return filtered


def build_hf_artifacts(
    model_id: str,
    prefix: str,
    skip_files: set[str] | None = None,
    skip_extensions: set[str] | None = None,
    skip_prefixes: list[str] | None = None,
) -> tuple[str, list[dict]]:
    org, repo = model_id.split("/", 1)
    commit = get_hf_model_commit(model_id)
    files = collect_hf_files(model_id, skip_files, skip_extensions, skip_prefixes)

    artifacts = []
    for f in files:
        path = f["path"]
        download_url = f"{HUGGINGFACE_RESOLVE}/{model_id}/resolve/main/{path}"
        filename = flat_filename(prefix, org, repo, path)

        lfs = f.get("lfs")
        if lfs:
            checksum = f"sha256:{lfs['oid']}"
            logging.info(f"  {path} (LFS, checksum from API)")
        else:
            logging.info(f"  {path} (downloading to compute checksum...)")
            sha = download_and_hash(download_url)
            checksum = f"sha256:{sha}"

        artifacts.append(
            {
                "download_url": download_url,
                "filename": filename,
                "checksum": checksum,
            }
        )

    return commit, artifacts


def build_nltk_artifacts(resources: set[str]) -> list[dict]:
    artifacts = []
    for resource in sorted(resources):
        url = f"{NLTK_DATA_BASE}/packages/tokenizers/{resource}.zip"
        logging.info(f"  {resource} (downloading to compute checksum...)")
        sha = download_and_hash(url)
        artifacts.append(
            {
                "download_url": url,
                "filename": f"nltk--{resource}.zip",
                "checksum": f"sha256:{sha}",
            }
        )
    return artifacts


def get_fastembed_hf_sources(model_names: set[str]) -> dict[str, str]:
    try:
        from fastembed import TextEmbedding
    except ImportError:
        logging.warning("fastembed not available — skipping ONNX model discovery")
        return {}

    mapping = {}
    supported = TextEmbedding.list_supported_models()
    for model_name in model_names:
        for m in supported:
            if m["model"] == model_name:
                hf_source = m.get("sources", {}).get("hf")
                if hf_source:
                    mapping[model_name] = hf_source
                    logging.info(f"  FastEmbed maps {model_name} -> {hf_source}")
                break
    return mapping


def discover_models(profile: str = "opensource") -> dict[str, set[str]]:
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from discover_required_models import ModelDiscoverer

    discoverer = ModelDiscoverer(profile)
    return discoverer.discover()


SUPPORTED_PROFILES = {"opensource"}


def main():
    profile = os.environ.get("GUARDRAILS_PROFILE", "opensource")
    if profile not in SUPPORTED_PROFILES:
        logging.error(
            f"Profile '{profile}' is not yet supported for hermetic model prefetch. "
            f"Supported profiles: {', '.join(sorted(SUPPORTED_PROFILES))}"
        )
        sys.exit(1)

    logging.info(f"Generating model lockfile for profile: {profile}")

    models = discover_models(profile)

    all_artifacts = []
    metadata_comments = [f"# profile: {profile}"]

    st_models = models.get("sentence_transformers", set())
    hf_models = models.get("huggingface", set())

    logging.info("\n--- Sentence Transformers models ---")
    for model_id in sorted(st_models):
        if "/" not in model_id:
            continue
        logging.info(f"Processing {model_id}:")
        commit, artifacts = build_hf_artifacts(
            model_id,
            prefix="hf",
            skip_files=SKIP_FILES,
            skip_extensions=SKIP_EXTENSIONS,
            skip_prefixes=SKIP_PREFIXES,
        )
        all_artifacts.extend(artifacts)
        metadata_comments.append(f"# {model_id} commit: {commit}")

    logging.info("\n--- FastEmbed ONNX models ---")
    fastembed_mapping = get_fastembed_hf_sources(st_models)
    for original_name, onnx_repo in sorted(fastembed_mapping.items()):
        logging.info(f"Processing {onnx_repo} (ONNX for {original_name}):")
        commit, artifacts = build_hf_artifacts(
            onnx_repo,
            prefix="hf",
            skip_files={".gitattributes", "README.md"},
        )
        all_artifacts.extend(artifacts)
        metadata_comments.append(f"# {onnx_repo} commit: {commit}")

    hf_only = hf_models - st_models
    if hf_only:
        logging.info("\n--- HuggingFace-only models ---")
        for model_id in sorted(hf_only):
            if "/" not in model_id:
                logging.info(f"  Skipping {model_id} (no org/repo format)")
                continue
            logging.info(f"Processing {model_id}:")
            commit, artifacts = build_hf_artifacts(
                model_id,
                prefix="hf",
                skip_files=SKIP_FILES,
            )
            all_artifacts.extend(artifacts)
            metadata_comments.append(f"# {model_id} commit: {commit}")

    nltk_resources = models.get("nltk", set())
    if nltk_resources:
        logging.info("\n--- NLTK data ---")
        nltk_artifacts = build_nltk_artifacts(nltk_resources)
        all_artifacts.extend(nltk_artifacts)

    lockfile = {
        "metadata": {"version": "1.0"},
        "artifacts": all_artifacts,
    }

    output_path = Path("artifacts.lock.yaml")
    header_lines = [
        "# Auto-generated by scripts/generate_model_lockfile.py",
        "# Re-run to update: uv run --with pyyaml --with fastembed scripts/generate_model_lockfile.py",
        "",
    ]
    header_lines.extend(metadata_comments)
    if metadata_comments:
        header_lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        yaml.dump(lockfile, f, default_flow_style=False, sort_keys=False)

    logging.info(f"\nWrote {len(all_artifacts)} artifacts to {output_path}")


if __name__ == "__main__":
    main()
