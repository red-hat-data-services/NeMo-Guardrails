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

# RedHatAI HuggingFace mirrors for supply chain security.
# Keys are the cache model IDs that libraries expect at runtime;
# values are the RedHatAI repos to download from instead.
HF_MODEL_MIRRORS = {
    "sentence-transformers/all-MiniLM-L6-v2": "RedHatAI/all-MiniLM-L6-v2",
}

# ONNX model mirrors.  fastembed expects specific cache model IDs (e.g.
# qdrant/all-MiniLM-L6-v2-onnx) so we download from RedHatAI but keep
# the original model ID in filenames for cache compatibility.
ONNX_MIRRORS: dict[str, dict] = {
    "qdrant/all-MiniLM-L6-v2-onnx": {
        "source": "RedHatAI/all-MiniLM-L6-v2",
        "file_mapping": {
            "config.json": "config.json",
            "onnx/model.onnx": "model.onnx",
            "special_tokens_map.json": "special_tokens_map.json",
            "tokenizer.json": "tokenizer.json",
            "tokenizer_config.json": "tokenizer_config.json",
            "vocab.txt": "vocab.txt",
        },
    },
}

# NLTK punkt_tab data from RedHatAI (replaces legacy punkt.zip from GitHub)
NLTK_PUNKT_TAB_REPO = "RedHatAI/nltk-punkt-tab"
NLTK_PUNKT_TAB_LANGS = ["english"]
NLTK_PUNKT_TAB_FILES = [
    "abbrev_types.txt",
    "collocations.tab",
    "ortho_context.tab",
    "sent_starters.txt",
]

SKIP_FILES = {
    ".gitattributes",
    "README.md",
    "train_script.py",
    "inference.py",
    "requirements.txt",
    "config.yaml",
    "burn_scars_config.yaml",
}

SKIP_EXTENSIONS = {
    ".bin",  # pytorch_model.bin (duplicate of safetensors)
    ".h5",  # tf_model.h5
    ".ot",  # rust_model.ot
    ".pt",  # checkpoint files (not needed for inference)
    ".tif",  # geospatial examples
}

SKIP_PREFIXES = [
    "onnx/",
    "openvino/",
    "examples/",
    "splits/",
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
    download_model_id: str | None = None,
    skip_files: set[str] | None = None,
    skip_extensions: set[str] | None = None,
    skip_prefixes: list[str] | None = None,
) -> tuple[str, list[dict]]:
    source_id = download_model_id or model_id
    org, repo = model_id.split("/", 1)
    commit = get_hf_model_commit(source_id)
    files = collect_hf_files(source_id, skip_files, skip_extensions, skip_prefixes)

    artifacts = []
    for f in files:
        path = f["path"]
        download_url = f"{HUGGINGFACE_RESOLVE}/{source_id}/resolve/main/{path}"
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


def build_nltk_punkt_tab_artifacts(
    repo: str,
    langs: list[str],
    files: list[str],
) -> tuple[str, list[dict]]:
    commit = get_hf_model_commit(repo)
    artifacts = []
    for lang in langs:
        for fname in files:
            path = f"{lang}/{fname}"
            download_url = f"{HUGGINGFACE_RESOLVE}/{repo}/resolve/main/{path}"
            logging.info(f"  {path} (downloading to compute checksum...)")
            sha = download_and_hash(download_url)
            artifacts.append(
                {
                    "download_url": download_url,
                    "filename": f"punkt_tab--{lang}--{fname}",
                    "checksum": f"sha256:{sha}",
                }
            )
    return commit, artifacts


def build_mirrored_onnx_artifacts(
    cache_model_id: str,
    mirror_config: dict,
    prefix: str = "hf",
) -> tuple[str, list[dict]]:
    source_id = mirror_config["source"]
    file_mapping = mirror_config["file_mapping"]
    cache_org, cache_repo = cache_model_id.split("/", 1)
    commit = get_hf_model_commit(source_id)

    all_files = {}
    for entry in list_hf_files(source_id):
        all_files[entry["path"]] = entry
        if entry["type"] == "directory":
            for subentry in list_hf_files(source_id, entry["path"]):
                if subentry["type"] == "file":
                    all_files[subentry["path"]] = subentry

    artifacts = []
    for source_path, cache_name in file_mapping.items():
        download_url = f"{HUGGINGFACE_RESOLVE}/{source_id}/resolve/main/{source_path}"
        filename = flat_filename(prefix, cache_org, cache_repo, cache_name)

        file_info = all_files.get(source_path)
        if file_info and file_info.get("lfs"):
            checksum = f"sha256:{file_info['lfs']['oid']}"
            logging.info(f"  {source_path} -> {cache_name} (LFS, checksum from API)")
        else:
            logging.info(f"  {source_path} -> {cache_name} (downloading to compute checksum...)")
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
        mirror_id = HF_MODEL_MIRRORS.get(model_id)
        if mirror_id:
            logging.info(f"Processing {model_id} (from {mirror_id}):")
        else:
            logging.info(f"Processing {model_id}:")
        commit, artifacts = build_hf_artifacts(
            model_id,
            prefix="hf",
            download_model_id=mirror_id,
            skip_files=SKIP_FILES,
            skip_extensions=SKIP_EXTENSIONS,
            skip_prefixes=SKIP_PREFIXES,
        )
        all_artifacts.extend(artifacts)
        metadata_comments.append(f"# {model_id} commit: {commit}")

    logging.info("\n--- FastEmbed ONNX models ---")
    fastembed_mapping = get_fastembed_hf_sources(st_models)
    for original_name, onnx_repo in sorted(fastembed_mapping.items()):
        if onnx_repo in ONNX_MIRRORS:
            mirror = ONNX_MIRRORS[onnx_repo]
            logging.info(
                f"Processing {onnx_repo} (ONNX for {original_name}, from {mirror['source']}):"
            )
            commit, artifacts = build_mirrored_onnx_artifacts(
                cache_model_id=onnx_repo,
                mirror_config=mirror,
            )
        else:
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
        logging.info(f"\n--- NLTK data (punkt_tab from {NLTK_PUNKT_TAB_REPO}) ---")
        commit, nltk_artifacts = build_nltk_punkt_tab_artifacts(
            repo=NLTK_PUNKT_TAB_REPO,
            langs=NLTK_PUNKT_TAB_LANGS,
            files=NLTK_PUNKT_TAB_FILES,
        )
        all_artifacts.extend(nltk_artifacts)
        metadata_comments.append(f"# {NLTK_PUNKT_TAB_REPO} commit: {commit}")

    seen_urls: dict[str, str] = {}
    deduped_artifacts = []
    for art in all_artifacts:
        url = art["download_url"]
        if url in seen_urls:
            logging.info(
                f"  Dedup: {art['filename']} shares URL with {seen_urls[url]}, skipping"
            )
            continue
        seen_urls[url] = art["filename"]
        deduped_artifacts.append(art)

    lockfile = {
        "metadata": {"version": "1.0"},
        "artifacts": deduped_artifacts,
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

    logging.info(f"\nWrote {len(deduped_artifacts)} artifacts to {output_path}")


if __name__ == "__main__":
    main()
