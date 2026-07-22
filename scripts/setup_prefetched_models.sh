#!/bin/bash
set -euo pipefail

# Reconstruct ML model cache directories from prefetched generic artifacts.
# Called during the Docker build in place of pre_download_required_models.py.
#
# Usage: GUARDRAILS_PROFILE=opensource ./scripts/setup_prefetched_models.sh
#
# Expects:
#   - GUARDRAILS_PROFILE env var (default: opensource)
#   - Prefetched files at /cachi2/output/deps/generic/hf--* and nltk--*
#   - Environment variables: HF_HOME, SENTENCE_TRANSFORMERS_HOME, FASTEMBED_CACHE_PATH, NLTK_DATA

PROFILE="${GUARDRAILS_PROFILE:-opensource}"
GENERIC_DIR="${CACHI2_GENERIC_DIR:-/cachi2/output/deps/generic}"

setup_hf_model() {
    local model_id="$1"    # e.g. sentence-transformers/all-MiniLM-L6-v2
    local cache_base="$2"  # e.g. /app/.cache/huggingface
    local commit="$3"      # git commit hash
    local use_hub="${4:-false}"  # true for HF_HOME (uses hub/ prefix), false for library caches

    local org="${model_id%%/*}"
    local repo="${model_id#*/}"
    local cache_name="models--${org}--${repo}"
    local cache_root="${cache_base}"
    if [ "${use_hub}" = "true" ]; then
        cache_root="${cache_base}/hub"
    fi
    local snapshot_dir="${cache_root}/${cache_name}/snapshots/${commit}"
    local refs_dir="${cache_root}/${cache_name}/refs"

    mkdir -p "${snapshot_dir}" "${refs_dir}"
    echo -n "${commit}" > "${refs_dir}/main"

    local prefix="hf--${org}--${repo}--"
    for src in "${GENERIC_DIR}/${prefix}"*; do
        [ -f "$src" ] || continue
        local basename="${src##*/}"
        # Strip the prefix to get the original filename, converting -- back to /
        local relpath="${basename#"${prefix}"}"
        relpath="${relpath//--//}"

        local dest_dir="${snapshot_dir}/$(dirname "${relpath}")"
        mkdir -p "${dest_dir}"
        cp "$src" "${dest_dir}/$(basename "${relpath}")"
    done

    echo "Installed HF model ${model_id} into ${cache_base}"
}

setup_nltk_data() {
    local resource="$1"  # e.g. punkt
    local src="${GENERIC_DIR}/nltk--${resource}.zip"

    if [ ! -f "$src" ]; then
        echo "Warning: NLTK resource ${resource} not found at ${src}"
        return
    fi

    local dest_dir="${NLTK_DATA}/tokenizers"
    mkdir -p "${dest_dir}"
    python3 -c "import zipfile; zipfile.ZipFile('${src}').extractall('${dest_dir}/')"
    echo "Installed NLTK resource ${resource}"
}

# Read commit hashes from artifacts.lock.yaml header comments.
# Format: "# org/repo commit: <hash>"
get_commit_from_lockfile() {
    local model_id="$1"
    local lockfile="${2:-/app/artifacts.lock.yaml}"
    grep "^# ${model_id} commit:" "${lockfile}" 2>/dev/null | sed 's/.*commit: //'
}

setup_opensource() {
    local lockfile="/app/artifacts.lock.yaml"

    # Sentence Transformers model -> HF_HOME and SENTENCE_TRANSFORMERS_HOME
    local st_commit
    st_commit=$(get_commit_from_lockfile "sentence-transformers/all-MiniLM-L6-v2" "${lockfile}")
    if [ -n "${st_commit}" ]; then
        setup_hf_model "sentence-transformers/all-MiniLM-L6-v2" "${HF_HOME}" "${st_commit}" true
        # sentence_transformers passes cache_dir directly (no hub/ prefix)
        setup_hf_model "sentence-transformers/all-MiniLM-L6-v2" "${SENTENCE_TRANSFORMERS_HOME}" "${st_commit}" false
    fi

    # FastEmbed ONNX model -> FASTEMBED_CACHE_PATH
    local fe_commit
    fe_commit=$(get_commit_from_lockfile "qdrant/all-MiniLM-L6-v2-onnx" "${lockfile}")
    if [ -n "${fe_commit}" ]; then
        setup_hf_model "qdrant/all-MiniLM-L6-v2-onnx" "${FASTEMBED_CACHE_PATH}" "${fe_commit}" false
    fi

    # NLTK data
    setup_nltk_data "punkt"
}

main() {
    echo "Setting up prefetched models for profile: ${PROFILE}"

    case "${PROFILE}" in
        opensource)
            setup_opensource
            ;;
        *)
            echo "ERROR: Profile '${PROFILE}' is not yet supported for hermetic model setup."
            echo "Supported profiles: opensource"
            exit 1
            ;;
    esac

    echo "Model setup complete"
}

main "$@"
