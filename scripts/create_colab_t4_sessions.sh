#!/usr/bin/env bash

set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  create_colab_t4_sessions.sh [COUNT] [NAME_PREFIX]

Examples:
  ./scripts/create_colab_t4_sessions.sh 3
  ./scripts/create_colab_t4_sessions.sh 2 knee-kl

If COUNT is omitted, the script asks for it interactively. Sessions are created
sequentially with a T4 GPU. Creation stops after the first failed request because
that normally indicates unavailable capacity, authentication trouble, or quota.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v colab >/dev/null 2>&1; then
    echo "Error: the 'colab' CLI is not installed or is not on PATH." >&2
    exit 1
fi

count="${1:-}"
if [[ -z "$count" ]]; then
    read -r -p "Number of T4 Colab sessions to create: " count
fi

if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: COUNT must be a positive integer; received '$count'." >&2
    usage >&2
    exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
prefix="${2:-knee-kl-t4-$timestamp}"
prefix="${prefix//[^[:alnum:]_-]/-}"
if [[ -z "$prefix" ]]; then
    echo "Error: NAME_PREFIX contains no usable characters." >&2
    exit 2
fi

state_root="${COLAB_SESSION_BATCH_DIR:-$HOME/.config/colab-cli/session-batches}"
mkdir -p "$state_root"
manifest="$state_root/${prefix}.txt"
: > "$manifest"

created_names=()
failed=0

echo "Requesting $count Colab session(s) with T4 GPUs..."
echo "Session manifest: $manifest"

for ((index = 1; index <= count; index++)); do
    session_name="${prefix}-$(printf '%02d' "$index")"
    echo
    echo "[$index/$count] Creating '$session_name'..."

    if creation_output="$(colab new --session "$session_name" --gpu T4 2>&1)"; then
        printf '%s\n' "$creation_output"
        created_names+=("$session_name")

        session_url=""
        for attempt in 1 2 3; do
            if session_url="$(colab url --session "$session_name" 2>/dev/null)"; then
                break
            fi
            sleep 2
        done

        {
            printf 'session=%s\n' "$session_name"
            printf 'gpu=T4\n'
            printf 'url=%s\n' "${session_url:-unavailable}"
            printf 'stop=colab stop --session %q\n\n' "$session_name"
        } >> "$manifest"

        if [[ -n "$session_url" ]]; then
            echo "URL: $session_url"
        else
            echo "Created, but its browser URL is not available yet."
        fi
    else
        failed=1
        echo "Failed to create '$session_name':" >&2
        printf '%s\n' "$creation_output" >&2
        echo "Stopping further creation attempts." >&2
        break
    fi
done

echo
echo "Created ${#created_names[@]} of $count requested T4 session(s)."
if ((${#created_names[@]} > 0)); then
    echo "Session names:"
    printf '  %s\n' "${created_names[@]}"
    echo "Saved details and stop commands to: $manifest"
fi

echo
echo "Current Colab sessions:"
colab sessions || true

if ((failed)); then
    exit 1
fi
