#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}/..}"

# Prefer repo .env, fall back to /root/.env, allow ENV_FILE override.
DEFAULT_ENV_FILE="${REPO_DIR}/.env"
if [[ ! -f "${DEFAULT_ENV_FILE}" && -f "/root/.env" ]]; then
  DEFAULT_ENV_FILE="/root/.env"
fi
ENV_FILE="${ENV_FILE:-${DEFAULT_ENV_FILE}}"

get_env_var() {
  local key="$1"
  python3 - "${ENV_FILE}" "${key}" <<'PY'
import sys

path, key = sys.argv[1], sys.argv[2]
try:
    lines = open(path, encoding="utf-8").read().splitlines()
except FileNotFoundError:
    sys.exit(0)

for line in lines:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    if k.strip() != key:
        continue
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    print(v)
    sys.exit(0)
PY
}

AZURE_STORAGE_CONNECTION_STRING_ENV=""
if [[ -f "${ENV_FILE}" ]]; then
  AZURE_STORAGE_CONNECTION_STRING_ENV="$(get_env_var AZURE_STORAGE_CONNECTION_STRING)"
fi

# Fall back to current environment if not in .env
AZURE_CONN="${AZURE_STORAGE_CONNECTION_STRING_ENV:-${AZURE_STORAGE_CONNECTION_STRING:-}}"

if [[ -z "${AZURE_CONN}" ]]; then
  echo "ERROR: AZURE_STORAGE_CONNECTION_STRING not found in ${ENV_FILE} or environment."
  echo "Please set it in your .env file or export it before running this script."
  exit 1
fi

cd "${REPO_DIR}"

echo "Installing dvc-azure..."
poetry run pip install dvc-azure

echo "Configuring DVC Azure remote..."
# Remove any broken config and set fresh
poetry run dvc remote modify --local --unset azure-remote connection_string 2>/dev/null || true
poetry run dvc remote modify --local azure-remote connection_string "${AZURE_CONN}"

echo ""
echo "DVC configured successfully!"
echo "You can now run:"
echo "  poetry run dvc push --remote azure-remote"
echo "  poetry run dvc pull --remote azure-remote"
