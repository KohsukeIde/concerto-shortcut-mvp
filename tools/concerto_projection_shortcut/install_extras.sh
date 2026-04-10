#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
REPO_ROOT="$(pwd -P)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/concerto_projection_shortcut/device_defaults.sh"

if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
torch is not available in this interpreter yet.
Install the base Pointcept environment first, then rerun this helper.
Use tools/concerto_projection_shortcut/create_env.sh for this device.
EOF
fi

"${PYTHON_BIN}" -m pip install transformers==4.50.3 peft
