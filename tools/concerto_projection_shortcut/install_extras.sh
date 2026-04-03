#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
torch is not available in this interpreter yet.
Install the base Pointcept environment first, then rerun this helper.
The repo ships /home/cvrt/Desktop/Pointcept/environment.yml for the full setup.
EOF
fi

"${PYTHON_BIN}" -m pip install transformers==4.50.3 peft
