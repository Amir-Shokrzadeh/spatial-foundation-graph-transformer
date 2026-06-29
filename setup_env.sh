#!/usr/bin/env bash
# =============================================================
# setup_env.sh
# Run once from the project root to create the conda environment
# and install the project in editable mode.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
# =============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="sfgt"

echo "============================================================"
echo "  Spatial Foundation Graph Transformer — Environment Setup"
echo "============================================================"
echo ""
echo "Project root : $PROJECT_ROOT"
echo "Conda env    : $ENV_NAME"
echo ""

# ── 1. Check conda is available ──────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda or Anaconda first."
    echo "       https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ── 2. Create (or update) the environment ────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists — updating..."
    conda env update -f "$PROJECT_ROOT/environment.yml" --prune
else
    echo "Creating environment '${ENV_NAME}'..."
    conda env create -f "$PROJECT_ROOT/environment.yml"
fi

# ── 3. Activate and install project in editable mode ─────────
echo ""
echo "Installing project in editable mode (pip install -e .) ..."
# Use 'conda run' so we don't need to source conda.sh
conda run -n "$ENV_NAME" pip install -e "$PROJECT_ROOT" --quiet

# ── 4. Verify key imports ─────────────────────────────────────
echo ""
echo "Verifying core imports..."
conda run -n "$ENV_NAME" python - <<'EOF'
import sys
packages = [
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("torch",        "torch"),
    ("torch_geometric", "torch_geometric"),
    ("scanpy",       "scanpy"),
    ("squidpy",      "squidpy"),
    ("anndata",      "anndata"),
    ("sklearn",      "scikit-learn"),
    ("umap",         "umap-learn"),
    ("networkx",     "networkx"),
    ("yaml",         "pyyaml"),
    ("rich",         "rich"),
]
ok, failed = [], []
for mod, label in packages:
    try:
        __import__(mod)
        ok.append(label)
    except ImportError:
        failed.append(label)

print(f"  ✓  Imported: {', '.join(ok)}")
if failed:
    print(f"  ✗  Missing : {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n  All core packages verified successfully.")
EOF

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Activate with:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  Then launch Jupyter:"
echo "    jupyter lab"
echo "============================================================"
