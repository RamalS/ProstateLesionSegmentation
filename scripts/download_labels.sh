#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_labels.sh  —  Download PI-CAI human expert lesion annotations
#
# Usage:
#   bash scripts/download_labels.sh [OPTIONS]
#
# What this downloads:
#   From https://github.com/DIAGNijmegen/picai_labels (sparse checkout):
#     • csPCa_lesion_delineations/human_expert/resampled/   (1295 cases)
#     • csPCa_lesion_delineations/human_expert/Pooch25/     (205 cases)
#   Together these cover all ~1500 labelled cases in the PI-CAI dataset.
#   All .nii.gz files are placed flat into <out-dir>/labels/.
#
# Options:
#   --out-dir DIR  Root data directory (default: ./data)
#   --dry-run      Print what would happen without doing anything
#   -h, --help     Show this help message
#
# Requirements:
#   git >= 2.25 (sparse-checkout support)
#   python3      (used for the flatten step)
#
# Examples:
#   bash scripts/download_labels.sh
#   bash scripts/download_labels.sh --out-dir /data
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="$REPO_ROOT/data"
DRY_RUN=false

LABELS_REPO="https://github.com/DIAGNijmegen/picai_labels"
SPARSE_PATHS=(
    "csPCa_lesion_delineations/human_expert/resampled"
    "csPCa_lesion_delineations/human_expert/Pooch25"
)

# ---------------------------------------------------------------------------
# Colours (only when stdout is a terminal)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; RESET=''
fi

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
print_help() {
    sed -n '/^# Usage/,/^# ---/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)   print_help ;;
        --out-dir)   OUT_DIR="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        *)
            error "Unknown argument: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if ! command -v git &>/dev/null; then
    error "git is required but not installed."
    exit 1
fi

GIT_VERSION="$(git --version | awk '{print $3}')"
GIT_MAJOR="$(echo "$GIT_VERSION" | cut -d. -f1)"
GIT_MINOR="$(echo "$GIT_VERSION" | cut -d. -f2)"
if (( GIT_MAJOR < 2 || ( GIT_MAJOR == 2 && GIT_MINOR < 25 ) )); then
    error "git >= 2.25 required for sparse-checkout (found $GIT_VERSION)."
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    error "python3 is required but not installed."
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABELS_DIR="$OUT_DIR/labels"
CLONE_DIR="$OUT_DIR/.picai_labels_tmp"
DONE_MARKER="$OUT_DIR/.labels_done"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}PI-CAI Labels Download${RESET}"
echo "  Source      : $LABELS_REPO"
echo "  Subdirs     : ${SPARSE_PATHS[*]}"
echo "  Output dir  : $LABELS_DIR"
echo "  Dry run     : $DRY_RUN"
echo ""

# ---------------------------------------------------------------------------
# Dry-run early exit
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" == true ]]; then
    info "[dry-run] Would sparse-clone $LABELS_REPO into $CLONE_DIR"
    for sp in "${SPARSE_PATHS[@]}"; do
        info "[dry-run] Would fetch path: $sp"
    done
    info "[dry-run] Would copy all .nii.gz files → $LABELS_DIR"
    info "[dry-run] Would remove $CLONE_DIR"
    success "Dry run complete. No files were downloaded."
    exit 0
fi

# ---------------------------------------------------------------------------
# Skip if already done
# ---------------------------------------------------------------------------
if [[ -f "$DONE_MARKER" ]]; then
    N_FILES="$(find "$LABELS_DIR" -name '*.nii.gz' 2>/dev/null | wc -l)"
    success "Labels already downloaded ($N_FILES .nii.gz files in $LABELS_DIR) — skipping."
    echo "  Delete $DONE_MARKER to force a re-download."
    exit 0
fi

# ---------------------------------------------------------------------------
# Create output directory
# ---------------------------------------------------------------------------
mkdir -p "$LABELS_DIR"

# ---------------------------------------------------------------------------
# Sparse clone
# ---------------------------------------------------------------------------
echo -e "${BOLD}──────────────────────────────────────────${RESET}"
info "Sparse-cloning label repository (no blobs yet)..."

# Remove any stale partial clone
rm -rf "$CLONE_DIR"

git clone \
    --filter=blob:none \
    --no-checkout \
    --depth=1 \
    "$LABELS_REPO" \
    "$CLONE_DIR"

info "Configuring sparse-checkout..."

git -C "$CLONE_DIR" sparse-checkout init --cone

# Set exactly the two subdirectories we need
git -C "$CLONE_DIR" sparse-checkout set \
    "csPCa_lesion_delineations/human_expert/resampled" \
    "csPCa_lesion_delineations/human_expert/Pooch25"

info "Checking out sparse paths (this may take a few minutes)..."
git -C "$CLONE_DIR" checkout

success "Sparse checkout complete."

# ---------------------------------------------------------------------------
# Flatten .nii.gz files into labels_dir
# ---------------------------------------------------------------------------
echo -e "${BOLD}──────────────────────────────────────────${RESET}"
info "Copying .nii.gz files → $LABELS_DIR ..."

python3 - <<PYEOF
import shutil
from pathlib import Path

clone_dir  = Path("$CLONE_DIR")
labels_dir = Path("$LABELS_DIR")
labels_dir.mkdir(parents=True, exist_ok=True)

sparse_paths = [
    "csPCa_lesion_delineations/human_expert/resampled",
    "csPCa_lesion_delineations/human_expert/Pooch25",
]

copied   = 0
skipped  = 0
conflict = 0

for rel in sparse_paths:
    src_dir = clone_dir / rel
    if not src_dir.is_dir():
        print(f"  [WARN] Expected directory not found: {src_dir}")
        continue

    files = sorted(src_dir.glob("*.nii.gz"))
    print(f"  {len(files):4d} files in {rel}")

    for src in files:
        dst = labels_dir / src.name
        if dst.exists():
            # File already present from the other source dir — keep the one
            # that is already there (resampled takes precedence over Pooch25
            # since it's set first in sparse_paths).
            conflict += 1
        else:
            shutil.copy2(src, dst)
            copied += 1

print(f"\n  Copied  : {copied}")
if conflict:
    print(f"  Skipped : {conflict}  (duplicate name — earlier source kept)")
print(f"  Total   : {copied + skipped}")
PYEOF

# ---------------------------------------------------------------------------
# Count result
# ---------------------------------------------------------------------------
N_LABELS="$(find "$LABELS_DIR" -name '*.nii.gz' | wc -l)"
success "Copied $N_LABELS label files to $LABELS_DIR"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
echo -e "${BOLD}──────────────────────────────────────────${RESET}"
info "Removing temporary clone ($CLONE_DIR)..."
rm -rf "$CLONE_DIR"
success "Cleanup done."

# Mark as complete
touch "$DONE_MARKER"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}──────────────────────────────────────────${RESET}"
success "Done! Labels are in: $LABELS_DIR"
echo ""
echo "  You now have images + labels. Start training:"
echo "  docker compose run --rm trainer train"
echo ""
