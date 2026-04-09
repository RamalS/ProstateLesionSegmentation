#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_dataset.sh  —  Download PI-CAI public training images
#
# Usage:
#   bash scripts/download_dataset.sh <N> [OPTIONS]
#
#   <N>            Number of folds to download (1–5).
#                  Fold indices are 0-based, so N=2 downloads fold0 + fold1.
#
# Options:
#   --out-dir DIR  Root data directory (default: ./data)
#   --keep-zip     Keep .zip files after extraction (default: delete)
#   --dry-run      Print what would be downloaded without doing anything
#   -h, --help     Show this help message
#
# Examples:
#   bash scripts/download_dataset.sh 1            # download fold0 only (~5 GB)
#   bash scripts/download_dataset.sh 5            # download all folds (~25 GB)
#   bash scripts/download_dataset.sh 2 --keep-zip # download fold0+1, keep zips
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="$REPO_ROOT/data"
KEEP_ZIP=false
DRY_RUN=false
N_FOLDS=""

BASE_URL="https://zenodo.org/records/6624726/files"
TOTAL_FOLDS=5

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
        -h|--help)    print_help ;;
        --out-dir)    OUT_DIR="$2"; shift 2 ;;
        --keep-zip)   KEEP_ZIP=true; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        [1-5])        N_FOLDS="$1"; shift ;;
        *)
            error "Unknown argument: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

if [[ -z "$N_FOLDS" ]]; then
    error "Missing required argument: <N> (number of folds to download, 1–5)"
    echo "  Example: bash scripts/download_dataset.sh 2"
    exit 1
fi

if (( N_FOLDS < 1 || N_FOLDS > TOTAL_FOLDS )); then
    error "N must be between 1 and $TOTAL_FOLDS, got: $N_FOLDS"
    exit 1
fi

# ---------------------------------------------------------------------------
# Check for downloader
# ---------------------------------------------------------------------------
if command -v wget &>/dev/null; then
    DOWNLOADER="wget"
elif command -v curl &>/dev/null; then
    DOWNLOADER="curl"
else
    error "Neither wget nor curl is installed. Install one and retry."
    exit 1
fi

info "Using downloader: $DOWNLOADER"

# ---------------------------------------------------------------------------
# Prepare directories
# ---------------------------------------------------------------------------
IMAGES_DIR="$OUT_DIR/images"

if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$IMAGES_DIR"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}PI-CAI Dataset Download${RESET}"
echo "  Folds to download : fold0 – fold$((N_FOLDS - 1))  ($N_FOLDS of $TOTAL_FOLDS)"
echo "  Output directory  : $IMAGES_DIR"
echo "  Keep zip files    : $KEEP_ZIP"
echo "  Dry run           : $DRY_RUN"
echo ""

# ---------------------------------------------------------------------------
# Download + extract loop
# ---------------------------------------------------------------------------
n_downloaded=0
n_cached=0

for (( i=0; i<N_FOLDS; i++ )); do
    FOLD_NAME="picai_public_images_fold${i}"
    ZIP_NAME="${FOLD_NAME}.zip"
    URL="${BASE_URL}/${ZIP_NAME}?download=1"
    ZIP_PATH="$OUT_DIR/$ZIP_NAME"
    FOLD_MARKER="$OUT_DIR/.${FOLD_NAME}_done"

    echo -e "${BOLD}──────────────────────────────────────────${RESET}"
    info "Fold $((i+1))/$N_FOLDS  (fold${i})  →  $ZIP_NAME"

    # Skip if already fully extracted
    if [[ -f "$FOLD_MARKER" ]]; then
        success "fold${i} already downloaded and extracted — skipping."
        (( n_cached++ )) || true
        continue
    fi

    if [[ "$DRY_RUN" == true ]]; then
        info "[dry-run] Would download: $URL"
        info "[dry-run] Would extract to: $IMAGES_DIR"
        continue
    fi

    # ---- Download ----
    info "Downloading $ZIP_NAME (~5 GB)..."

    if [[ "$DOWNLOADER" == "wget" ]]; then
        # -c  : resume interrupted download
        # --show-progress : progress bar even when not interactive
        wget -c --show-progress -O "$ZIP_PATH" "$URL"
    else
        # curl: -L follow redirects, -C - resume, --progress-bar
        curl -L -C - --progress-bar -o "$ZIP_PATH" "$URL"
    fi

    success "Download complete: $ZIP_PATH"

    # ---- Extract ----
    info "Extracting $ZIP_NAME → $IMAGES_DIR ..."

    # Use Python's zipfile module (always available, no unzip dependency)
    python3 - <<PYEOF
import zipfile, sys, os, shutil
from pathlib import Path

zip_path   = "$ZIP_PATH"
images_dir = Path("$IMAGES_DIR")
fold_name  = "$FOLD_NAME"

print(f"  Opening {zip_path} ...")
with zipfile.ZipFile(zip_path, 'r') as zf:
    members = zf.namelist()
    total   = len(members)
    print(f"  {total} entries in archive")

    for idx, member in enumerate(members, 1):
        # Strip the top-level fold directory:
        # e.g.  picai_public_images_fold0/10000/... → 10000/...
        parts = Path(member).parts
        if len(parts) < 2:
            continue  # skip the root directory entry itself

        relative = Path(*parts[1:])  # drop fold dir prefix
        target   = images_dir / relative

        if member.endswith('/'):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)

        if idx % 500 == 0 or idx == total:
            pct = idx / total * 100
            print(f"  [{idx}/{total}]  {pct:.0f}%", flush=True)

print("  Extraction complete.")
PYEOF

    success "Extracted fold${i} to $IMAGES_DIR"

    # ---- Cleanup ----
    if [[ "$KEEP_ZIP" == false ]]; then
        rm -f "$ZIP_PATH"
        info "Removed $ZIP_NAME"
    fi

    # Mark fold as done so re-runs skip it
    touch "$FOLD_MARKER"
    (( n_downloaded++ )) || true
done

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}──────────────────────────────────────────${RESET}"
if [[ "$DRY_RUN" == true ]]; then
    success "Dry run complete. No files were downloaded."
else
    success "All $N_FOLDS fold(s) now available in: $IMAGES_DIR"
    echo "  Newly downloaded : $n_downloaded fold(s)"
    echo "  Already cached   : $n_cached fold(s)"
    echo ""
    echo "  Next step: download the labels (csPCa lesion masks):"
    echo "  https://zenodo.org/records/6667655"
    echo ""
    echo "  Then run training:"
    echo "  docker compose run --rm trainer train"
fi
