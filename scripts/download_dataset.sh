#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_dataset.sh - Download PI-CAI data + Prostate158 unlabeled images
#
# Usage:
#   bash scripts/download_dataset.sh [N] [OPTIONS]
#
#   [N]            Optional number of PI-CAI image folds to download (1-5).
#                  Fold indices are 0-based, so N=2 downloads fold0 + fold1.
#                  Default: 5
#
# Options:
#   --out-dir DIR   Root data directory (default: ./data)
#   --keep-zip      Keep image .zip files after extraction (default: delete)
#   --no-images     Skip PI-CAI image download/extraction
#   --no-labels     Skip label download
#   --no-unlabeled  Skip Prostate158 unlabeled image download
#   --images-only   Alias for --no-labels
#   --labels-only   Alias for --no-images --no-unlabeled
#   --dry-run       Print what would be downloaded without doing anything
#   -h, --help      Show this help message
#
# Examples:
#   bash scripts/download_dataset.sh
#   bash scripts/download_dataset.sh 2
#   bash scripts/download_dataset.sh --images-only
#   bash scripts/download_dataset.sh --no-unlabeled
#   docker compose run --rm trainer download
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Fixed test-set case IDs
# These 10 cases are permanently reserved for evaluation and are always
# segregated into data/test_images/ - they are never mixed into the training
# pool. Five have confirmed lesions (positive) and five are negative.
# ---------------------------------------------------------------------------
TEST_CASE_IDS=(
    11377_1001400   # positive
    10418_1000426   # positive
    10059_1000059   # positive
    10760_1000776   # positive
    10726_1000742   # positive
    10352_1000358   # negative
    10189_1000192   # negative
    11221_1001244   # negative
    10142_1000144   # negative
    11151_1001174   # negative
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="$REPO_ROOT/data"
KEEP_ZIP=false
DRY_RUN=false
DOWNLOAD_IMAGES=true
DOWNLOAD_LABELS=true
DOWNLOAD_UNLABELED=true
N_FOLDS=5

IMAGE_BASE_URL="https://zenodo.org/records/6624726/files"
TOTAL_FOLDS=5
UNLABELED_ZIP_URL="https://zenodo.org/records/6481141/files/prostate158_train.zip?download=1"
UNLABELED_ZIP_NAME="prostate158_train.zip"

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
# Helpers
# ---------------------------------------------------------------------------
print_help() {
    cat <<'EOF'
Usage:
  bash scripts/download_dataset.sh [N] [OPTIONS]

  [N]            Optional number of PI-CAI image folds to download (1-5).
                 Fold indices are 0-based, so N=2 downloads fold0 + fold1.
                 Default: 5

Options:
  --out-dir DIR   Root data directory (default: ./data)
  --keep-zip      Keep image .zip files after extraction (default: delete)
  --no-images     Skip PI-CAI image download/extraction
  --no-labels     Skip label download
  --no-unlabeled  Skip Prostate158 unlabeled image download
  --images-only   Alias for --no-labels
  --labels-only   Alias for --no-images --no-unlabeled
  --dry-run       Print what would be downloaded without doing anything
  -h, --help      Show this help message

Examples:
  bash scripts/download_dataset.sh
  bash scripts/download_dataset.sh 2
  bash scripts/download_dataset.sh --images-only
  bash scripts/download_dataset.sh --no-unlabeled
  docker compose run --rm trainer download
EOF
    exit 0
}

require_python3() {
    if ! command -v python3 &>/dev/null; then
        error "python3 is required but not installed."
        exit 1
    fi
}

detect_downloader() {
    if command -v wget &>/dev/null; then
        DOWNLOADER="wget"
    elif command -v curl &>/dev/null; then
        DOWNLOADER="curl"
    else
        error "Neither wget nor curl is installed. Install one and retry."
        exit 1
    fi
}

check_git_sparse_checkout_support() {
    local git_version
    local git_major
    local git_minor
    local remainder

    if ! command -v git &>/dev/null; then
        error "git is required for label download but is not installed."
        exit 1
    fi

    git_version="$(git --version | cut -d' ' -f3)"
    git_major="${git_version%%.*}"
    remainder="${git_version#*.}"
    git_minor="${remainder%%.*}"

    if (( git_major < 2 || ( git_major == 2 && git_minor < 25 ) )); then
        error "git >= 2.25 required for sparse-checkout (found $git_version)."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            ;;
        --out-dir)
            if [[ $# -lt 2 ]]; then
                error "--out-dir requires a directory argument."
                exit 1
            fi
            OUT_DIR="$2"
            shift 2
            ;;
        --keep-zip)
            KEEP_ZIP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-images)
            DOWNLOAD_IMAGES=false
            shift
            ;;
        --labels-only)
            DOWNLOAD_IMAGES=false
            DOWNLOAD_UNLABELED=false
            shift
            ;;
        --no-labels|--images-only)
            DOWNLOAD_LABELS=false
            shift
            ;;
        --no-unlabeled)
            DOWNLOAD_UNLABELED=false
            shift
            ;;
        [1-5])
            N_FOLDS="$1"
            shift
            ;;
        [0-9]*)
            error "Invalid fold count '$1'. N must be between 1 and $TOTAL_FOLDS."
            exit 1
            ;;
        *)
            error "Unknown argument: $1"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

if [[ "$DOWNLOAD_IMAGES" == false && "$DOWNLOAD_LABELS" == false && "$DOWNLOAD_UNLABELED" == false ]]; then
    error "Nothing to do: images, labels, and unlabeled images are all disabled."
    exit 1
fi

if (( N_FOLDS < 1 || N_FOLDS > TOTAL_FOLDS )); then
    error "N must be between 1 and $TOTAL_FOLDS, got: $N_FOLDS"
    exit 1
fi

if [[ "$DOWNLOAD_IMAGES" == false && "$N_FOLDS" != "5" ]]; then
    warn "Ignoring fold count ($N_FOLDS) because image download is disabled."
fi

IMAGES_DIR="$OUT_DIR/images"
TEST_IMAGES_DIR="$OUT_DIR/test_images"
LABELS_DIR="$OUT_DIR/labels"
UNLABELED_DIR="$OUT_DIR/unlabeled_images"
LABELS_CLONE_DIR="$OUT_DIR/.picai_labels_tmp"
LABELS_DONE_MARKER="$OUT_DIR/.labels_done"
UNLABELED_DONE_MARKER="$OUT_DIR/.prostate158_unlabeled_done"

if [[ "$DRY_RUN" == false ]]; then
    require_python3
    if [[ "$DOWNLOAD_IMAGES" == true || "$DOWNLOAD_UNLABELED" == true ]]; then
        detect_downloader
        info "Using downloader: $DOWNLOADER"
    fi
    if [[ "$DOWNLOAD_LABELS" == true ]]; then
        check_git_sparse_checkout_support
    fi
fi

# ---------------------------------------------------------------------------
# segregate_test_cases - move fixed test-set images into test_images/
#
# Called before image download to handle cases already extracted into
# data/images/, and after each fold extraction to relocate any newly extracted
# test-case files.
#
# Labels are NOT moved - they remain in data/labels/ and are looked up by
# case ID as usual.
# ---------------------------------------------------------------------------
segregate_test_cases() {
    [[ "$DRY_RUN" == true ]] && return

    local moved=0
    local already=0
    local case_id
    local -a files=()

    shopt -s nullglob
    for case_id in "${TEST_CASE_IDS[@]}"; do
        files=("$TEST_IMAGES_DIR/${case_id}_"*.mha)
        if (( ${#files[@]} > 0 )); then
            (( already++ )) || true
            continue
        fi

        files=("$IMAGES_DIR/${case_id}_"*.mha)
        if (( ${#files[@]} > 0 )); then
            mv "${files[@]}" "$TEST_IMAGES_DIR/"
            info "Test case $case_id -> test_images/ (moved ${#files[@]} files)"
            (( moved++ )) || true
        fi
    done
    shopt -u nullglob

    if (( moved > 0 || already > 0 )); then
        success "Test-set segregation: $moved case(s) moved, $already already in test_images/"
    fi
}

download_images() {
    local n_downloaded=0
    local n_cached=0
    local i
    local fold_name
    local zip_name
    local url
    local zip_path
    local fold_marker

    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$IMAGES_DIR"
        mkdir -p "$TEST_IMAGES_DIR"
    fi

    segregate_test_cases

    echo ""
    echo -e "${BOLD}PI-CAI Image Download${RESET}"
    echo "  Folds to download : fold0 - fold$((N_FOLDS - 1))  ($N_FOLDS of $TOTAL_FOLDS)"
    echo "  Output directory  : $IMAGES_DIR"
    echo "  Keep zip files    : $KEEP_ZIP"
    echo "  Dry run           : $DRY_RUN"
    echo ""

    for (( i=0; i<N_FOLDS; i++ )); do
        fold_name="picai_public_images_fold${i}"
        zip_name="${fold_name}.zip"
        url="${IMAGE_BASE_URL}/${zip_name}?download=1"
        zip_path="$OUT_DIR/$zip_name"
        fold_marker="$OUT_DIR/.${fold_name}_done"

        echo -e "${BOLD}------------------------------------------${RESET}"
        info "Fold $((i + 1))/$N_FOLDS (fold${i}) -> $zip_name"

        if [[ -f "$fold_marker" ]]; then
            success "fold${i} already downloaded and extracted - skipping."
            (( n_cached++ )) || true
            continue
        fi

        if [[ "$DRY_RUN" == true ]]; then
            info "[dry-run] Would download: $url"
            info "[dry-run] Would extract to: $IMAGES_DIR"
            continue
        fi

        info "Downloading $zip_name (~5 GB)..."
        if [[ "$DOWNLOADER" == "wget" ]]; then
            wget -c --show-progress -O "$zip_path" "$url"
        else
            curl -L -C - --progress-bar -o "$zip_path" "$url"
        fi
        success "Download complete: $zip_path"

        info "Extracting $zip_name -> $IMAGES_DIR ..."
        python3 - <<PYEOF
import shutil
import zipfile
from pathlib import Path

zip_path = "$zip_path"
images_dir = Path("$IMAGES_DIR")

print(f"  Opening {zip_path} ...")
with zipfile.ZipFile(zip_path, "r") as zf:
    members = zf.namelist()
    total = len(members)
    print(f"  {total} entries in archive")

    for idx, member in enumerate(members, 1):
        parts = Path(member).parts
        if len(parts) < 2:
            continue

        relative = Path(*parts[1:])
        target = images_dir / relative

        if member.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

        if idx % 500 == 0 or idx == total:
            pct = idx / total * 100
            print(f"  [{idx}/{total}]  {pct:.0f}%", flush=True)

print("  Extraction complete.")
PYEOF

        success "Extracted fold${i} to $IMAGES_DIR"

        segregate_test_cases

        if [[ "$KEEP_ZIP" == false ]]; then
            rm -f "$zip_path"
            info "Removed $zip_name"
        fi

        touch "$fold_marker"
        (( n_downloaded++ )) || true
    done

    echo ""
    echo -e "${BOLD}------------------------------------------${RESET}"
    if [[ "$DRY_RUN" == true ]]; then
        success "Image dry run complete."
    else
        success "Image download complete."
        echo "  Newly downloaded : $n_downloaded fold(s)"
        echo "  Already cached   : $n_cached fold(s)"
        echo "  Images           : $IMAGES_DIR"
        echo "  Test images      : $TEST_IMAGES_DIR"
    fi
}

download_unlabeled_images() {
    local zip_name
    local zip_path
    local n_files

    zip_name="$UNLABELED_ZIP_NAME"
    zip_path="$OUT_DIR/$zip_name"

    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$UNLABELED_DIR"
    fi

    echo ""
    echo -e "${BOLD}Prostate158 Unlabeled Download${RESET}"
    echo "  Source            : $UNLABELED_ZIP_URL"
    echo "  Output directory  : $UNLABELED_DIR"
    echo "  Keep zip files    : $KEEP_ZIP"
    echo "  Dry run           : $DRY_RUN"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        info "[dry-run] Would download: $UNLABELED_ZIP_URL"
        info "[dry-run] Would extract only t2/adc/dwi from prostate158_train/train/<case_id>/"
        info "[dry-run] Would flatten to: <case_id>_{t2,adc,dwi}.nii.gz"
        if [[ "$KEEP_ZIP" == false ]]; then
            info "[dry-run] Would remove: $zip_path"
        fi
        success "Unlabeled image dry run complete."
        return
    fi

    if [[ -f "$UNLABELED_DONE_MARKER" ]]; then
        n_files="$(find "$UNLABELED_DIR" -name '*.nii.gz' 2>/dev/null | wc -l | tr -d ' ')"
        success "Unlabeled images already downloaded ($n_files .nii.gz files in $UNLABELED_DIR) - skipping."
        echo "  Delete $UNLABELED_DONE_MARKER to force a re-download."
        return
    fi

    echo -e "${BOLD}------------------------------------------${RESET}"
    info "Downloading $zip_name (~2.8 GB)..."
    if [[ "$DOWNLOADER" == "wget" ]]; then
        wget -c --show-progress -O "$zip_path" "$UNLABELED_ZIP_URL"
    else
        curl -L -C - --progress-bar -o "$zip_path" "$UNLABELED_ZIP_URL"
    fi
    success "Download complete: $zip_path"

    echo -e "${BOLD}------------------------------------------${RESET}"
    info "Extracting t2/adc/dwi files -> $UNLABELED_DIR (flattened) ..."

    python3 - <<PYEOF
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

zip_path = "$zip_path"
out_dir = Path("$UNLABELED_DIR")
out_dir.mkdir(parents=True, exist_ok=True)

wanted = {
    "t2.nii.gz": "t2",
    "adc.nii.gz": "adc",
    "dwi.nii.gz": "dwi",
}

written = 0
already_present = 0
ignored = 0
case_modalities = defaultdict(set)

print(f"  Opening {zip_path} ...")
with zipfile.ZipFile(zip_path, "r") as zf:
    members = zf.namelist()
    total = len(members)
    print(f"  {total} entries in archive")

    for idx, member in enumerate(members, 1):
        if idx % 1000 == 0 or idx == total:
            pct = idx / total * 100
            print(f"  [{idx}/{total}]  {pct:.0f}%", flush=True)

        path = PurePosixPath(member)
        parts = path.parts

        if len(parts) != 4:
            ignored += 1
            continue
        if parts[0] != "prostate158_train" or parts[1] != "train":
            ignored += 1
            continue

        case_id = parts[2]
        filename = parts[3]

        if not (case_id.isdigit() and 20 <= int(case_id) <= 158):
            ignored += 1
            continue

        modality = wanted.get(filename)
        if modality is None:
            ignored += 1
            continue

        target = out_dir / f"{case_id}_{modality}.nii.gz"
        if target.exists():
            already_present += 1
            case_modalities[case_id].add(modality)
            continue

        with zf.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)

        written += 1
        case_modalities[case_id].add(modality)

cases = sorted(case_modalities.keys(), key=int)
complete_cases = sum(1 for cid in cases if len(case_modalities[cid]) == 3)
incomplete_cases = [cid for cid in cases if len(case_modalities[cid]) != 3]

print("  Extraction complete.")
print(f"  Cases found      : {len(cases)}")
print(f"  Complete cases   : {complete_cases}")
print(f"  Files written    : {written}")
print(f"  Files pre-existing: {already_present}")
print(f"  Archive entries ignored: {ignored}")

if incomplete_cases:
    preview = ", ".join(incomplete_cases[:10])
    suffix = " ..." if len(incomplete_cases) > 10 else ""
    print(f"  [WARN] Incomplete modality set for: {preview}{suffix}")
PYEOF

    n_files="$(find "$UNLABELED_DIR" -name '*.nii.gz' | wc -l | tr -d ' ')"
    success "Prepared $n_files unlabeled files in $UNLABELED_DIR"

    if [[ "$KEEP_ZIP" == false ]]; then
        rm -f "$zip_path"
        info "Removed $zip_name"
    fi

    touch "$UNLABELED_DONE_MARKER"
}

download_labels() {
    local sp
    local n_labels

    echo ""
    echo -e "${BOLD}PI-CAI Labels Download${RESET}"
    echo "  Source      : $LABELS_REPO"
    echo "  Subdirs     : ${SPARSE_PATHS[*]}"
    echo "  Output dir  : $LABELS_DIR"
    echo "  Dry run     : $DRY_RUN"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        info "[dry-run] Would sparse-clone $LABELS_REPO into $LABELS_CLONE_DIR"
        for sp in "${SPARSE_PATHS[@]}"; do
            info "[dry-run] Would fetch path: $sp"
        done
        info "[dry-run] Would copy all .nii.gz files -> $LABELS_DIR"
        info "[dry-run] Would remove $LABELS_CLONE_DIR"
        success "Label dry run complete."
        return
    fi

    if [[ -f "$LABELS_DONE_MARKER" ]]; then
        n_labels="$(find "$LABELS_DIR" -name '*.nii.gz' 2>/dev/null | wc -l | tr -d ' ')"
        success "Labels already downloaded ($n_labels .nii.gz files in $LABELS_DIR) - skipping."
        echo "  Delete $LABELS_DONE_MARKER to force a re-download."
        return
    fi

    mkdir -p "$LABELS_DIR"

    echo -e "${BOLD}------------------------------------------${RESET}"
    info "Sparse-cloning label repository (no blobs yet)..."

    rm -rf "$LABELS_CLONE_DIR"

    git clone \
        --filter=blob:none \
        --no-checkout \
        --depth=1 \
        "$LABELS_REPO" \
        "$LABELS_CLONE_DIR"

    info "Configuring sparse-checkout..."
    git -C "$LABELS_CLONE_DIR" sparse-checkout init --cone
    git -C "$LABELS_CLONE_DIR" sparse-checkout set "${SPARSE_PATHS[@]}"

    info "Checking out sparse paths (this may take a few minutes)..."
    git -C "$LABELS_CLONE_DIR" checkout
    success "Sparse checkout complete."

    echo -e "${BOLD}------------------------------------------${RESET}"
    info "Copying .nii.gz files -> $LABELS_DIR ..."

    python3 - <<PYEOF
import shutil
from pathlib import Path

clone_dir = Path("$LABELS_CLONE_DIR")
labels_dir = Path("$LABELS_DIR")
labels_dir.mkdir(parents=True, exist_ok=True)

sparse_paths = [
    "csPCa_lesion_delineations/human_expert/resampled",
    "csPCa_lesion_delineations/human_expert/Pooch25",
]

copied = 0
duplicates = 0

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
            duplicates += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

print(f"\n  Copied    : {copied}")
if duplicates:
    print(f"  Duplicates: {duplicates} (earlier source kept)")
print(f"  Total     : {copied + duplicates}")
PYEOF

    n_labels="$(find "$LABELS_DIR" -name '*.nii.gz' | wc -l | tr -d ' ')"
    success "Copied $n_labels label files to $LABELS_DIR"

    echo -e "${BOLD}------------------------------------------${RESET}"
    info "Removing temporary clone ($LABELS_CLONE_DIR)..."
    rm -rf "$LABELS_CLONE_DIR"
    success "Cleanup done."

    touch "$LABELS_DONE_MARKER"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Dataset Download${RESET}"
echo "  Output root      : $OUT_DIR"
echo "  Download images  : $DOWNLOAD_IMAGES"
if [[ "$DOWNLOAD_IMAGES" == true ]]; then
    echo "  Image folds      : fold0 - fold$((N_FOLDS - 1))  ($N_FOLDS of $TOTAL_FOLDS)"
fi
echo "  Download labels  : $DOWNLOAD_LABELS"
echo "  Download unlabeled: $DOWNLOAD_UNLABELED"
echo "  Keep zip files   : $KEEP_ZIP"
echo "  Dry run          : $DRY_RUN"

if [[ "$DOWNLOAD_IMAGES" == true ]]; then
    download_images
else
    info "Skipping image download."
fi

if [[ "$DOWNLOAD_LABELS" == true ]]; then
    download_labels
else
    info "Skipping label download."
fi

if [[ "$DOWNLOAD_UNLABELED" == true ]]; then
    download_unlabeled_images
else
    info "Skipping unlabeled image download."
fi

echo ""
echo -e "${BOLD}------------------------------------------${RESET}"
if [[ "$DRY_RUN" == true ]]; then
    success "Dry run complete. No files were downloaded."
else
    success "Done."
    if [[ "$DOWNLOAD_IMAGES" == true ]]; then
        echo "  Images      : $IMAGES_DIR"
        echo "  Test images : $TEST_IMAGES_DIR"
    fi
    if [[ "$DOWNLOAD_LABELS" == true ]]; then
        echo "  Labels      : $LABELS_DIR"
    fi
    if [[ "$DOWNLOAD_UNLABELED" == true ]]; then
        echo "  Unlabeled   : $UNLABELED_DIR"
    fi
    echo ""
    echo "Start training with:"
    echo "  docker compose run --rm trainer train"
fi
