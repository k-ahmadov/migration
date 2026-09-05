#!/usr/bin/env bash
#
# Back up this repository to cloud storage with rclone.
#
# `rclone sync SOURCE DEST` makes DEST identical to SOURCE: files present on
# DEST but not in SOURCE are DELETED. This wrapper exists so that direction is
# written down once (local repo -> cloud) and can never be run the other way
# by accident. It also:
#   * always applies .rclone-filter (so .git/, caches, Quarto junk stay out);
#   * routes every file the remote would delete or overwrite into a
#     timestamped trash folder, so a mistake is recoverable;
#   * writes a single-file snapshot of the full git history alongside the
#     data, restorable with `git clone history.bundle`.
#
# Usage:
#   scripts/backup.sh            # dry run: report what would change, touch nothing
#   scripts/backup.sh --run      # perform the sync
#
# Configuration (environment variables):
#   BACKUP_REMOTE   rclone destination   (default: seafile:mylib/phd/backups/migration)
#   BACKUP_TRASH    where replaced/deleted remote files are moved
#                                        (default: seafile:mylib/phd/backups/migration-trash)
#
# The destination is a subfolder dedicated to this repo: `rclone sync` mirrors
# it exactly, so anything else living directly under it would be deleted. Keep
# BACKUP_REMOTE pointed at a path that holds only this backup.
#
# BACKUP_TRASH must not sit inside BACKUP_REMOTE or rclone will refuse to run
# (hence the sibling `migration-trash`, not a folder nested in `migration`).
# For a Seafile remote the top-level library must already exist (rclone will
# not create it).

set -euo pipefail

usage() { sed -n '3,31p' "$0" | sed 's/^# \{0,1\}//'; }

# --- resolve paths -----------------------------------------------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

FILTER_FILE="$REPO_DIR/.rclone-filter"
LOG_DIR="$REPO_DIR/backup"
LOG_FILE="$LOG_DIR/rclone-backup.log"
BUNDLE_FILE="$LOG_DIR/history.bundle"

BACKUP_REMOTE="${BACKUP_REMOTE:-seafile:mylib/phd/backups/migration}"
BACKUP_TRASH="${BACKUP_TRASH:-seafile:mylib/phd/backups/migration-trash}"
STAMP=$(date +%Y%m%d-%H%M%S)

# --- parse arguments -------------------------------------------------------
DRY_RUN=1
for arg in "$@"; do
  case "$arg" in
    --run|-y)     DRY_RUN=0 ;;
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "backup.sh: unknown argument '$arg' (try --help)" >&2; exit 2 ;;
  esac
done

# --- sanity checks -------------------------------------------------------
command -v rclone >/dev/null || { echo "backup.sh: rclone not found on PATH" >&2; exit 1; }
[ -f "$FILTER_FILE" ] || { echo "backup.sh: missing $FILTER_FILE" >&2; exit 1; }

remote_name=${BACKUP_REMOTE%%:*}
if ! rclone listremotes | grep -qx "${remote_name}:"; then
  echo "backup.sh: rclone remote '${remote_name}:' is not configured (run: rclone config)" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

# --- assemble the rclone command ---------------------------------------
rclone_args=(
  sync "$REPO_DIR/" "$BACKUP_REMOTE"
  --filter-from "$FILTER_FILE"
  --backup-dir "$BACKUP_TRASH/$STAMP"
  --track-renames
  --create-empty-src-dirs
  --stats 30s
  --stats-one-line
)

hint_on_failure() {
  echo >&2
  echo "backup.sh: rclone failed (see above / $LOG_FILE)." >&2
  echo "  If it says \"library ... was not found\", set BACKUP_REMOTE to an" >&2
  echo "  existing destination, e.g.  export BACKUP_REMOTE=seafile:Backups/migration" >&2
}

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY RUN -- nothing will be uploaded, deleted, or bundled."
  echo "  source : $REPO_DIR"
  echo "  dest   : $BACKUP_REMOTE"
  echo "  re-run with --run to apply."
  echo
  rclone "${rclone_args[@]}" --dry-run --verbose || { hint_on_failure; exit 1; }
  exit 0
fi

# --- real run ----------------------------------------------------------
echo "git bundle -> $BUNDLE_FILE"
git -C "$REPO_DIR" bundle create "$BUNDLE_FILE" --all

echo "sync $REPO_DIR -> $BACKUP_REMOTE"
echo "  replaced/deleted remote files go to $BACKUP_TRASH/$STAMP"
rclone "${rclone_args[@]}" --log-file "$LOG_FILE" --log-level INFO \
  || { hint_on_failure; exit 1; }
echo "done -- log at $LOG_FILE"
