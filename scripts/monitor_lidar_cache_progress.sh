#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 LIDAR_CACHE_DIR [TARGET_COUNT] [INTERVAL_SECONDS] [LOG_PATH]" >&2
  exit 2
fi

cache_dir="$1"
target_count="${2:-13900}"
interval_seconds="${3:-300}"
log_path="${4:-/data2/waymo_paired_latents/logs/lidar_cache_progress.log}"

mkdir -p "$(dirname "$log_path")"

while true; do
  count="$(find "$cache_dir" -maxdepth 1 -type f -name "*.pt" | wc -l)"
  pct="$(awk -v count="$count" -v target="$target_count" 'BEGIN { if (target > 0) printf "%.2f", 100.0 * count / target; else printf "0.00" }')"
  active_shards="$(tmux ls 2>/dev/null | grep -c "lidar_cache_real_train_14s_s" || true)"
  timestamp="$(date "+%Y-%m-%d %H:%M:%S")"

  echo "[$timestamp] count=$count/$target_count pct=$pct active_shards=$active_shards" | tee -a "$log_path"

  if [[ "$count" -ge "$target_count" ]]; then
    exit 0
  fi
  sleep "$interval_seconds"
done
