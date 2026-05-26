#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TARGET_PER_CLASS="${TARGET_PER_CLASS:-100000}"
RUN_NAME="${RUN_NAME:-hst_100k_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/data/${RUN_NAME}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"

LENS_ROOT="${LENS_ROOT:-${OUTPUT_ROOT}/lens}"
NONLENS_ROOT="${NONLENS_ROOT:-${OUTPUT_ROOT}/nonlens}"

LENS_INITIAL_RUNS="${LENS_INITIAL_RUNS:-44}"
LENS_TOPUP_RUNS="${LENS_TOPUP_RUNS:-4}"
LENS_WORKERS="${LENS_WORKERS:-3}"
LENS_SKY_AREA="${LENS_SKY_AREA:-100}"
LENS_SKY_AREA_GALAXIES="${LENS_SKY_AREA_GALAXIES:-10}"

NONLENS_BATCH_SIZE="${NONLENS_BATCH_SIZE:-500}"
NONLENS_INITIAL_RUNS="${NONLENS_INITIAL_RUNS:-$(((TARGET_PER_CLASS + NONLENS_BATCH_SIZE - 1) / NONLENS_BATCH_SIZE))}"
NONLENS_WORKERS="${NONLENS_WORKERS:-5}"
NONLENS_SKY_AREA="${NONLENS_SKY_AREA:-8}"
NONLENS_SKY_AREA_FULL="${NONLENS_SKY_AREA_FULL:-8}"

RESUME_LENS_OFFSET="${RESUME_LENS_OFFSET:-0}"
RESUME_NONLENS_OFFSET="${RESUME_NONLENS_OFFSET:-0}"
RESUME_NONLENS_START_INDEX="${RESUME_NONLENS_START_INDEX:-$((RESUME_NONLENS_OFFSET * NONLENS_BATCH_SIZE))}"

DTYPE="${DTYPE:-float16}"
ZARR_CLEVEL="${ZARR_CLEVEL:-7}"
ZARR_CHUNK_SAMPLES="${ZARR_CHUNK_SAMPLES:-32}"
WRITE_BATCH="${WRITE_BATCH:-32}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"
MIN_AVAILABLE_GB="${MIN_AVAILABLE_GB:-35}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export TMPDIR="${TMPDIR:-${SCRIPT_DIR}/.tmp_hst_100k}"

mkdir -p "${LOG_ROOT}" "${LENS_ROOT}" "${NONLENS_ROOT}" "${TMPDIR}"

HST_ARGS=(--enable-hst)
if [[ -n "${HST_COSMOS_PATH_OVERRIDE:-}" ]]; then
  HST_ARGS+=(--hst-cosmos-path "${HST_COSMOS_PATH_OVERRIDE}")
elif [[ -n "${HST_COSMOS_PATH:-}" ]]; then
  HST_ARGS+=(--hst-cosmos-path "${HST_COSMOS_PATH}")
fi

count_rows() {
  local csv_path="$1"
  if [[ ! -f "${csv_path}" ]]; then
    echo 0
    return 0
  fi
  awk 'END { if (NR > 0) print NR - 1; else print 0 }' "${csv_path}"
}

mem_available_gb() {
  awk '/MemAvailable:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo
}

matching_rss_gb() {
  ps -eo rss=,cmd= | awk '
    /LRE_gg_lens.py|LRE_gg_nonlens.py/ && !/awk/ { sum += $1 }
    END { printf "%.1f", sum / 1048576 }
  '
}

disk_line() {
  df -h "${OUTPUT_ROOT}" | awk 'NR == 2 { print "disk_avail=" $4 " disk_use=" $5 " fs=" $1 }'
}

summarize_progress() {
  local lens_count nonlens_count avail rss
  lens_count="$(count_rows "${LENS_ROOT}/shards/lens_meta.csv")"
  nonlens_count="$(count_rows "${NONLENS_ROOT}/shards/nonlens_meta.csv")"
  avail="$(mem_available_gb)"
  rss="$(matching_rss_gb)"
  printf '[monitor] %s lens=%s/%s nonlens=%s/%s mem_avail=%sGiB generator_rss=%sGiB %s\n' \
    "$(date '+%F %T')" \
    "${lens_count}" "${TARGET_PER_CLASS}" \
    "${nonlens_count}" "${TARGET_PER_CLASS}" \
    "${avail}" "${rss}" "$(disk_line)"
  awk -v avail="${avail}" -v min="${MIN_AVAILABLE_GB}" 'BEGIN { exit !(avail < min) }' && {
    echo "[monitor] low available memory threshold crossed: ${avail}GiB < ${MIN_AVAILABLE_GB}GiB"
  } || true
}

run_and_log() {
  local log_path="$1"
  shift
  set -o pipefail
  "$@" 2>&1 | tee "${log_path}"
}

common_args() {
  printf '%s\0' \
    --storage zarr \
    --dtype "${DTYPE}" \
    --zarr-clevel "${ZARR_CLEVEL}" \
    --zarr-chunk-samples "${ZARR_CHUNK_SAMPLES}" \
    --write-batch "${WRITE_BATCH}" \
    --log-every 200 \
    --disable-jit \
    --euclid-bands VIS \
    --lsst-bands all \
    --roman-bands F106,F129,F158 \
    --num-pix 83 \
    --lens-field-num-pix 83 \
    --field-galaxy-area-arcsec2 50
}

run_lens_wave() {
  local run_offset="$1"
  local num_runs="$2"
  local log_path="${LOG_ROOT}/lens_offset_${run_offset}_runs_${num_runs}.log"
  mapfile -d '' args < <(common_args)
  run_and_log "${log_path}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/LRE_gg_lens.py" \
    "${args[@]}" \
    "${HST_ARGS[@]}" \
    --num-runs "${num_runs}" \
    --batch-size 1 \
    --num-workers "${LENS_WORKERS}" \
    --mp-start spawn \
    --lens-sky-area "${LENS_SKY_AREA}" \
    --lens-sky-area-galaxies "${LENS_SKY_AREA_GALAXIES}" \
    --data-root "${LENS_ROOT}" \
    --seed-base 300000 \
    --run-id-offset "${run_offset}"
}

run_nonlens_wave() {
  local run_offset="$1"
  local start_index_offset="$2"
  local num_runs="$3"
  local log_path="${LOG_ROOT}/nonlens_offset_${run_offset}_runs_${num_runs}.log"
  mapfile -d '' args < <(common_args)
  run_and_log "${log_path}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/LRE_gg_nonlens.py" \
    "${args[@]}" \
    "${HST_ARGS[@]}" \
    --num-runs "${num_runs}" \
    --batch-size "${NONLENS_BATCH_SIZE}" \
    --num-workers "${NONLENS_WORKERS}" \
    --mp-start spawn \
    --nonlens-sky-area "${NONLENS_SKY_AREA}" \
    --nonlens-sky-area-full "${NONLENS_SKY_AREA_FULL}" \
    --data-root "${NONLENS_ROOT}" \
    --seed-base 400000 \
    --run-id-offset "${run_offset}" \
    --start-index-offset "${start_index_offset}"
}

wait_with_monitor() {
  local -a pids=("$@")
  local alive failed=0
  while true; do
    alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
      fi
    done
    summarize_progress
    if [[ "${alive}" == "0" ]]; then
      break
    fi
    sleep "${MONITOR_INTERVAL}"
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  return "${failed}"
}

echo "[setup] output_root=${OUTPUT_ROOT}"
echo "[setup] logs=${LOG_ROOT}"
echo "[setup] hst_cosmos_path=${HST_COSMOS_PATH_OVERRIDE:-auto}"
echo "[setup] lens: initial_runs=${LENS_INITIAL_RUNS} workers=${LENS_WORKERS} sky_area=${LENS_SKY_AREA} galaxy_sky_area=${LENS_SKY_AREA_GALAXIES}"
echo "[setup] nonlens: runs=${NONLENS_INITIAL_RUNS} batch_size=${NONLENS_BATCH_SIZE} workers=${NONLENS_WORKERS}"
echo "[setup] resume: lens_offset=${RESUME_LENS_OFFSET} nonlens_offset=${RESUME_NONLENS_OFFSET} nonlens_start_index=${RESUME_NONLENS_START_INDEX}"
echo "[setup] dtype=${DTYPE} write_batch=${WRITE_BATCH} zarr_chunk_samples=${ZARR_CHUNK_SAMPLES} zarr_clevel=${ZARR_CLEVEL}"
summarize_progress

run_lens_wave "${RESUME_LENS_OFFSET}" "${LENS_INITIAL_RUNS}" &
lens_pid=$!
run_nonlens_wave "${RESUME_NONLENS_OFFSET}" "${RESUME_NONLENS_START_INDEX}" "${NONLENS_INITIAL_RUNS}" &
nonlens_pid=$!

if ! wait_with_monitor "${lens_pid}" "${nonlens_pid}"; then
  echo "[error] initial production wave failed; inspect ${LOG_ROOT}" >&2
  exit 1
fi

next_lens_offset=$((RESUME_LENS_OFFSET + LENS_INITIAL_RUNS))
next_nonlens_offset=$((RESUME_NONLENS_OFFSET + NONLENS_INITIAL_RUNS))
next_nonlens_start=$((RESUME_NONLENS_START_INDEX + NONLENS_INITIAL_RUNS * NONLENS_BATCH_SIZE))

while true; do
  lens_total="$(count_rows "${LENS_ROOT}/shards/lens_meta.csv")"
  nonlens_total="$(count_rows "${NONLENS_ROOT}/shards/nonlens_meta.csv")"
  if (( lens_total >= TARGET_PER_CLASS && nonlens_total >= TARGET_PER_CLASS )); then
    break
  fi

  pids=()
  if (( lens_total < TARGET_PER_CLASS )); then
    echo "[topup] lens=${lens_total}/${TARGET_PER_CLASS}; launching ${LENS_TOPUP_RUNS} more lens runs at offset ${next_lens_offset}"
    run_lens_wave "${next_lens_offset}" "${LENS_TOPUP_RUNS}" &
    pids+=("$!")
    next_lens_offset=$((next_lens_offset + LENS_TOPUP_RUNS))
  fi
  if (( nonlens_total < TARGET_PER_CLASS )); then
    remaining=$((TARGET_PER_CLASS - nonlens_total))
    topup_runs=$(((remaining + NONLENS_BATCH_SIZE - 1) / NONLENS_BATCH_SIZE))
    echo "[topup] nonlens=${nonlens_total}/${TARGET_PER_CLASS}; launching ${topup_runs} more nonlens runs at offset ${next_nonlens_offset}"
    run_nonlens_wave "${next_nonlens_offset}" "${next_nonlens_start}" "${topup_runs}" &
    pids+=("$!")
    next_nonlens_start=$((next_nonlens_start + topup_runs * NONLENS_BATCH_SIZE))
    next_nonlens_offset=$((next_nonlens_offset + topup_runs))
  fi
  if ! wait_with_monitor "${pids[@]}"; then
    echo "[error] top-up wave failed; inspect ${LOG_ROOT}" >&2
    exit 1
  fi
done

summarize_progress
echo "[done] lens_csv=${LENS_ROOT}/shards/lens_meta.csv"
echo "[done] nonlens_csv=${NONLENS_ROOT}/shards/nonlens_meta.csv"
echo "[done] output_root=${OUTPUT_ROOT}"
