#!/usr/bin/env bash
# ============================================================
# safe_sweep.sh — 안정성 최우선 스윕 실행기
# ============================================================
# 목적: host kernel hard lockup 방지.  성능보다 안정성 우선.
#
# 권장 실행:
#   tmux new -s fmnist_sweep -c /path/to/onlyextrinsic_ada_sigma
#   ./safe_sweep.sh
#
# 환경변수 — 모두 override 가능
#   ── 실행 노브
#     PY=python3   BATCH=64   ROUNDS=2   GPU_MB=2048
#     RUN_TIMEOUT=3600          # run 1개 최대 실행 시간(초)
#     GPU_TEMP_LIMIT=80         # 이 온도(°C) 이상이면 추가 대기
#
#   ── 디코더 파라미터
#     ALPHA=0.1   BETA=0.1   SIGMA=0.3
#     EBNO_LIST="0.6 0.7 0.8 0.9 1.0"
#
#   ── 캘리브레이션 단계
#     RUN_PROXY_CALIBRATE=1     # source_posterior_ber (proxy) 캘리브레이션
#     RUN_TAIL_CALIBRATE=1      # tail_final_nack (recommended) 캘리브레이션
#     RUN_CALIBRATE=1           # alias of RUN_PROXY_CALIBRATE  (구버전 호환)
#     RUN_RECALIBRATE=1         # alias of RUN_TAIL_CALIBRATE   (구버전 호환)
#
#   ── 모노토닉 σ 옵션
#     MONOTONIC_SIGMA=1         # 비교 모드(--compare-four-modes …)에 적용
# ============================================================

set -euo pipefail

# ── 경로 ──────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── CUDA / 스레드 환경변수 ────────────────────────────────
export CUDA_VISIBLE_DEVICES=0

# CPU 스레드 상한
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=4

# TF 메모리 안정성
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export TF_FORCE_GPU_ALLOW_GROWTH=1

# PyTorch 메모리 파편화 완화
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# BLAS 싱글 스레드 강제 (중복 보호)
export GOTO_NUM_THREADS=4
export NUMBA_NUM_THREADS=4

# ── 실행 노브 ──────────────────────────────────────────────
PY="${PY:-python3}"
BATCH="${BATCH:-64}"
ROUNDS="${ROUNDS:-2}"
GPU_MB="${GPU_MB:-2048}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"
GPU_TEMP_LIMIT="${GPU_TEMP_LIMIT:-80}"
CSV="${CSV:-results/safe_sweep_scalar.csv}"
LOG="${LOG:-results/safe_sweep_run.log}"
SIG_JSON="${SIG_JSON:-results/sigma_lookup_calibrated.json}"
TAIL_JSON="${TAIL_JSON:-results/sigma_lookup_tail_nack.json}"

# ── 디코더 파라미터 (ALPHA / BETA / SIGMA / EBNO_LIST) ────
ALPHA="${ALPHA:-0.1}"
BETA="${BETA:-0.1}"
SIGMA="${SIGMA:-0.3}"
EBNO_LIST="${EBNO_LIST:-0.6 0.7 0.8 0.9 1.0}"

# ── Monotonic σ 옵션 ───────────────────────────────────────
MONOTONIC_FLAG=()
if [[ "${MONOTONIC_SIGMA:-0}" == "1" ]]; then
    MONOTONIC_FLAG=(--monotonic-sigma)
fi

# ── α/β 태그 (CSV sweep_tag 에 추기) ──────────────────────
TAG_AB="a${ALPHA}_b${BETA}"

mkdir -p results

# ── 파일 디스크립터 상한 ──────────────────────────────────
ulimit -n 4096 2>/dev/null || true

# ── 로깅 헬퍼 ─────────────────────────────────────────────
log() { echo "$(date -uIs) $*" | tee -a "$LOG"; }

# ── 자식 프로세스 추적 및 종료 trap ──────────────────────
_CHILD_PID=""
cleanup_on_exit() {
    local sig="${1:-EXIT}"
    log "=== TRAP $sig: cleaning up ==="
    if [[ -n "$_CHILD_PID" ]] && kill -0 "$_CHILD_PID" 2>/dev/null; then
        log "Killing child PID $_CHILD_PID ..."
        kill -TERM "$_CHILD_PID" 2>/dev/null || true
        sleep 3
        kill -KILL "$_CHILD_PID" 2>/dev/null || true
    fi
    "$PY" cuda_cleanup.py 2>&1 | tee -a "$LOG" || true
    log "=== safe_sweep exiting due to $sig ==="
}
trap 'cleanup_on_exit EXIT'  EXIT
trap 'cleanup_on_exit INT;  exit 130' INT
trap 'cleanup_on_exit TERM; exit 143' TERM

# ── 중복 실행 방지 ────────────────────────────────────────
if pgrep -f "plot_comparison\.py|calibrate_sigma_lookup\.py|sigma_sweep\.py|experiment\.py" \
      >/dev/null 2>&1; then
    log "ERROR: 이미 sweep 관련 Python 프로세스가 실행 중입니다."
    exit 1
fi

# ── nvidia-smi 스냅샷 ─────────────────────────────────────
smi_log() {
    local label="$1"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi \
            --query-gpu=index,memory.used,memory.free,temperature.gpu,power.draw,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null \
        | while IFS= read -r line; do
            log "[smi/$label] $line"
          done || true
    fi
}

# ── GPU 온도 초과 시 추가 대기 ────────────────────────────
wait_for_cool() {
    if ! command -v nvidia-smi &>/dev/null; then return; fi
    while true; do
        local temp
        temp=$(nvidia-smi --query-gpu=temperature.gpu \
                          --format=csv,noheader,nounits 2>/dev/null \
               | head -1 | tr -d ' ' || echo "0")
        if [[ "$temp" -lt "$GPU_TEMP_LIMIT" ]] 2>/dev/null; then
            break
        fi
        log "[thermal] GPU ${temp}°C >= ${GPU_TEMP_LIMIT}°C 한계. 30s 추가 대기..."
        sleep 30
    done
}

# ── run 사이 슬립 ─────────────────────────────────────────
between_runs() {
    local s=$((10 + RANDOM % 11))
    log "--- sleeping ${s}s (CUDA context cooldown) ---"
    sleep "$s"
    log "--- cuda_cleanup.py ---"
    "$PY" cuda_cleanup.py 2>&1 | tee -a "$LOG" || true
    smi_log "after_cleanup"
    wait_for_cool
}

# ── 단일 run 실행 (timeout + 자식 PID 추적) ──────────────
run_one() {
    local tag="$1"
    shift
    log ">>> START $tag"
    smi_log "pre_${tag}"

    set +e
    timeout "$RUN_TIMEOUT" "$PY" "$@" 2>&1 | tee -a "$LOG"
    local exit_code="${PIPESTATUS[0]}"
    set -e

    if [[ "$exit_code" -eq 124 ]]; then
        log "!!! TIMEOUT ($RUN_TIMEOUT s) reached for $tag — 강제 종료됨"
        "$PY" cuda_cleanup.py 2>&1 | tee -a "$LOG" || true
        exit 1
    elif [[ "$exit_code" -ne 0 ]]; then
        log "!!! FAILED (exit $exit_code) for $tag"
        exit "$exit_code"
    fi

    log "<<< END $tag (exit 0)"
    between_runs
}

# ── 시작 배너 ─────────────────────────────────────────────
log "======== safe_sweep start ========"
log "PY=$PY  BATCH=$BATCH  ROUNDS=$ROUNDS  GPU_MB=$GPU_MB"
log "ALPHA=$ALPHA  BETA=$BETA  SIGMA=$SIGMA  EBNO_LIST=\"$EBNO_LIST\""
log "MONOTONIC_SIGMA=${MONOTONIC_SIGMA:-0}"
log "RUN_TIMEOUT=${RUN_TIMEOUT}s  GPU_TEMP_LIMIT=${GPU_TEMP_LIMIT}°C"
log "CSV=$CSV  SIG_JSON=$SIG_JSON  TAIL_JSON=$TAIL_JSON"
smi_log "start"

# ──────────────────────────────────────────────────────────
# (선택) 프록시 캘리브레이션 — RUN_PROXY_CALIBRATE / RUN_CALIBRATE
#   목적: source_posterior_ber 기반 빠른 lookup 생성 (proxy)
# ──────────────────────────────────────────────────────────
if [[ "${RUN_PROXY_CALIBRATE:-${RUN_CALIBRATE:-0}}" == "1" ]]; then
    run_one calibrate_proxy \
        calibrate_sigma_lookup.py \
        --gpu-memory-mb "$GPU_MB" \
        --alpha "$ALPHA" --beta "$BETA" \
        --calibration-objective source_posterior_ber \
        --collection-sigma "$SIGMA" --trace-tail-sigma "$SIGMA" \
        --candidate-sigmas 0.20 0.25 0.30 0.35 0.40 \
        --sigma-min 0.20 --sigma-max 0.40 \
        --batch "$BATCH" --rounds "$ROUNDS" \
        --ebno $EBNO_LIST \
        "${MONOTONIC_FLAG[@]}" \
        --output-json "$SIG_JSON"
fi

# ──────────────────────────────────────────────────────────
# (권장) tail_final_nack 캘리브레이션 — RUN_TAIL_CALIBRATE / RUN_RECALIBRATE
#   sigma 후보를 [0.20, 0.40] 좁은 범위로 제한 + 최종 NACK 최소화
#   per-chunk lookup (default).  source_posterior_ber 결과 대신 이걸 권장.
# ──────────────────────────────────────────────────────────
if [[ "${RUN_TAIL_CALIBRATE:-${RUN_RECALIBRATE:-0}}" == "1" ]]; then
    run_one calibrate_tail_nack \
        calibrate_sigma_lookup.py \
        --gpu-memory-mb "$GPU_MB" \
        --alpha "$ALPHA" --beta "$BETA" \
        --calibration-objective tail_final_nack \
        --collection-sigma "$SIGMA" --trace-tail-sigma "$SIGMA" \
        --candidate-sigmas 0.20 0.25 0.30 0.35 0.40 \
        --sigma-min 0.20 --sigma-max 0.40 \
        --batch "$BATCH" --rounds "$ROUNDS" \
        --ebno $EBNO_LIST \
        "${MONOTONIC_FLAG[@]}" \
        --output-json "$TAIL_JSON"
fi

# ──────────────────────────────────────────────────────────
# 1) baseline + fixed σ  (PNG 생략, 스칼라만 CSV)
# ──────────────────────────────────────────────────────────
run_one plot_baseline_fixed \
    plot_comparison.py \
    --gpu-memory-mb "$GPU_MB" \
    --alpha "$ALPHA" --beta "$BETA" --sigma "$SIGMA" \
    --ebno $EBNO_LIST \
    --batch "$BATCH" --rounds "$ROUNDS" \
    --no-save-plot \
    --append-csv "$CSV" \
    --sweep-tag "baseline_fixed_${TAG_AB}"

# ──────────────────────────────────────────────────────────
# 2) baseline + fixed + adaptive (hand-crafted)
# ──────────────────────────────────────────────────────────
run_one plot_compare_adaptive \
    plot_comparison.py \
    --gpu-memory-mb "$GPU_MB" \
    --alpha "$ALPHA" --beta "$BETA" --sigma "$SIGMA" \
    --ebno $EBNO_LIST \
    --batch "$BATCH" --rounds "$ROUNDS" \
    --compare-adaptive \
    "${MONOTONIC_FLAG[@]}" \
    --no-save-plot \
    --append-csv "$CSV" \
    --sweep-tag "compare_adaptive_${TAG_AB}"

# ──────────────────────────────────────────────────────────
# 3) 4 모드 — 기존 SIG_JSON 사용 (있을 때만)
# ──────────────────────────────────────────────────────────
if [[ -f "$SIG_JSON" ]]; then
    run_one plot_four_modes_proxy \
        plot_comparison.py \
        --gpu-memory-mb "$GPU_MB" \
        --alpha "$ALPHA" --beta "$BETA" --sigma "$SIGMA" \
        --ebno $EBNO_LIST \
        --batch "$BATCH" --rounds "$ROUNDS" \
        --compare-four-modes \
        --sigma-lookup-json "$SIG_JSON" \
        "${MONOTONIC_FLAG[@]}" \
        --no-save-plot \
        --append-csv "$CSV" \
        --sweep-tag "four_modes_proxy_${TAG_AB}"
else
    log "SKIP four_modes_proxy: $SIG_JSON 없음 (RUN_PROXY_CALIBRATE=1 로 생성)"
fi

# ──────────────────────────────────────────────────────────
# 4) 4 모드 — tail_final_nack 캘리브 JSON 사용 (있을 때만)
#    sigma clipping [0.20, 0.40] 적용
# ──────────────────────────────────────────────────────────
if [[ -f "$TAIL_JSON" ]]; then
    run_one plot_four_modes_tail \
        plot_comparison.py \
        --gpu-memory-mb "$GPU_MB" \
        --alpha "$ALPHA" --beta "$BETA" --sigma "$SIGMA" \
        --ebno $EBNO_LIST \
        --batch "$BATCH" --rounds "$ROUNDS" \
        --compare-four-modes \
        --sigma-lookup-json "$TAIL_JSON" \
        --sigma-min 0.20 --sigma-max 0.40 \
        "${MONOTONIC_FLAG[@]}" \
        --no-save-plot \
        --append-csv "$CSV" \
        --sweep-tag "four_modes_tail_nack_${TAG_AB}"
else
    log "SKIP four_modes_tail: $TAIL_JSON 없음 (RUN_TAIL_CALIBRATE=1 로 생성)"
fi

# ──────────────────────────────────────────────────────────
log "======== safe_sweep finished OK ========"
smi_log "finish"
