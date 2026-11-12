#!/usr/bin/env bash
# Run the 9 training runs of the study: {d11n, d8n, w11n} x seeds {0, 1, 2}.
#
#   GPUS="6 7" scripts/run_all_seeds.sh
#
# * GPUS (required): space-separated nvidia-smi GPU indices. One job runs per listed GPU at a
#   time; the remaining jobs wait in a queue and start as GPUs become free. There is no default
#   (GPUs are shared: never guess, never use all of them).
# * Per run: absent -> start; partial (interrupted) -> `python -m fpyolo.runstate resume`;
#   done -> skipped; broken (directory but nothing to resume from) -> reported as failed, see
#   the README. The state logic lives in src/fpyolo/runstate.py.
# * After each finished run, `python -m fpyolo.runstate verify` checks optimizer, batch (Ultralytics
#   halves it silently on a first-epoch OOM) and the GPU record in env.json.
# * Logs: <RUNS_DIR>/<name>.log (appended on resume). Run directories: <RUNS_DIR>/<name>.
#
# Environment (all optional except GPUS):
#   BATCH_D, BATCH_W   override the config batch for D (both D configs, identical) / W
#   DATA_ROOT          dataset root holding D/ and W/   (default: data/fpcad_v0)
#   RUNS_DIR           Ultralytics project directory    (default: runs)
#   PYTHON             interpreter                      (default: python)
#   DRY_RUN=1          print what every GPU slot would do (incl. skip/resume decisions) and exit;
#                      jobs are dealt round-robin, i.e. it assumes equal run times
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS=(0 1 2)

die() { echo "run_all_seeds: $*" >&2; exit 1; }

# ---- inputs -------------------------------------------------------------------------------
[ -n "${GPUS:-}" ] || die 'GPUS is required, e.g. GPUS="6 7" scripts/run_all_seeds.sh (see nvidia-smi for free GPUs)'
read -r -a GPU_LIST <<< "$GPUS"
declare -A seen_gpu=()
for g in "${GPU_LIST[@]}"; do
    [[ "$g" =~ ^[0-9]+$ ]] || die "GPUS must be space-separated integer indices, got '$g' in '$GPUS'"
    [ -z "${seen_gpu[$g]:-}" ] || die "GPU $g is listed twice in GPUS"
    seen_gpu[$g]=1
done
for v in BATCH_D BATCH_W; do
    [ -z "${!v:-}" ] || [[ "${!v}" =~ ^[1-9][0-9]*$ ]] || die "$v must be a positive integer, got '${!v}'"
done

abspath() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
DATA_ROOT="$(abspath "${DATA_ROOT:-data/fpcad_v0}")"
RUNS_DIR="$(abspath "${RUNS_DIR:-runs}")"
for kind in D W; do
    [ -f "$DATA_ROOT/$kind/data.yaml" ] || [ "$DRY_RUN" = 1 ] || die "$DATA_ROOT/$kind/data.yaml not found (scripts/fetch_dataset.sh?)"
done

# ---- job table: name|config|data.yaml|batch override ("" = config value) ------------------
JOBS=()
for s in "${SEEDS[@]}"; do
    JOBS+=("d11n_s$s|$REPO_DIR/configs/train_d_yolo11n.yaml|$DATA_ROOT/D/data.yaml|${BATCH_D:-}")
    JOBS+=("d8n_s$s|$REPO_DIR/configs/train_d_yolov8n.yaml|$DATA_ROOT/D/data.yaml|${BATCH_D:-}")
    JOBS+=("w11n_s$s|$REPO_DIR/configs/train_w_yolo11n_seg.yaml|$DATA_ROOT/W/data.yaml|${BATCH_W:-}")
done
seed_of() { local n="${1%%|*}"; printf '%s' "${n##*_s}"; }

cd "$REPO_DIR"

quote() { printf '%q ' "$@"; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ---- one job on one GPU ---------------------------------------------------------------
CHILD=""
STOP=0
# Run a command in the background with its output appended to $1, so a TERM/INT trap can forward
# the signal to it (the run then stays resumable from last.pt).
run_logged() {
    local log="$1" rc=0
    shift
    "$@" >>"$log" 2>&1 &
    CHILD=$!
    wait "$CHILD" || rc=$?
    while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD" || rc=$?; done
    CHILD=""
    return "$rc"
}

run_job() {  # run_job "<job row>" <gpu>  -> 0 ok/skipped, 1 failed
    local row="$1" gpu="$2" name config data batch run_dir log state rc=0
    IFS='|' read -r name config data batch <<< "$row"
    run_dir="$RUNS_DIR/$name"
    log="$RUNS_DIR/$name.log"
    state="$("$PYTHON" -m fpyolo.runstate state "$run_dir")"

    local -a cmd=()
    local action
    case "$state" in
        done) action=skip ;;
        broken) action=FAIL ;;
        partial)
            action=resume
            cmd=("$PYTHON" -m fpyolo.runstate resume "$run_dir" --device "$gpu") ;;
        absent)
            action=start
            cmd=("$PYTHON" -m fpyolo.train --config "$config" --data "$data" --seed "$(seed_of "$row")"
                 --project "$RUNS_DIR" --name "$name" --device "$gpu")
            [ -z "$batch" ] || cmd+=(--batch "$batch") ;;
        *) echo "[GPU $gpu] $name: unexpected state '$state'" >&2; return 1 ;;
    esac

    if [ "$DRY_RUN" = 1 ]; then
        echo "[dry-run] GPU $gpu: $name state=$state -> $action"
        [ "${#cmd[@]}" -eq 0 ] || echo "    $(quote "${cmd[@]}")>> $(quote "$log")"
        [ "$action" != FAIL ] || return 1
        return 0
    fi

    case "$action" in
        skip)
            echo "[GPU $gpu] $(stamp) $name: done already, skipping"
            return 0 ;;
        FAIL)
            echo "[GPU $gpu] $(stamp) $name: FAILED, $run_dir exists but has nothing to resume from;" \
                 "move or delete it (see README, Resume) and rerun" >&2
            return 1 ;;
    esac

    mkdir -p "$RUNS_DIR"
    echo "[GPU $gpu] $(stamp) $name: $action, log $log"
    { echo "=== $(stamp) GPU $gpu: $(quote "${cmd[@]}")"; } >>"$log"
    run_logged "$log" "${cmd[@]}" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[GPU $gpu] $(stamp) $name: FAILED (exit $rc), see $log; if it died from an OOM caused by another" \
             "user, rerun this script once memory is free (it resumes)" >&2
        return 1
    fi
    echo "[GPU $gpu] $(stamp) $name: finished, checking"
    if ! "$PYTHON" -m fpyolo.runstate verify "$run_dir" --config "$config" ${batch:+--batch "$batch"} 2>&1 | tee -a "$log"; then
        echo "[GPU $gpu] $(stamp) $name: CHECK FAILED (see PROBLEM lines above)" >&2
        return 1
    fi
    return 0
}

# ---- dry run: deal jobs round-robin, deterministic ------------------------------------------
if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] GPUs: ${GPU_LIST[*]}; runs dir: $RUNS_DIR; data root: $DATA_ROOT"
    i=0 failed=0
    for row in "${JOBS[@]}"; do
        run_job "$row" "${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}" || failed=1
        i=$((i + 1))
    done
    exit "$failed"
fi

# ---- real run: one worker per GPU pulling from a shared queue ---------------------------------
CLAIMS="$(mktemp -d)"
trap 'rm -rf "$CLAIMS"' EXIT
worker() {
    local gpu="$1" row name
    trap 'STOP=1; [ -z "$CHILD" ] || kill -TERM "$CHILD" 2>/dev/null || true' TERM INT
    for row in "${JOBS[@]}"; do
        [ "$STOP" = 0 ] || break
        name="${row%%|*}"
        mkdir "$CLAIMS/$name" 2>/dev/null || continue   # atomic claim: another GPU took it
        run_job "$row" "$gpu" || echo "$name" >>"$CLAIMS/failed"
    done
}

PIDS=()
for g in "${GPU_LIST[@]}"; do
    worker "$g" &
    PIDS+=($!)
done
trap 'kill -TERM "${PIDS[@]}" 2>/dev/null || true' TERM INT
for pid in "${PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do wait "$pid" || true; done
done

if [ -s "$CLAIMS/failed" ]; then
    echo "run_all_seeds: $(stamp) FAILED runs: $(tr '\n' ' ' < "$CLAIMS/failed")" >&2
    exit 1
fi
echo "run_all_seeds: $(stamp) all runs finished; next: evaluation (README)"
