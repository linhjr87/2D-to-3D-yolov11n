#!/usr/bin/env bash
# Run the 9 training runs of the study: {d11n, d8n, w11n} x seeds {0, 1, 2}.
#
#   GPUS="6 7" scripts/run_all_seeds.sh
#
# * GPUS (required): space-separated nvidia-smi GPU indices. One job runs per listed GPU at a
#   time; the remaining jobs wait in a queue and start as GPUs become free. There is no default
#   (GPUs are shared: never guess, never use all of them).
# * Per run: absent -> start; partial (interrupted) -> `python -m fpyolo.runstate resume`;
#   done -> skipped (but still verified); running (another process holds the run's lock) -> skipped
#   with a warning; broken (directory but nothing to resume from) -> reported as failed, see the
#   README. The state logic lives in src/fpyolo/runstate.py.
# * Exit code: 0 all runs complete and verified; 1 some run failed; 3 none failed but some run was
#   skipped because another process is training it (the failed and skipped lists are always printed).
# * Only one sweep per RUNS_DIR: the script holds flock on <RUNS_DIR>/.sweep.lock for its lifetime.
#   Each run is also locked (<run>/.fpyolo.lock) by fpyolo.train / `runstate resume`.
# * After each finished (or already done) run, `python -m fpyolo.runstate verify` checks the
#   optimizer, the effective batch from the checkpoint plus the log (Ultralytics halves batch on a
#   first-epoch OOM and args.yaml does not show it) and the GPU record in env.json.
# * Logs: <RUNS_DIR>/<name>.log, appended on resume; a fresh start first moves an existing log to
#   <name>.log.<UTC stamp>.old. Every attempt begins with a "=== fpyolo-attempt" line and `verify`
#   only scans the log after the last one. Run directories: <RUNS_DIR>/<name>.
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
[ "${#GPU_LIST[@]}" -gt 0 ] || die "GPUS contains no GPU index (got '$GPUS')"
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

verify_job() {  # verify_job <name> <run_dir> <config> <batch> <log> <gpu>  -> 0 ok, 1 problems
    local name="$1" run_dir="$2" config="$3" batch="$4" log="$5" gpu="$6"
    mkdir -p "$RUNS_DIR"
    if ! "$PYTHON" -m fpyolo.runstate verify "$run_dir" --config "$config" --log "$log" ${batch:+--batch "$batch"} 2>&1 | tee -a "$log"; then
        echo "[GPU $gpu] $(stamp) $name: CHECK FAILED (see PROBLEM lines above)" >&2
        return 1
    fi
}

run_job() {  # run_job "<job row>" <gpu>  -> 0 ok/skipped, 1 failed
    local row="$1" gpu="$2" name config data batch run_dir log state rc=0
    IFS='|' read -r name config data batch <<< "$row"
    run_dir="$RUNS_DIR/$name"
    log="$RUNS_DIR/$name.log"
    # the state is the last stdout line (imports may print other lines before it)
    state="$("$PYTHON" -m fpyolo.runstate state "$run_dir" | tail -n 1)"

    local -a cmd=()
    local action
    case "$state" in
        done) action=skip ;;
        running) action=skip-running ;;
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
        [ "$action" != start ] || [ ! -f "$log" ] || echo "    (existing log is rotated to $log.<UTC stamp>.old)"
        [ "$action" != skip ] || echo "    then: $(quote "$PYTHON" -m fpyolo.runstate verify "$run_dir" --config "$config" --log "$log" ${batch:+--batch "$batch"})"
        [ "$action" != FAIL ] || return 1
        return 0
    fi

    case "$action" in
        skip)
            echo "[GPU $gpu] $(stamp) $name: done already, skipping (verifying)"
            verify_job "$name" "$run_dir" "$config" "$batch" "$log" "$gpu"
            return ;;
        skip-running)
            echo "[GPU $gpu] $(stamp) $name: WARNING: another process is training this run (its lock is held), skipping" >&2
            echo "$name" >>"$CLAIMS/running"
            return 0 ;;
        FAIL)
            echo "[GPU $gpu] $(stamp) $name: FAILED, $run_dir exists but has nothing to resume from;" \
                 "move or delete it (see README, Resume) and rerun" >&2
            return 1 ;;
    esac

    mkdir -p "$RUNS_DIR"
    echo "[GPU $gpu] $(stamp) $name: $action, log $log"
    # rotate the log of a fresh start (e.g. after a discarded run); resumes keep appending
    if [ "$action" = start ] && [ -f "$log" ]; then
        mv "$log" "$log.$(date -u +%Y%m%dT%H%M%SZ).old"
    fi
    # the attempt header: fpyolo.runstate.ATTEMPT_MARKER, verify only reads the log after the last one
    { echo; echo "=== fpyolo-attempt $(stamp) GPU $gpu: $action $(quote "${cmd[@]}")"; } >>"$log"
    run_logged "$log" "${cmd[@]}" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[GPU $gpu] $(stamp) $name: FAILED (exit $rc), last log line: $(tail -n 1 "$log")" \
             "| see README (Resume). If it died from an OOM caused by another user, rerun this script after the sweep has exited" \
             "and memory is free (it resumes)" >&2
        return 1
    fi
    echo "[GPU $gpu] $(stamp) $name: finished, checking"
    verify_job "$name" "$run_dir" "$config" "$batch" "$log" "$gpu"
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

# ---- real run: one sweep per runs dir, then one worker per GPU pulling from a shared queue ------
command -v flock >/dev/null 2>&1 || die "flock (util-linux) is required"
mkdir -p "$RUNS_DIR"
exec 9>"$RUNS_DIR/.sweep.lock"
flock -n 9 || die "another sweep is running on $RUNS_DIR (holds $RUNS_DIR/.sweep.lock); wait for it to exit"
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

# exit 0 = all 9 runs complete and verified; 1 = some run failed; 3 = none failed but some run was skipped
# because another process is training it (so `sweep && evaluate` never continues with runs missing)
rc=0
if [ -s "$CLAIMS/failed" ]; then
    echo "run_all_seeds: $(stamp) FAILED runs: $(tr '\n' ' ' < "$CLAIMS/failed")" >&2
    rc=1
fi
if [ -s "$CLAIMS/running" ]; then
    echo "run_all_seeds: $(stamp) WARNING: skipped runs that another process is training: $(tr '\n' ' ' < "$CLAIMS/running");" \
         "rerun this script when they have finished" >&2
    [ "$rc" -ne 0 ] || rc=3
fi
if [ "$rc" -eq 0 ]; then
    echo "run_all_seeds: $(stamp) all runs finished; next: evaluation (README)"
fi
exit "$rc"
