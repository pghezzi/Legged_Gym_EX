#!/usr/bin/env bash
# Only the new, exclusively created run directory is writable from the container.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash run_paper_experiments_docker.sh MODE [options]
Modes: offline | locomotion | all | plot-only

  --image IMAGE             Default: leggedgym-ex:isaacgym (PAPER_IMAGE)
  --gpu INDEX               Host GPU index; default 0 (TRAIN_GPU).
                            'none' allowed for plot-only / smoke tests.
  --classifier-data DIR     Structural train.pt/val.pt/test.pt/calibration.pt
  --ordered-data DIR        Compiled ordered test.pt
  --paper-offline-dir DIR   Existing frozen offline results (locomotion only)
  --jit FILE                Specialist JIT/LoRA bundle
  --distilled-jit FILE      Unified distilled JIT policy
  --bundle FILE             figure_data.pt for plot-only
  --output-root DIR         Default: <repository>/paper_runs (PAPER_OUTPUT_ROOT)
  --run-id NAME             New directory name; existing names are rejected.
                            Default: timestamp plus a unique random suffix.
  --offline-arg TOKEN       Repeat once per additional offline argument/value.
  --locomotion-arg TOKEN    Repeat once per additional locomotion argument/value.
                            Path/mode overrides are reserved by this wrapper.
  --dry-run                Validate inputs and print commands; create no outputs.
  --smoke-test success|failure
                            Exercise Docker writes/cleanup only; no experiments,
                            Python, simulator, or dependency installation.
  -h, --help

All mode runs offline first, then locomotion with those new artifacts.
Inputs and source subdirectories are read-only; the image's .venv is preserved.
Outputs: <root>/<run-id>/{offline,locomotion,logs}, commands/mounts/status files.
Directories become 0777 and regular files 0666 (including on ordinary failure).
Everyone can modify these outputs: use a trusted host and traversable parent
directories. SIGKILL/host loss cannot guarantee cleanup; failures are reported.
HELP
}
die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${1:---help}
case "$mode" in -h|--help) usage; exit 0;; offline|locomotion|all|plot-only) shift;; *) usage >&2; exit 2;; esac
image=${PAPER_IMAGE:-leggedgym-ex:isaacgym}
gpu=${TRAIN_GPU:-0}
output_root=${PAPER_OUTPUT_ROOT:-$repo/paper_runs}
run_id= classifier_data= ordered_data= offline_dir= jit= distilled= bundle= smoke=
dry_run=false
offline_extra=() locomotion_extra=()
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0;;
        --dry-run) dry_run=true; shift; continue;;
        --image|--gpu|--classifier-data|--ordered-data|--paper-offline-dir|--jit|--distilled-jit|--bundle|--output-root|--run-id|--offline-arg|--locomotion-arg|--smoke-test)
            (($# >= 2)) && [[ -n $2 ]] || die "Missing value for $1";;
        *) die "Unknown option: $1";;
    esac
    case "$1" in
        --image) image=$2;; --gpu) gpu=$2;; --classifier-data) classifier_data=$2;;
        --ordered-data) ordered_data=$2;; --paper-offline-dir) offline_dir=$2;;
        --jit) jit=$2;; --distilled-jit) distilled=$2;; --bundle) bundle=$2;;
        --output-root) output_root=$2;; --run-id) run_id=$2;;
        --offline-arg) offline_extra+=("$2");; --locomotion-arg) locomotion_extra+=("$2");;
        --smoke-test) smoke=$2;;
    esac
    shift 2
done
[[ -z $smoke || $smoke == success || $smoke == failure ]] || die 'Invalid smoke-test outcome'
[[ $gpu =~ ^[0-9]+$ || ($gpu == none && ($mode == plot-only || -n $smoke)) ]] || die 'Invalid host GPU index'
[[ -z $run_id || $run_id =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die 'run-id must be a single safe directory name'

# Reject argparse abbreviations too: passthrough must not redirect managed paths.
check_extra() {
    local reserved=$1 token option name
    shift
    for token in "$@"; do
        [[ $token == --* ]] || continue
        option=${token%%=*}
        for name in $reserved; do
            [[ $name != "$option"* ]] || die "Use wrapper options instead of passthrough $option"
        done
    done
}
check_extra '--dataset --classifier-data --ordered-data --bayesian-data --output --plot-only --bundle-from-existing' "${offline_extra[@]}"
check_extra '--paper_offline_dir --jit --distilled_jit --output' "${locomotion_extra[@]}"
file_required() { [[ -f $1 && -r $1 ]] || die "Missing/unreadable artifact: $1"; }
dir_required() { [[ -n $1 && -d $1 && -r $1 && -x $1 ]] || die "Missing/unreadable directory: $1"; }
if [[ $mode == offline || $mode == all ]]; then
    dir_required "$classifier_data"; dir_required "$ordered_data"
    for split in train val test calibration; do file_required "$classifier_data/$split.pt"; done
    file_required "$ordered_data/test.pt"
fi
if [[ $mode == locomotion || $mode == all ]]; then
    file_required "$jit"; file_required "$distilled"
fi
if [[ $mode == locomotion ]]; then
    dir_required "$offline_dir"
    file_required "$offline_dir/manifest.json"
    file_required "$offline_dir/experiment_1_instantaneous_per_seed.json"
    for architecture in feature_nn raw_depth_nn; do
        for seed in 0 1 2; do
            for artifact in classifier.pt nn_model_args.pt; do
                file_required "$offline_dir/artifacts/$architecture/seed_$seed/$artifact"
            done
        done
    done
    for artifact in extractor.pt standardizer.pt; do file_required "$offline_dir/artifacts/feature_nn/$artifact"; done
elif [[ -n $offline_dir ]]; then
    die '--paper-offline-dir applies only to locomotion; all uses its new offline results'
fi
[[ $mode != plot-only ]] || file_required "$bundle"

output_root=$(realpath -m -- "$output_root")
run_id=${run_id:-$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM}
run_dir=$output_root/$run_id
[[ ! -e $run_dir && ! -L $run_dir ]] || die "Run already exists (will not overwrite): $run_dir"
workspace=/workspace/LeggedGym-Ex
mounts=() mappings=()
mount_path() {
    local host=$1 target=$2 access=$3
    # Docker --mount uses commas as separators; fail rather than misparse a path.
    [[ $host != *','* && $host != *$'\n'* && $target != *','* ]] || die 'Mount paths cannot contain commas/newlines'
    mounts+=(--mount "type=bind,source=$host,target=$target${access:+,$access}")
    mappings+=("$host -> $target (${access:-readwrite})")
}
mount_path "$repo/legged_gym" "$workspace/legged_gym" readonly
mount_path "$repo/rsl_rl" "$workspace/rsl_rl" readonly
mount_path "$run_dir" /paper ''
mount_path "$run_dir/logs" "$workspace/logs" ''
if [[ $mode == offline || $mode == all ]]; then
    mount_path "$(realpath -- "$classifier_data")" /inputs/classifier readonly
    mount_path "$(realpath -- "$ordered_data")" /inputs/ordered readonly
fi
if [[ $mode == locomotion || $mode == all ]]; then
    mount_path "$(realpath -- "$jit")" /inputs/specialists.pt readonly
    mount_path "$(realpath -- "$distilled")" /inputs/distilled.pt readonly
fi
if [[ $mode == locomotion ]]; then mount_path "$(realpath -- "$offline_dir")" /inputs/offline readonly; fi
if [[ $mode == plot-only ]]; then mount_path "$(realpath -- "$bundle")" /inputs/figure_data.pt readonly; fi
offline_cmd=(python -m legged_gym.scripts.depth_data_pipeline.evaluate_paper_offline_experiments_1_2)
if [[ $mode == plot-only ]]; then
    offline_cmd+=(--plot-only /inputs/figure_data.pt --output /paper/offline)
else
    offline_cmd+=(--classifier-data /inputs/classifier --ordered-data /inputs/ordered --output /paper/offline)
fi
offline_cmd+=("${offline_extra[@]}")
locomotion_cmd=(python -m legged_gym.scripts.evaluation.run_paper_locomotion_evaluation
    --paper_offline_dir "$([[ $mode == all ]] && printf /paper/offline || printf /inputs/offline)"
    --jit /inputs/specialists.pt --distilled_jit /inputs/distilled.pt --output /paper/locomotion --headless
    "${locomotion_extra[@]}")
docker_base=(docker run --rm --init --ipc=host --network=host
    -e NVIDIA_DRIVER_CAPABILITIES=all -e SIMULATOR=isaacgym -e PYTHONUNBUFFERED=1
    -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/paper/logs/matplotlib
    -e NUMBA_CACHE_DIR=/paper/logs/numba -w "$workspace")
[[ $gpu == none ]] || docker_base+=(--gpus "device=$gpu")
docker_base+=("${mounts[@]}" --entrypoint bash "$image")
# This exact wrapper also runs in smoke tests. Cleanup never follows symlinks.
container_script='
set -euo pipefail
umask 000
finish() {
    status=$?
    trap - EXIT
    printf "%s\n" "$status" > "/paper/logs/$stage.container.exit_status"
    find -P /paper -xdev -type d -exec chmod 0777 {} + || { [[ $status != 0 ]] || status=125; }
    find -P /paper -xdev -type f -exec chmod 0666 {} + || { [[ $status != 0 ]] || status=125; }
    exit "$status"
}
stage=$1; smoke=$2; shift 2
trap finish EXIT
trap "exit 130" INT
trap "exit 143" TERM
export PATH="/workspace/LeggedGym-Ex/.venv/bin:$PATH"
if [[ -n $smoke ]]; then
    mkdir -p /paper/offline/smoke /paper/locomotion/smoke
    printf "host-visible artifact\n" > /paper/offline/smoke/artifact.txt
    printf "replay\n" > /paper/locomotion/smoke/replay.txt
    chmod 0700 /paper/offline/smoke
    chmod 0600 /paper/offline/smoke/artifact.txt
    printf "smoke stdout\n"; printf "smoke stderr\n" >&2
    [[ $smoke != failure ]] || exit 23
else
    # Match omy_run_isaacgym.sh: install into the image venv, never the host.
    uv pip install --python /workspace/LeggedGym-Ex/.venv/bin/python python-dotenv
    "$@"
fi
'
print_command() { printf '%q ' "$@"; printf '\n'; }
if $dry_run; then
    printf 'New host run: %s\n' "$run_dir"
    printf 'Mount: %s\n' "${mappings[@]}"
    if [[ $mode != locomotion ]]; then print_command "${docker_base[@]}" -c "$container_script" paper offline "$smoke" "${offline_cmd[@]}"; fi
    if [[ $mode == locomotion || $mode == all ]]; then print_command "${docker_base[@]}" -c "$container_script" paper locomotion "$smoke" "${locomotion_cmd[@]}"; fi
    exit 0
fi
command -v docker >/dev/null || die 'docker is not installed'
umask 000
mkdir -p -- "$output_root"
mkdir -- "$run_dir" # Atomic exclusive creation; never reuse or chmod old results.
mkdir -- "$run_dir/offline" "$run_dir/locomotion" "$run_dir/logs"
finish_host() {
    local status=$? permissions=0
    trap - EXIT
    printf '%s\n' "$status" > "$run_dir/exit_status"
    # Host handles console logs; a no-GPU container can repair root-owned files
    # after a killed runner. Only this new run is mounted, never sources/inputs.
    find -P "$run_dir" -xdev -type d -exec chmod 0777 {} + 2>/dev/null || permissions=1
    find -P "$run_dir" -xdev -type f -exec chmod 0666 {} + 2>/dev/null || permissions=1
    if ((permissions)); then
        if ! docker run --rm --network none --mount "type=bind,source=$run_dir,target=/paper" \
            --entrypoint bash "$image" -c 'find -P /paper -xdev -type d -exec chmod 0777 {} + && find -P /paper -xdev -type f -exec chmod 0666 {} +' \
            >> "$run_dir/logs/permissions.log" 2>&1; then
            printf 'WARNING: permission cleanup failed; see %s/logs/permissions.log\n' "$run_dir" >&2
            [[ $status != 0 ]] || status=125
        fi
    fi
    printf '%s\n' "$status" > "$run_dir/exit_status"
    printf 'Run outputs: %s (exit %s)\n' "$run_dir" "$status"
    exit "$status"
}
trap finish_host EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s\n' "mode=$mode" "image=$image" "host_gpu=$gpu" "run_id=$run_id" "smoke_test=$smoke" \
    "commit=$(git -C "$repo" rev-parse HEAD)" "started_utc=$(date -u +%FT%TZ)" > "$run_dir/run_metadata.txt"
git -C "$repo" status --short > "$run_dir/git_status.txt"
printf '%s\n' "${mappings[@]}" > "$run_dir/mounts.txt"
run_stage() {
    local stage=$1
    shift
    local command=("${docker_base[@]}" -c "$container_script" paper "$stage" "$smoke" "$@")
    print_command "${command[@]}" >> "$run_dir/commands.sh"
    set +e
    "${command[@]}" 2>&1 | tee "$run_dir/logs/$stage.log"
    local codes=("${PIPESTATUS[@]}")
    set -e
    printf '%s\n' "${codes[0]}" > "$run_dir/logs/$stage.exit_status"
    ((codes[0] == 0)) || return "${codes[0]}"
    return "${codes[1]}"
}
if [[ $mode != locomotion ]]; then run_stage offline "${offline_cmd[@]}"; fi
if [[ $mode == locomotion || $mode == all ]]; then run_stage locomotion "${locomotion_cmd[@]}"; fi
