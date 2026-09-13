#!/bin/bash
# Run the complete MM-SLUICE-01 bundle/consumer-boundary campaign from any
# working directory. Every run is fresh and retained, including failures.

set -u

script_dir="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
repo_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
output_root="${LMSLUICE_MM_SLUICE_OUT:-${TMPDIR:-/tmp}}"

if ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "FAIL: python3 is required; no run directory was created" >&2
    exit 2
fi

if [ ! -d "$output_root" ] && ! mkdir -p "$output_root"; then
    printf '%s\n' "FAIL: cannot create output root: $output_root" >&2
    exit 2
fi

run_dir="$(mktemp -d "$output_root/lmsluice-mm-sluice-01.XXXXXX")"
if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    printf '%s\n' "FAIL: cannot create a fresh MM-SLUICE-01 run directory" >&2
    exit 2
fi

log_path="$run_dir/probe.log"
index_path="$run_dir/result-index.txt"
archive_path="$run_dir/evidence.zip"
preflight_status=0

for required_path in \
    "$repo_root/lmsluice/__init__.py" \
    "$repo_root/lmsluice/bundle.py" \
    "$repo_root/lmsluice/onnxruntime_adapter.py" \
    "$repo_root/lmsluice/observability.py" \
    "$repo_root/experiments/mm_sluice/fixtures.py" \
    "$repo_root/experiments/mm_sluice/harness.py" \
    "$repo_root/experiments/device_readiness/archive.py" \
    "$repo_root/tests/test_bundle.py" \
    "$repo_root/docs/evaluations/workload.json"; do
    if [ ! -f "$required_path" ]; then
        printf 'missing tracked input: %s\n' "$required_path" >> "$log_path"
        preflight_status=2
    fi
done

if [ "$preflight_status" -eq 0 ]; then
    (
        cd "$repo_root" || exit 2
        PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
            python3 -c 'import experiments.mm_sluice.fixtures, experiments.mm_sluice.harness, lmsluice.bundle, lmsluice.onnxruntime_adapter' \
            >> "$log_path" 2>&1
    ) || preflight_status=$?
fi

campaign_status=0
if [ "$preflight_status" -eq 0 ]; then
    (
        cd "$repo_root" || exit 2
        PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
            python3 -m experiments.mm_sluice.harness \
            --out "$run_dir" --repo-root "$repo_root"
    ) >"$log_path" 2>&1
    campaign_status=$?
else
    campaign_status="$preflight_status"
fi

# Finalize independently of campaign status. The archive reads all evidence
# reached so far, including failed-stage records and error details.
archive_status=0
if [ -f "$repo_root/experiments/device_readiness/archive.py" ]; then
    (
        cd "$repo_root" || exit 2
        PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
            python3 -m experiments.device_readiness.archive \
            --run-dir "$run_dir" --archive "$archive_path"
    ) >>"$log_path" 2>&1
    archive_status=$?
else
    printf '%s\n' "archive source unavailable" >> "$log_path"
    archive_status=2
fi

archive_verify_status=0
if [ "$archive_status" -eq 0 ] && [ -f "$archive_path" ]; then
    (
        cd "$repo_root" || exit 2
        PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
            python3 -m experiments.device_readiness.archive \
            --run-dir "$run_dir" --archive "$archive_path" --verify
    ) >>"$log_path" 2>&1
    archive_verify_status=$?
else
    archive_verify_status=2
fi

archive_size="unavailable"
archive_hash="unavailable"
archive_facts_status=0
if [ -f "$archive_path" ]; then
    archive_facts="$(python3 -c 'import hashlib,os,sys; h=hashlib.sha256(); f=open(sys.argv[1], "rb"); h.update(f.read()); f.close(); print(str(os.path.getsize(sys.argv[1]))+" "+h.hexdigest())' "$archive_path" 2>>"$log_path")"
    archive_facts_status=$?
    if [ "$archive_facts_status" -eq 0 ]; then
        archive_size="${archive_facts%% *}"
        archive_hash="${archive_facts#* }"
    fi
else
    archive_facts_status=2
fi

summary_status="unavailable"
summary_hash="unavailable"
if [ -f "$run_dir/summary.json" ]; then
    summary_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("engineering_status", "unavailable"))' "$run_dir/summary.json" 2>>"$log_path")"
    summary_status_exit=$?
    if [ "$summary_status_exit" -ne 0 ]; then
        summary_status="unavailable"
    fi
    summary_hash="$(python3 -c 'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1], "rb"); h.update(f.read()); f.close(); print(h.hexdigest())' "$run_dir/summary.json" 2>>"$log_path")"
    summary_hash_exit=$?
    if [ "$summary_hash_exit" -ne 0 ]; then
        summary_hash="unavailable"
    fi
fi

final_status=0
if [ "$preflight_status" -ne 0 ]; then
    final_status="$preflight_status"
elif [ "$campaign_status" -ne 0 ]; then
    final_status="$campaign_status"
elif [ "$archive_status" -ne 0 ]; then
    final_status="$archive_status"
elif [ "$archive_verify_status" -ne 0 ]; then
    final_status="$archive_verify_status"
elif [ "$archive_facts_status" -ne 0 ]; then
    final_status="$archive_facts_status"
fi

{
    printf 'repository=%s\n' "$repo_root"
    printf 'run_directory=%s\n' "$run_dir"
    printf 'preflight_exit=%s\n' "$preflight_status"
    printf 'campaign_exit=%s\n' "$campaign_status"
    printf 'archive_exit=%s\n' "$archive_status"
    printf 'archive_verify_exit=%s\n' "$archive_verify_status"
    printf 'archive_facts_exit=%s\n' "$archive_facts_status"
    printf 'final_exit=%s\n' "$final_status"
    printf 'engineering_status=%s\n' "$summary_status"
    printf 'summary_sha256=%s\n' "$summary_hash"
    printf 'summary=%s\n' "$run_dir/summary.json"
    printf 'results=%s\n' "$run_dir/results.jsonl"
    printf 'log=%s\n' "$log_path"
    printf 'records=%s\n' "$run_dir/records"
    printf 'errors=%s\n' "$run_dir/errors"
    printf 'evidence_archive=%s\n' "$archive_path"
    printf 'evidence_archive_bytes=%s\n' "$archive_size"
    printf 'evidence_archive_sha256=%s\n' "$archive_hash"
} > "$index_path"

if [ "$final_status" -eq 0 ]; then
    printf '%s\n' "mm-sluice-01: PASS"
else
    printf '%s\n' "mm-sluice-01: FAIL"
fi
printf 'run directory: %s\n' "$run_dir"
printf 'result index: %s\n' "$index_path"
printf 'summary: %s\n' "$run_dir/summary.json"
printf 'evidence archive: %s\n' "$archive_path"
printf 'evidence archive bytes: %s\n' "$archive_size"
printf 'evidence archive sha256: %s\n' "$archive_hash"
printf 'campaign exit: %s\n' "$campaign_status"
printf 'final exit: %s\n' "$final_status"
exit "$final_status"
