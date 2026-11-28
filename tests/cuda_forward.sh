#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
#
# cuda_forward.sh — a whole forward pass on the CUDA backend against the CPU.
#
#   make WASTE_ENABLE_CUDA=1 cuda-check        (runs test_cuda, then this)
#   tests/cuda_forward.sh [model.waste]        (default: a synthetic container)
#
# Why it exists: `make WASTE_ENABLE_CUDA=1 check` passed 82/0/27 on the day
# the backend landed, and none of it ran on the GPU. Every matvec in the
# synthetic container is smaller than WASTE_CUDA_MIN_KB (2 MB), so the CPU
# took them all and the "CUDA" logits were byte-identical to the CPU's. This
# forces the floor to 0, requires the device counters to show work, and only
# then compares the two paths.
#
# The comparison follows tests/run.sh: logits within LOGIT_EPS pass; past it,
# tests/route_diff.py judges the first routing decision that differed, because
# on a MoE a flipped tie moves the logits by far more than the arithmetic did.
# Exit status: 0 pass, 1 fail, 77 skip (no CUDA build or no usable device).
set -u
cd "$(dirname "$0")/.." || exit 1

LOGIT_EPS="${LOGIT_EPS:-1e-4}"
IDS="${CUDA_FWD_IDS:-3,7,11,5,2,9,4,8}"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

[ -x ./test_forward ] || { echo "SKIP: ./test_forward not built (make test)"; exit 77; }
./waste version 2>/dev/null | grep -q "backend CUDA" || {
    echo "SKIP: this build has no usable CUDA device (make WASTE_ENABLE_CUDA=1, and a GPU)"
    exit 77
}

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
    MODEL="$TMP/tiny.waste"
    python3 tools/make_test_container.py "$MODEL" >/dev/null 2>&1 || {
        echo "SKIP: could not build the synthetic container"; exit 77; }
fi

WASTE_CUDA=0 WASTE_DUMP_ROUTE="$TMP/cpu.route" WASTE_DUMP_SCORES="$TMP/cpu.scores" \
    ./test_forward "$MODEL" "$IDS" "$TMP/cpu.bin" 4 >/dev/null 2>"$TMP/cpu.err" || {
    echo "FAIL: CPU forward pass did not run"; head -3 "$TMP/cpu.err"; exit 1; }

WASTE_CUDA_MIN_KB=0 WASTE_CUDA_STATS=1 WASTE_DUMP_ROUTE="$TMP/gpu.route" \
    ./test_forward "$MODEL" "$IDS" "$TMP/gpu.bin" 4 >/dev/null 2>"$TMP/gpu.err" || {
    echo "FAIL: CUDA forward pass did not run"; head -3 "$TMP/gpu.err"; exit 1; }

stats=$(grep "CUDA stats" "$TMP/gpu.err")
calls=$(printf '%s\n' "$stats" | sed -n 's/.*stats: \([0-9]*\) device calls.*/\1/p')
if [ -z "$calls" ] || [ "$calls" -eq 0 ]; then
    echo "FAIL: the forward pass never reached the device (${stats:-no stats line})"
    exit 1
fi
echo "      ${stats#waste: }"

python3 - "$TMP/cpu.bin" "$TMP/gpu.bin" "$LOGIT_EPS" <<'PY'
import struct, sys
def L(p):
    b = open(p, "rb").read()
    return struct.unpack(f"<{len(b)//4}f", b)
a, b = L(sys.argv[1]), L(sys.argv[2])
if len(a) != len(b) or not a:
    print(f"      logit dumps differ in length: {len(a)} vs {len(b)}")
    sys.exit(1)
d = max(abs(x - y) for x, y in zip(a, b))
print(f"      max |cpu - cuda| {d:.3g} over {len(a)} logits (eps {sys.argv[3]})")
sys.exit(0 if d < float(sys.argv[3]) and a.index(max(a)) == b.index(max(b)) else 2)
PY
case $? in
    0) echo "PASS: CUDA forward pass matches the CPU"; exit 0 ;;
    2) why=$(python3 tests/route_diff.py --ref "$TMP/cpu.route" --other "$TMP/gpu.route" \
                     --scores "$TMP/cpu.scores" 2>&1)
       rc=$?
       printf "      %s\n" "$why"
       if [ $rc -eq 2 ]; then
           echo "PASS: CUDA forward pass differs only past a routing tie"; exit 0
       fi
       echo "FAIL: CUDA forward pass diverges from the CPU"; exit 1 ;;
    *) echo "FAIL: could not compare the logit dumps"; exit 1 ;;
esac
