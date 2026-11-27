#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
# fetch_weights.sh — long-haul download of a very large model, built to survive.
#
# Storage split (docs/GATES.md, Gate H): raw shards land on the external
# staging disk; the converted WASTE container goes to internal NVMe. That is
# why --dest defaults to a volume rather than to the repo.
#
# A 1.4 TB pull over hours will hit dropped connections, 5xx from the CDN,
# and at least one interrupted run. So:
#   - every shard is resumed with `curl -C -`, never restarted;
#   - each shard is retried with exponential backoff and jitter, and so is
#     every small file the repo's own listing names — a file that exists and
#     was not fetched is a failure, not a 404;
#   - a shard counts as done only when its size matches Content-Length, and
#     one that is *longer* than Content-Length is deleted rather than
#     resumed — `-C -` would ask for a range past the end and burn every
#     retry on it;
#     and completed shards are recorded in a state file so re-runs skip
#     them without even a HEAD request;
#   - free space is checked before starting and again before every shard,
#     so the run stops cleanly instead of filling the disk;
#   - progress, failures and the resume point are logged with timestamps.
#
#   tools/fetch_weights.sh --dry-run          # preflight only, no download
#   tools/fetch_weights.sh                    # start or resume
#   tools/fetch_weights.sh --check            # verify what is on disk, no fetch
#   tools/fetch_weights.sh --verify           # --check, then sha256 of every shard
#                                             # against the Hub's LFS hash
#   tools/fetch_weights.sh --repo moonshotai/Kimi-Linear --dest /data/kl
#   tools/fetch_weights.sh --repo Qwen/Qwen3.8-Flash-Next \
#       --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
#       --dest /Users/admin/mnt/llm/qwen38-flash-next/raw --dry-run
#
# --revision pins every fetch URL (API listing, small files, shards).
# Omit it and the script uses main, which moves. Qwen3.8-Flash-Next
# must pass the SHA recorded in docs/QWEN.md.
#
# Set HF_TOKEN for a gated repo. Safe to run repeatedly and safe to kill:
# the next run picks up where it stopped, mid-shard.

set -uo pipefail

# python3, resolved by RUNNING a candidate rather than by looking one up.
# On Windows the name on PATH is usually the Microsoft Store App Execution
# Alias: a zero-byte reparse point that exists, exits 49, and prints an
# advert for the Store instead of running anything. Same finding as
# tests/run.sh, which shims PATH because it has 49 call sites and
# subprocesses to cover; the handful here get a name and pass it down.
#
# Windows installs the versionless `python`, and `py` besides, so a working
# interpreter is normally there under another name. If none answers, this
# stays `python3` and the script fails loudly at first use, which is right
# here: unlike the suite there is nothing to skip, and a run that cannot
# read its own index must stop rather than carry on.
if [ -z "${PY:-}" ]; then
    for _cand in python3 python py; do
        "$_cand" -c '' >/dev/null 2>&1 && { PY="$_cand"; break; }
    done
    : "${PY:=python3}"
fi
export PY

REPO="${REPO:-moonshotai/Kimi-K3}"
REVISION="${REVISION:-main}"
DEST="${DEST:-/Volumes/WasteDisk/k3}"
JOBS="${JOBS:-3}"
MAX_RETRY="${MAX_RETRY:-8}"
SMALL_RETRY="${SMALL_RETRY:-5}"
SMALL_BACKOFF="${SMALL_BACKOFF:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"
DRY=0
CHECK_ONLY=0
VERIFY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --revision) REVISION="$2"; shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        --verify) CHECK_ONLY=1; VERIFY=1; shift ;;
        -h|--help) sed -n '4,36p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

API="https://huggingface.co/api/models/${REPO}/revision/${REVISION}"
RAW="https://huggingface.co/${REPO}/resolve/${REVISION}"
STATE="$DEST/.download-state"
LOG="$DEST/download.log"

# GNU and BSD disagree on how to ask a file its size, and this script has to
# run on the laptop that converts and on a Linux box that only fetches.
if stat --version >/dev/null 2>&1; then
    fsize() { stat -c%s "$1" 2>/dev/null || echo 0; }
    STAT_MODE=gnu
else
    fsize() { stat -f%z "$1" 2>/dev/null || echo 0; }
    STAT_MODE=bsd
fi

# -P forces POSIX single-line output: without it a long device name wraps
# and `NR==2` picks up half a row.
free_kb() { df -kP "$DEST" | awk 'NR==2 {print $4}'; }
free_gb() { echo $(( $(free_kb) / 1048576 )); }

# curl, with the auth header when HF_TOKEN is set. A function rather than an
# array of extra arguments because "${arr[@]}" on an *empty* array trips
# `set -u` in bash 3.2 — which is exactly what macOS ships, so the array
# version failed on the first machine it ran on. The token never reaches the
# log or the argument list of anything but curl itself.
hcurl() {
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -H "Authorization: Bearer $HF_TOKEN" "$@"
    else
        curl "$@"
    fi
}

echo "repo:  $REPO"
echo "rev:   $REVISION"
echo "dest:  $DEST"

code=$(hcurl -s -o /dev/null -w '%{http_code}' --max-time 30 "$API")
if [ "$code" != "200" ]; then
    echo "!! repo not reachable (HTTP $code)."
    echo "   HuggingFace answers 401 both for a missing repo and for a gated"
    echo "   one, so this does not tell them apart. If the release just"
    echo "   happened, try the -Base / -Instruct variants; if the repo is"
    echo "   gated, set HF_TOKEN."
    exit 1
fi

mkdir -p "$DEST" || exit 1
touch "$STATE"

# The message goes to the terminal first and to the log file second. It was
# one `tee` and the two shared a fate: $LOG lives on $DEST, and when a USB
# enclosure drops off the bus mid-run every log() prints "tee: ... No such
# file or directory" and swallows what it was told to say. The run that
# found this refused correctly and said nothing about why -- so the failure
# looked like the silence it was there to break.
log() {
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
    # 2>/dev/null BEFORE the append, not after. Redirections are applied
    # left to right, so with the append first the shell reports "No such
    # file or directory" on a stderr that has not been silenced yet -- which
    # is the same message, from a different mouth.
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" 2>/dev/null >> "$LOG"
    return 0
}

# --- small files ----------------------------------------------------------
# The index is needed by every mode, because it is what says which shards
# exist and how big they are. Everything else is only needed by a real
# download and by convert.py afterwards, so --check and --dry-run stay
# inspection modes: they fetch nothing but the index, and nothing at all
# once it is on disk.
# get_small MODE FILE...  — MODE is `req` or `opt`, and the difference is
# whether "a 404 here is normal" is true.
#
# For `opt`, the guessed fallback list further down, it is: a whitelist names
# files a given repo may not ship. For `req` — the repo's own API listing, and
# the index every mode needs — it is not. Those files demonstrably exist, so
# anything short of fetching one is a file that was there and was not taken.
#
# This used to be one attempt with the error discarded into `rm -f`, which
# cost a real download (#35): ten of eleven small files lost to transient
# failures, on a run that then reported ALL SHARDS COMPLETE and rc=0. All ten
# returned 200 when probed by hand minutes later. The gap between cause and
# symptom is the expensive part — a missing config.json fails convert.py
# hours later, and a missing tiktoken.model does not fail it at all: the
# conversion completes and produces a container that cannot tokenize.
#
# The listing block below already makes this argument about whitelists —
# "a whitelist cannot report what it never knew to ask for" — and then the
# fetch dropped the answer without saying so.
SMALL_MISSING=""
get_small() {
    mode="$1"; shift
    for f in "$@"; do
        [ -s "$DEST/$f" ] && continue
        ok=0
        for try in $(seq 1 "$SMALL_RETRY"); do
            # No -f, deliberately: with it, every HTTP status collapses into
            # exit 22 and a 404 is indistinguishable from a 503. Letting curl
            # succeed on an HTTP error keeps rc for "did not reach the server"
            # and %{http_code} for "the server answered, and this is what it
            # said" — which is the whole distinction this function needs.
            code=$(hcurl -sL --max-time 300 -w '%{http_code}' \
                         -o "$DEST/$f.part" "$RAW/$f" 2>/dev/null)
            rc=$?
            if [ "$rc" = 0 ] && [ "$code" = 200 ]; then
                mv "$DEST/$f.part" "$DEST/$f"
                log "got $f"
                ok=1
                break
            fi
            rm -f "$DEST/$f.part"
            if [ "$rc" = 0 ] && [ "$code" = 404 ]; then
                # Retrying cannot help either way. Silent for `opt`, which is
                # the case the old comment was written about; for `req` the
                # repo changed under the run, and that is worth a line.
                [ "$mode" = req ] &&
                    log "MISSING $f: the repo listed it and now answers 404"
                break
            fi
            [ "$try" = "$SMALL_RETRY" ] && break
            # Short backoff: these are kilobytes to a few MB, not the shard
            # path's tens of GB, so the worker's 1<<try schedule would spend
            # more time waiting than the file takes to fetch. Overridable
            # for the same reason MAX_RETRY is — tests/run.sh drives this
            # against a local server and must not sleep through it.
            wait=$(( try * SMALL_BACKOFF + RANDOM % (SMALL_BACKOFF + 2) ))
            log "fail $f (rc=$rc http=$code), retry in ${wait}s"
            sleep "$wait"
        done
        if [ "$ok" = 0 ] && [ "$mode" = req ]; then
            SMALL_MISSING="${SMALL_MISSING:+$SMALL_MISSING }$f"
        fi
    done
}

# `req`: a transient failure here used to reach the FATAL below, which then
# blamed the repo layout for a network error.
get_small req model.safetensors.index.json
if [ "$DRY" = 0 ] && [ "$CHECK_ONLY" = 0 ]; then
    # Ask the repo what it contains instead of guessing filenames. The
    # hardcoded list this replaces cost real money: it did not know about
    # encoding_k3.py, so the chat template looked absent and had to be
    # reconstructed from a figure in the technical report; and it silently
    # missed preprocessor_config.json, so the image normalization was the
    # CLIP convention for a day when the release states mean = std = 0.5.
    # A whitelist cannot report what it never knew to ask for.
    SMALL=$(hcurl -sfL --max-time 60 "$API" 2>/dev/null | "$PY" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
for s in d.get("siblings", []):
    f = s.get("rfilename", "")
    # shards come down the resumable path; skip directories and the index
    if f.endswith(".safetensors") or "/" in f or f == "model.safetensors.index.json":
        continue
    print(f)
' 2>/dev/null | tr -d '\r')       # see the .shards comment below: CRLF here
                                  # would put a %0D on every URL get_small
                                  # asks for, and now record every one of
                                  # them as missing
    if [ -n "$SMALL" ]; then
        log "repo lists $(printf '%s\n' "$SMALL" | wc -l | tr -d ' ') small files"
        # shellcheck disable=SC2086
        get_small req $SMALL
    else
        log "WARNING: could not list the repo; falling back to known names."
        log "         Check $API by hand for files this misses."
        # `opt`: this list is guessed, so a 404 is the whitelist being wrong
        # about one repo rather than a file going astray.
        get_small opt config.json generation_config.json tokenizer.json \
                  tokenizer_config.json tiktoken.model preprocessor_config.json \
                  chat_template.jinja configuration_kimi_k3.py \
                  modeling_kimi_k3.py modeling_kimi_linear.py
    fi
    # Said here as well as at the end, because the shard download that
    # follows can run for hours and this is the moment it is still cheap
    # to fix.
    if [ -n "$SMALL_MISSING" ]; then
        log "WARNING: the repo listed these and they are not on disk:"
        for f in $SMALL_MISSING; do log "         $f"; done
        log "         re-running fetches them; already-downloaded files are"
        log "         skipped, so it costs only what is missing."
    fi
fi

[ -s "$DEST/model.safetensors.index.json" ] || {
    log "FATAL: no safetensors index — single-file model or a different"
    log "       layout. Inspect $API and adapt."
    exit 1; }

# --- plan ------------------------------------------------------------------
# Every python3 below is piped through `tr -d '\r'`, and that is load-bearing
# rather than tidy (#36, gap 2). MSYS2 ships a *Windows* python3, whose stdout
# does text-mode \n -> \r\n; .download-state is appended by bash and does not.
# So `grep -qxF "$f" "$STATE"` compared "model-00001-of-00020.safetensors\r"
# against the same name without it and never matched: on Windows every shard
# looked absent, a finished 91.5 GB download reported "0 / 20 complete", and
# the free-space check then refused to start on a total it had already
# fetched. Everything the header above promises about surviving a long haul
# was inert there, and the first run looks perfect, which is what makes it
# easy to miss.
#
# `tr` rather than sys.stdout.reconfigure(): this has to be right on a Python
# build that cannot be tested from here, and deleting a byte that must never
# appear in a safetensors filename is checkable by reading. It is a no-op
# everywhere else.
"$PY" - "$DEST/model.safetensors.index.json" <<'PY' | tr -d '\r' > "$DEST/.shards"
import json, sys
idx = json.load(open(sys.argv[1]))
for s in sorted(set(idx["weight_map"].values())):
    print(s)
PY
TOTAL=$(wc -l < "$DEST/.shards" | tr -d ' ')
TOTAL_BYTES=$("$PY" - "$DEST/model.safetensors.index.json" <<'PY' | tr -d '\r'
import json, sys
# Qwen stores total_size as a JSON float (359999963128.0). bash $(( ))
# cannot parse the trailing .0, so this must be a bare integer.
print(int(json.load(open(sys.argv[1])).get("metadata", {}).get("total_size", 0) or 0))
PY
)

# What is already here, in bytes rather than by shard count: a proportional
# estimate is wrong whenever shards differ in size, which they do.
have=0
have_bytes=0
while read -r f; do
    if grep -qxF "$f" "$STATE"; then
        have=$((have + 1))
        have_bytes=$((have_bytes + $(fsize "$DEST/$f")))
    fi
done < "$DEST/.shards"

log "repo $REPO -> $DEST  (stat: $STAT_MODE)"
"$PY" - "$TOTAL_BYTES" "$have_bytes" "$(free_kb)" "$TOTAL" "$have" <<'PY'
import sys
tot, got, availkb, n, nhave = (int(x) for x in sys.argv[1:6])
g = 1 << 30
avail = availkb * 1024
# The index's total_size counts tensor bytes; the files on disk carry their
# safetensors headers too, so a complete download measures slightly *more*
# than the advertised total. Without the clamp the last line of a finished
# run reads "still to fetch: -0.1 GB".
todo = max(0, tot - got)
print(f"\nshards        : {nhave} / {n} complete")
print(f"download size : {tot/g:8.1f} GB")
print(f"already here  : {got/g:8.1f} GB")
print(f"still to fetch: {todo/g:8.1f} GB")
print(f"free on dest  : {avail/g:8.1f} GB")
print(f"after download: {(avail-todo)/g:8.1f} GB free\n")
PY

# --- verify-only ----------------------------------------------------------
if [ "$CHECK_ONLY" = 1 ]; then
    log "--check: verifying sizes on disk against the remote"
    bad=0
    fixed=0
    dropped=0
    while read -r f; do
        [ -f "$DEST/$f" ] || continue
        want=$(hcurl -sIL --max-time 60 "$RAW/$f" \
               | awk -F': ' '/^[Cc]ontent-[Ll]ength/{print $2}' | tr -d '\r' | tail -1)
        got=$(fsize "$DEST/$f")
        if [ "$want" != "$got" ]; then
            log "  INCOMPLETE $f ($got / ${want:-?})"
            bad=$((bad + 1))
            # The other direction of the repair below. A crash of the machine
            # (or of a WSL VM) can keep the line in $STATE but lose the end of
            # the file it vouches for: seen 2026-11-27, six shards listed as
            # done and short on disk, and every later run skipped them, said
            # ALL SHARDS COMPLETE and left --check failing for ever. The
            # remote is the stronger evidence, so the ledger loses the entry
            # and the next plain run resumes the shard.
            if grep -qxF "$f" "$STATE" 2>/dev/null; then
                grep -vxF "$f" "$STATE" > "$STATE.tmp"; mv "$STATE.tmp" "$STATE"
                dropped=$((dropped + 1))
            fi
        elif ! grep -qxF "$f" "$STATE" 2>/dev/null; then
            # It matches the remote and the state file does not say so.
            # The remote is the stronger evidence of the two, and this is
            # the only place that has asked it — so record it, rather than
            # leave convert.py reading a ledger that disagrees with the
            # bytes on disk and refusing a checkpoint that is complete.
            echo "$f" >> "$STATE"
            fixed=$((fixed + 1))
        fi
    done < "$DEST/.shards"
    log "--check done: $bad incomplete$([ "$fixed" -gt 0 ] && \
        printf ', %s verified against the remote and recorded' "$fixed")$([ "$dropped" -gt 0 ] && \
        printf ', %s taken off %s so the next run resumes them' "$dropped" "${STATE##*/}")"
    # --verify: the same crash also leaves files at the right SIZE with the
    # wrong bytes (2026-11-27: 16 of 20 shards, zero-filled blocks where
    # the lost writes were). Size cannot see that; the Hub's LFS sha256 can.
    # A mismatch is renamed *.bad (kept for a look, never fed to convert.py)
    # and dropped from $STATE, so the next plain run fetches it whole.
    if [ "$VERIFY" = 1 ] && [ "$bad" = 0 ]; then
        log "--verify: sha256 of every shard against the Hub (reads every byte)"
        "$PY" - "$REPO" "$REVISION" "$DEST" "$STATE" > "$DEST/.verify.out" 2>&1 <<'PY'
import concurrent.futures as cf, hashlib, json, os, sys, urllib.request
repo, rev, dest, state = sys.argv[1:5]
req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true")
if os.environ.get("HF_TOKEN"):
    req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
want = {e["path"]: e["lfs"]["oid"] for e in json.load(urllib.request.urlopen(req, timeout=90))
        if e.get("lfs") and os.path.exists(os.path.join(dest, e["path"]))}
if not want:
    print("  FAILED: the Hub listed no LFS hashes for files on disk"); sys.exit(2)
def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(dest, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return p, h.hexdigest()
bad = []
with cf.ThreadPoolExecutor(4) as ex:
    for p, got in ex.map(sha, sorted(want)):
        if got != want[p]:
            bad.append(p)
            os.replace(os.path.join(dest, p), os.path.join(dest, p + ".bad"))
            print(f"  BAD sha256 {p}: renamed {p}.bad, will be fetched again")
if bad and os.path.exists(state):
    keep = [l for l in open(state).read().split("\n") if l.strip() and l.strip() not in bad]
    open(state, "w").write("".join(l + "\n" for l in keep))
print(f"  --verify done: {len(want) - len(bad)} / {len(want)} match the Hub")
sys.exit(1 if bad else 0)
PY
        vrc=$?
        tr -d '\r' < "$DEST/.verify.out" | while read -r l; do log "$l"; done
        rm -f "$DEST/.verify.out"
        [ "$vrc" = 0 ] || bad=1
    fi
    rm -f "$DEST/.shards"
    exit $(( bad > 0 ))
fi

# --- preflight ------------------------------------------------------------
TODO_BYTES=$(( TOTAL_BYTES - have_bytes ))
[ "$TODO_BYTES" -lt 0 ] && TODO_BYTES=0
NEED_KB=$(( TODO_BYTES / 1024 * 105 / 100 ))                    # 5% margin
if [ "$(free_kb)" -lt "$NEED_KB" ]; then
    log "FATAL: need ~$(( NEED_KB / 1048576 )) GB with margin, only $(free_gb) GB free"
    exit 1
fi

if [ "$DRY" = 1 ]; then
    echo "--dry-run: stopping before download."
    rm -f "$DEST/.shards"
    exit 0
fi

# --- worker ---------------------------------------------------------------
# In its own file rather than inline: passing this to `xargs -I` blows
# macOS's command-line assembly limit.
cat > "$DEST/.worker.sh" <<WORKER
#!/bin/bash
f="\$1"
# GNU xargs runs the command once with no argument when its input is empty
# (every shard already recorded), and that call then retried a pull of ""
# eight times with backoff, minutes of nothing. No name, nothing to do.
[ -n "\$f" ] || exit 0
dest="$DEST"; raw="$RAW"; state="$STATE"; log="$LOG"
max_retry=$MAX_RETRY; min_free=$MIN_FREE_GB
hcurl() {
    if [ -n "\${HF_TOKEN:-}" ]; then
        curl -H "Authorization: Bearer \$HF_TOKEN" "\$@"
    else
        curl "\$@"
    fi
}
if [ "$STAT_MODE" = gnu ]; then
    fsize() { stat -c%s "\$1" 2>/dev/null || echo 0; }
else
    fsize() { stat -f%z "\$1" 2>/dev/null || echo 0; }
fi

say() { printf '%s  %s\n' "\$(date '+%H:%M:%S')" "\$*" >> "\$log"; }

grep -qxF "\$f" "\$state" && exit 0

for try in \$(seq 1 \$max_retry); do
    free=\$(( \$(df -kP "\$dest" | awk 'NR==2 {print \$4}') / 1048576 ))
    if [ "\$free" -lt "\$min_free" ]; then say "STOP \$f: only \${free} GB free"; exit 2; fi

    want=\$(hcurl -sIL --max-time 90 "\$raw/\$f" \\
           | awk -F': ' '/^[Cc]ontent-[Ll]ength/{print \$2}' | tr -d '\r' | tail -1)
    got=\$(fsize "\$dest/\$f")
    if [ -n "\$want" ] && [ "\$got" = "\$want" ]; then
        # A single line well under the pipe buffer, so concurrent appends
        # from the other workers do not interleave. flock is not portable
        # enough to rely on here.
        echo "\$f" >> "\$state"
        say "ok   \$f (\$((got/1048576)) MB)"
        exit 0
    fi

    # A file LONGER than the shard is not a partial download of it, and
    # -C - cannot fix it: the range it asks for starts past the end, every
    # retry fails the same way, and the run gives up on a shard whose bytes
    # are all on the CDN. Seen for real — killing this script mid-shard and
    # restarting it left 9.03 GB where the shard is 5.36 GB, and 8 retries
    # went into resuming from an offset that does not exist. Start clean.
    if [ -n "\$want" ] && [ "\$got" -gt "\$want" ]; then
        say "\$f is \$((got/1048576)) MB against \$((want/1048576)) MB, restarting from zero"
        rm -f "\$dest/\$f"
        got=0
    fi

    [ "\$got" -gt 0 ] && say "resume \$f at \$((got/1048576)) MB (try \$try)" \\
                      || say "pull \$f (try \$try)"
    # -C - resumes; --speed-limit kills a connection that has stalled rather
    # than waiting out a TCP timeout that may never come.
    #
    # --http1.1 is not a preference. Over a long single transfer this CDN
    # returns CURLE_HTTP2_STREAM (rc=92) often enough to matter: measured
    # on the DeepSeek-V4.1 pull, one 7.4 GB shard took five resumes and an
    # hour where its neighbours took twenty minutes each. The retry logic
    # survives it and the throughput does not, and a multi-hundred-GB
    # download is exactly where that compounds. HTTP/1.1 costs a little
    # multiplexing this script never uses — one file per connection.
    hcurl -fL --http1.1 -C - --retry 3 --retry-delay 5 \\
          --speed-limit 1024 --speed-time 120 \\
          -o "\$dest/\$f" "\$raw/\$f" 2>/dev/null
    rc=\$?
    [ \$rc -eq 0 ] && continue          # the next pass verifies the size

    # 33 is "server does not support byte ranges". Retrying the resume can
    # only fail the same way, so drop the partial file and let the next try
    # start clean — otherwise a mirror without Range support burns every
    # retry and gives up on a file it could have fetched whole.
    if [ \$rc -eq 33 ]; then
        say "no range support for \$f, restarting from zero"
        rm -f "\$dest/\$f"
        continue
    fi

    wait=\$(( (1 << (try > 6 ? 6 : try)) * 5 + RANDOM % 10 ))
    say "fail \$f rc=\$rc, retry in \${wait}s"
    sleep \$wait
done
# The size check happens at the TOP of the loop, so a transfer that
# completed on the last attempt is never verified and never recorded. Seen
# for real: two 7.4 GB shards finished, took an rc=18 on the retry after
# that, and were reported GIVE UP while sitting on disk byte-exact — and
# .download-state then disagreed with the remote for the rest of the run.
# Ask once more before saying no.
want=\$(hcurl -sIL --max-time 90 "\$raw/\$f" \\
       | awk -F': ' '/^[Cc]ontent-[Ll]ength/{print \$2}' | tr -d '\r' | tail -1)
got=\$(fsize "\$dest/\$f")
if [ -n "\$want" ] && [ "\$got" = "\$want" ]; then
    echo "\$f" >> "\$state"
    say "ok   \$f (\$((got/1048576)) MB, on the last attempt)"
    exit 0
fi
say "GIVE UP \$f after \$max_retry tries"
exit 1
WORKER
chmod +x "$DEST/.worker.sh"

log "downloading with $JOBS parallel streams (resumable, $MAX_RETRY retries each)"
grep -vxF -f "$STATE" "$DEST/.shards" 2>/dev/null \
    | xargs -P "$JOBS" -n1 "$DEST/.worker.sh"
rc=$?

# How many shards are actually on disk, or a refusal.
#
# The destination can go away under a long run. A USB enclosure that drops
# off the bus takes $STATE with it; `wc -l` then produces nothing, and
# `[ "" -lt 48 ]` is an error that `test` reports as false — so the run fell
# through every check after it and printed ALL SHARDS COMPLETE with rc=0
# over a directory that no longer existed. Measured on 2026-09-15, 184 GB
# into a 475 GB pull, and the next thing that would have happened is
# convert.py writing a container out of 39% of a model.
#
# So: ask whether the destination is still there before believing anything
# counted from it, and refuse a count that is not a number. #35 was this
# same shape one level up — completion reported on evidence nobody read.
count_done() {
    # Its answer goes to stdout, so everything it says goes to stderr —
    # otherwise the caller's $(count_done) would capture the diagnostics as
    # if they were the count.
    if [ ! -d "$DEST" ] || ! : > "$DEST/.writable" 2>/dev/null; then
        log "FAILED: $DEST is gone or not writable. The download is NOT" >&2
        log "        complete; reconnect the device and re-run — nothing" >&2
        log "        already on disk is refetched." >&2
        return 1
    fi
    rm -f "$DEST/.writable"
    local n
    n=$(wc -l < "$STATE" 2>/dev/null | tr -d ' ')
    case "$n" in
        ''|*[!0-9]*)
            log "FAILED: cannot read $STATE, so how much is on disk is" >&2
            log "        unknown. That is incomplete, not complete." >&2
            return 1 ;;
    esac
    printf '%s\n' "$n"
}

done_now=$(count_done) || exit 1
log "pass finished (rc=$rc): $done_now / $TOTAL shards complete, $(free_gb) GB free"
if [ "$done_now" -lt "$TOTAL" ]; then
    log "re-run to continue; nothing already downloaded is refetched"
    exit 1
fi
# The shards are what takes the hours, but they are not the whole checkpoint,
# and reporting completion on them alone is what #35 was about: the run said
# ALL SHARDS COMPLETE and rc=0 over a directory convert.py could not use.
if [ -n "$SMALL_MISSING" ]; then
    log "shards complete, but these files the repo listed are not on disk:"
    for f in $SMALL_MISSING; do log "    $f"; done
    log "NOT COMPLETE: re-run to fetch them. convert.py needs config.json,"
    log "              and without tiktoken.model it still succeeds and"
    log "              writes a container that cannot tokenize."
    exit 1
fi
log "ALL SHARDS COMPLETE"
rm -f "$DEST/.worker.sh" "$DEST/.shards"
echo "Next: tools/convert.py --src $DEST --out model.waste"
