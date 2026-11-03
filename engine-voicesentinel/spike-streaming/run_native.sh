#!/bin/bash
# Native run (timing-critical, run on an otherwise idle machine). usage: run_native.sh <tag> <python>
# Trimmed to the finalists after the first pass over every model (see the report for the full list).
TAG=${1:-mac}; PY=${2:-python3}; D=$(cd "$(dirname "$0")" && pwd)
SETS=pos_clean,pos_other,tts
$PY $D/bench_wer.py $TAG off:parakeet,off:moonshine,asr:zip20M,asr:zip2023,asr:nemo80,asr:nemo480,asr:nemo1040
for c in today asr:zip20M asr:zip2023 asr:nemo80 asr:nemo480 asr:nemo1040 kws:1.0:0.25 kws:2.0:0.25 kws:3.0:0.1; do
  $PY $D/bench_latency.py $TAG $c $SETS
done
for c in asr:zip2023 asr:nemo80 asr:nemo480 asr:nemo1040 kws:2.0:0.25; do
  for n in 1 4 8; do $PY $D/bench_load.py $TAG $c $n 30; done
done
