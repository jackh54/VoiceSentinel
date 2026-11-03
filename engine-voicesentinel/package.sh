#!/usr/bin/env bash
# Packs the engine runtime and speech models into zips for the GitHub release.
set -euo pipefail
SHERPA=v2.0.1
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/release"
WORK="$(mktemp -d)"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"
rm -rf "$OUT"; mkdir -p "$OUT"
for platform in linux-x64 linux-aarch64 osx-arm64 osx-x86_64 win-x64; do
  dir="$WORK/engine-$platform"
  mkdir -p "$dir/native"
  curl -fsSL "$BASE/$SHERPA/sherpa-onnx-$SHERPA-$platform-jni.tar.bz2" | tar xj -C "$WORK"
  src="$WORK/sherpa-onnx-$SHERPA-$platform-jni/lib"
  cp "$src"/*onnxruntime* "$src"/*sherpa-onnx-jni* "$dir/native/"
  cp "$ROOT/target/engine.jar" "$dir/"
  cp "$ROOT/THIRD_PARTY.md" "$dir/"
  (cd "$dir" && zip -qr "$OUT/engine-$platform.zip" .)
done
fetch_model() {
  curl -fsSL "$BASE/${3:-asr-models}/$1.tar.bz2" | tar xj -C "$WORK"
  mkdir -p "$(dirname "$2")"
  mv "$WORK/$1" "$2"
  rm -rf "$2/test_wavs" "$2/.git"
}
EN="$WORK/model-en"
fetch_model sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8 "$EN/primary"
fetch_model sherpa-onnx-moonshine-base-en-int8 "$EN/confirm"
# Keyword spotter for live detection: the engine looks for <models>/kws and only advertises "live" when it exists.
fetch_model sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 "$EN/kws-full" kws-models
mkdir -p "$EN/kws"
K="$EN/kws-full"
cp "$K/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx" "$EN/kws/encoder.int8.onnx"
cp "$K/decoder-epoch-12-avg-2-chunk-16-left-64.onnx" "$EN/kws/decoder.onnx"
cp "$K/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx" "$EN/kws/joiner.int8.onnx"
cp "$K/tokens.txt" "$K/bpe.model" "$EN/kws/"
rm -rf "$K"
cp "$ROOT/THIRD_PARTY.md" "$EN/"
(cd "$EN" && zip -qr "$OUT/model-en.zip" .)
MULTI="$WORK/model-multi"
fetch_model sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 "$MULTI/primary"
cp "$ROOT/THIRD_PARTY.md" "$MULTI/"
(cd "$MULTI" && zip -qr "$OUT/model-multi.zip" .)
(cd "$OUT" && shasum -a 256 *.zip > SHA256SUMS)
cat "$OUT/SHA256SUMS"
