"""Word error rate on the 73-clip LibriSpeech set from the earlier engine spike, for offline and streaming models.
usage: python bench_wer.py <tag> <model,...>   model: off:parakeet | off:moonshine | asr:<name>"""
import sys, time, json, os, jiwer
from common import *

tag = sys.argv[1]
os.makedirs(f"{OUT}/{tag}", exist_ok=True)
res = {}
for m in sys.argv[2].split(","):
    kind, name = m.split(":")
    rec = {"off": lambda: build_parakeet() if name == "parakeet" else build_moonshine(), "asr": lambda: build_streaming(name)}[kind]()
    refs, hyps, cpu, dur = [], [], 0.0, 0.0
    for it, a in load_clips("wer73"):
        t0 = time.perf_counter()
        if kind == "off":
            hyp = Confirm._run(rec, a[int(0.5 * SR):int(it["speech_end"] * SR)])[0]
        else:
            d = AsrDet(rec, [])
            for i in range(len(a) // CHUNK):
                d.feed(a[i * CHUNK:(i + 1) * CHUNK])
            hyp = d.finish()
        cpu += time.perf_counter() - t0; dur += it["speech_end"] - 0.5
        refs.append(norm(it["text"])); hyps.append(norm(hyp))
    res[m] = {"wer": jiwer.wer(refs, hyps), "rtf": cpu / dur, "audio_s": dur}
    print(tag, m, res[m], flush=True)
json.dump(res, open(f"{OUT}/{tag}/wer_{sys.argv[2].replace(':', '_').replace(',', '+')}.json", "w"))
