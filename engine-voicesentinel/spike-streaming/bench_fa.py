"""False alarms on word-free speech (plus WER for streaming ASR) and how many Double Check removes.
usage: python bench_fa.py <tag> <cand> <set> [shard nshards]     set: fa_clean | fa_other | noise"""
import sys, time, json, os
from common import *

tag, cand, sname = sys.argv[1:4]
shard, nsh = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
os.makedirs(f"{OUT}/{tag}", exist_ok=True)
recs = {}
conf = Confirm(1)
mk = make_detector(cand, BAD, recs)


def noise_clips():
    rng = np.random.default_rng(3)
    n = 48000 * 300
    kinds = {
        "room": rng.normal(0, 3e-4, n),
        "hiss": lfilter_hiss(rng, n),
        "rumble_clicks": rumble_clicks(rng, n),
    }
    for k, v in kinds.items():
        for i in range(0, n, 48000 * 20):
            yield {"id": f"{k}{i}", "text": "", "dur": 20.0}, v[i:i + 48000 * 20].astype(np.float32)


def lfilter_hiss(rng, n):
    from scipy.signal import lfilter
    return lfilter([1.0], [1.0, -0.6], rng.normal(0, 0.01, n))


def rumble_clicks(rng, n):
    from scipy.signal import lfilter
    x = lfilter([1.0], [1.0, -0.995], rng.normal(0, 0.002, n))
    for p in rng.integers(0, n - 200, n // 48000 * 2):
        x[p:p + 120] += rng.normal(0, 0.15, 120) * np.hanning(120)
    return x


src = noise_clips() if sname == "noise" else load_clips(sname)
audio_s = 0.0
fa, tp, refs, hyps, words_out = [], 0, [], [], 0
t0 = time.time()
for i, (it, audio) in enumerate(src):
    if i % nsh != shard:
        continue
    det = mk()
    events, cpu = simulate(det, audio)
    audio_s += it["dur"]
    _, bad = score_clip(events, [], it["text"])
    tp += len(events) - len(bad)
    for w, t in bad:
        d = conf.check(audio, t, w); d.update(w=w, id=it["id"], t=t); fa.append(d)
    if cand.startswith("asr:"):
        hyp = det.finish()
        words_out += len(hyp.split())
        if sname != "noise":
            refs.append(norm(it["text"])); hyps.append(norm(hyp))
out = {"cand": cand, "set": sname, "shard": shard, "audio_s": audio_s, "fa": fa, "true_hits": tp, "refs": refs,
       "hyps": hyps, "noise_words": words_out, "wall_s": time.time() - t0}
json.dump(out, open(f"{OUT}/{tag}/fa_{cand.replace(':', '_')}_{sname}_{shard}.json", "w"))
print(tag, cand, sname, shard, "audio_h", round(audio_s / 3600, 2), "fa", len(fa), "noise_words", words_out, flush=True)
