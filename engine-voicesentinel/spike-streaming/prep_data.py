"""Builds the test sets. Run once: python prep_data.py <librispeech_root> <spike_parquet>.

Every clip goes through the production audio path: 16 kHz source -> 48 kHz -> Opus 24 kbps VoIP -> 48 kHz PCM,
then 0.5 s of room noise before the speech and 1.5 s after it. Word times come from CTC forced alignment of
the reference transcript with wav2vec2-base-960h (accurate to roughly +-50 ms; word ends tend to be early)."""
import collections, glob, json, os, random, subprocess, sys, shutil
from multiprocessing import Pool
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, lfilter
import onnxruntime as ort
from common import DATA, SR, BAD, TTS_WORDS

LEAD, TAIL = 0.5, 1.5
W2V = os.environ.get("W2V", os.path.dirname(DATA) + "/w2v")
VOCAB = json.load(open(W2V + "/vocab.json"))

# Stand-in "bad words": ordinary mid-frequency words/phrases so natural human speech can be used.
# The real profanity list (BAD) is used for false alarms and the TTS test.
CARRIERS = ["I can't believe you are such a {w} to everyone here today",
            "Honestly that was the most {w} thing I have heard all week and I am done",
            "Why would you say {w} in front of all of them just now",
            "He looked at me and said {w} before he walked out of the room",
            "Come on you {w} give me back my stuff right now",
            "The whole team kept calling him a {w} and he just laughed about it"]
VOICES = ["Samantha", "Daniel", "Karen", "Rishi"]


def read16(path):
    a, sr = sf.read(path, dtype="float32")
    assert sr == 16000
    return a


def to_opus48(a16, noise=None, seed=0):
    """16 kHz float -> padded 48 kHz -> Opus roundtrip -> int16 48 kHz."""
    a = resample_poly(a16, 3, 1).astype(np.float32)
    rng = np.random.default_rng(seed)
    body = a.copy()
    if noise is not None:
        kind, snr, bed = noise
        rms = np.sqrt(np.mean(a ** 2) + 1e-12)
        if kind == "babble":
            n = np.zeros(len(a), np.float32)
            for _ in range(4):
                b = bed[rng.integers(len(bed))]
                b = np.resize(b, len(a)) if len(b) < len(a) else b[:len(a)]
                n += b
        else:  # hiss + rumble
            n = lfilter([1.0], [1.0, -0.6], rng.normal(0, 1, len(a))).astype(np.float32)
        n *= rms / (np.sqrt(np.mean(n ** 2) + 1e-12) * 10 ** (snr / 20))
        body = a + n
    floor = lambda s: rng.normal(0, 3e-4, int(s * SR)).astype(np.float32)
    full = np.concatenate([floor(LEAD), body, floor(TAIL)])
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", "48000", "-ac", "1", "-i", "pipe:0",
                        "-c:a", "libopus", "-b:a", "24k", "-application", "voip", "-frame_duration", "20",
                        "-f", "ogg", "pipe:1"], input=full.tobytes(), capture_output=True)
    q = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ar", "48000", "-ac", "1", "-f", "s16le",
                        "pipe:1"], input=p.stdout, capture_output=True)
    return np.frombuffer(q.stdout, dtype=np.int16)


def _job(args):
    a16, noise, seed = args
    return to_opus48(a16, noise, seed)


class Aligner:
    def __init__(self):
        self.s = ort.InferenceSession(W2V + "/model.onnx", providers=["CPUExecutionProvider"])

    def words(self, a16, text):
        x = ((a16 - a16.mean()) / (a16.std() + 1e-7)).astype(np.float32)[None]
        logits = self.s.run(None, {self.s.get_inputs()[0].name: x})[0][0]
        logp = logits - np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True)) - logits.max(1, keepdims=True)
        ws = text.upper().split()
        tgt, owner = [], []
        for i, w in enumerate(ws):
            if i:
                tgt.append(VOCAB["|"]); owner.append(-1)
            for c in w:
                tgt.append(VOCAB.get(c, 3)); owner.append(i)
        path = ctc_align(logp, np.array(tgt))
        T = len(logp)
        first, last = {}, {}
        for t in range(T):
            j = path[t]
            if j >= 0 and owner[j] >= 0:
                first.setdefault(owner[j], t); last[owner[j]] = t
        return [(ws[i].lower(), first[i] * 0.02, (last[i] + 1) * 0.02) for i in range(len(ws)) if i in first]


def ctc_align(logp, tgt, blank=0):
    T, L = len(logp), len(tgt)
    S = 2 * L + 1
    ext = np.full(S, blank); ext[1::2] = tgt
    NEG = -1e30
    dp = np.full((T, S), NEG, np.float32); bp = np.zeros((T, S), np.int8)
    dp[0, 0] = logp[0, blank]; dp[0, 1] = logp[0, ext[1]]
    skip = np.zeros(S, bool)
    for s in range(3, S, 2):
        skip[s] = ext[s] != ext[s - 2]
    for t in range(1, T):
        p = dp[t - 1]
        c1 = np.concatenate([[NEG], p[:-1]]); c2 = np.concatenate([[NEG, NEG], p[:-2]]); c2 = np.where(skip, c2, NEG)
        st = np.stack([p, c1, c2]); a = st.argmax(0)
        dp[t] = st.max(0) + logp[t, ext]; bp[t] = a
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.full(T, -1)
    for t in range(T - 1, -1, -1):
        path[t] = (s - 1) // 2 if s % 2 == 1 else -1
        s -= int(bp[t, s])
    return path


def occurrences(words, entries):
    """All places a watch entry (word or phrase) occurs in the aligned transcript."""
    out = []
    toks = [w for w, _, _ in words]
    for e in entries:
        parts = e.split()
        for i in range(len(toks) - len(parts) + 1):
            if toks[i:i + len(parts)] == parts:
                out.append({"w": e, "s": words[i][1] + LEAD, "e": words[i + len(parts) - 1][2] + LEAD,
                            "pos": i / max(1, len(toks))})
    return sorted(out, key=lambda o: o["s"])


def utts(root, split):
    res = []
    for tf in sorted(glob.glob(f"{root}/{split}/*/*/*.trans.txt")):
        d = os.path.dirname(tf)
        for line in open(tf):
            k, t = line.strip().split(" ", 1)
            res.append((k, f"{d}/{k}.flac", t))
    return res


def save(name, items):
    os.makedirs(f"{DATA}/proc/{name}", exist_ok=True)
    for it in items:
        np.save(f"{DATA}/proc/{name}/{it['id']}.npy", it.pop("pcm"))
    json.dump(items, open(f"{DATA}/{name}.json", "w"))
    print(name, len(items), "clips", round(sum(i["dur"] for i in items) / 3600, 2), "h", flush=True)


def main(root, parquet):
    random.seed(7)
    pool = Pool(8)
    clean, other = utts(root, "test-clean"), utts(root, "test-other")

    # --- WER set from the earlier engine spike (73 LibriSpeech dev-clean clips) ---
    import pyarrow.parquet as pq, io
    rows = pq.read_table(parquet).to_pylist()
    a16 = [sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")[0] for r in rows]
    pcm = pool.map(_job, [(a, None, i) for i, a in enumerate(a16)])
    save("wer73", [{"id": f"d{i}", "text": r["text"].lower(), "dur": len(p) / SR, "speech_end": LEAD + len(a) / 16000,
                    "pcm": p} for i, (r, a, p) in enumerate(zip(rows, a16, pcm))])

    # --- false-alarm sets: every 2nd test-clean / every 3rd test-other utterance, no alignment needed ---
    for name, src, step in (("fa_clean", clean, 2), ("fa_other", other, 3)):
        sub = src[::step]
        a = [read16(p) for _, p, _ in sub]
        pcm = pool.map(_job, [(x, None, i) for i, x in enumerate(a)], chunksize=8)
        save(name, [{"id": k, "text": t.lower(), "dur": len(p) / SR, "speech_end": LEAD + len(x) / 16000, "pcm": p}
                    for (k, _, t), x, p in zip(sub, a, pcm)])

    # --- stand-in watch list: mid-frequency words + a few phrases ---
    cnt, bi = collections.Counter(), collections.Counter()
    for _, _, t in clean + other:
        w = t.lower().split()
        if len(w) < 8:
            continue
        cnt.update(set(w)); bi.update(set(zip(w, w[1:])))
    words = [w for w, c in cnt.items() if 6 <= c <= 30 and 5 <= len(w) <= 9 and w.isalpha()]
    random.shuffle(words)
    phr = [" ".join(b) for b, c in bi.items() if 6 <= c <= 25 and all(len(x) >= 4 for x in b) and "jesus" not in b]
    random.shuffle(phr)
    P = sorted(words[:40]) + sorted(phr[:6])
    json.dump(P, open(f"{DATA}/standin_words.json", "w"))
    print("stand-in words:", P, flush=True)

    # --- positives: natural speech with a stand-in word in the middle of the sentence ---
    al = Aligner()
    sel = {"pos_clean": [], "pos_other": []}
    for name, src, want in (("pos_clean", clean, 110), ("pos_other", other, 90)):
        order = src[:]; random.shuffle(order)
        per = collections.Counter()
        for k, p, t in order:
            ws = t.lower().split()
            if len(ws) < 8 or len(ws) > 45:
                continue
            hit = [e for e in P if (e in " ".join(ws)) and per[e] < 5]
            if not hit:
                continue
            a = read16(p)
            if len(a) > 16000 * 20:
                continue
            occ = occurrences(al.words(a, t), P)
            mid = [o for o in occ if 0.2 <= o["pos"] <= 0.8]
            if not mid:
                continue
            for o in mid:
                per[o["w"]] += 1
            sel[name].append({"id": k, "text": t.lower(), "a": a, "occ": occ, "speech_end": LEAD + len(a) / 16000})
            if len(sel[name]) >= want:
                break
    bed = [resample_poly(read16(p), 3, 1).astype(np.float32) for _, p, _ in clean[1::37][:40]]
    variants = [("pos_clean", "pos_clean", None), ("pos_other", "pos_other", None),
                ("pos_babble10", "pos_clean", ("babble", 10, bed)), ("pos_white15", "pos_clean", ("hiss", 15, bed))]
    for name, base, noise in variants:
        items = sel[base]
        pcm = pool.map(_job, [(it["a"], noise, i) for i, it in enumerate(items)])
        save(name, [{"id": it["id"], "text": it["text"], "occ": it["occ"], "dur": len(p) / SR,
                     "speech_end": it["speech_end"], "pcm": p} for it, p in zip(items, pcm)])

    # --- TTS with real profanity (synthetic voice; tokenisation + recall sanity) ---
    tmp = DATA + "/tts_tmp"; os.makedirs(tmp, exist_ok=True)
    items = []
    for v in VOICES:
        for w in TTS_WORDS:
            for ci, c in enumerate(CARRIERS):
                text = c.format(w=w)
                f = f"{tmp}/{v}_{w.replace(' ', '_')}_{ci}"
                subprocess.run(["say", "-v", v, "-o", f + ".aiff", text], check=True)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f + ".aiff", "-ar", "16000", "-ac", "1", f + ".wav"], check=True)
                a = read16(f + ".wav")
                occ = occurrences(al.words(a, text.lower()), [w])
                items.append({"id": f"{v}_{w.replace(' ', '_')}_{ci}", "text": text.lower(), "a": a, "occ": occ,
                              "speech_end": LEAD + len(a) / 16000})
    pcm = pool.map(_job, [(it["a"], None, i) for i, it in enumerate(items)])
    save("tts", [{"id": it["id"], "text": it["text"], "occ": it["occ"], "dur": len(p) / SR,
                  "speech_end": it["speech_end"], "pcm": p} for it, p in zip(items, pcm)])
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
