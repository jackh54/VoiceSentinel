"""Shared helpers for the streaming spike: model registry, word matching, real-time simulation.

Everything here uses sherpa-onnx 2.0.1 from Python. That is the same C++ runtime the Java engine
reaches through JNI, so decode cost is the same; only the thin binding differs.
"""
import json, os, re, time
import numpy as np
import sherpa_onnx as so

ROOT = os.environ.get("ST_ROOT", os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("ST_MODELS", ROOT + "/models")
DATA = os.environ.get("ST_DATA", ROOT + "/data")
OUT = os.environ.get("ST_OUT", ROOT + "/results")
SR = 48000          # plugin audio rate (Opus decoded)
CHUNK = 960         # 20 ms at 48 kHz
THREADS = int(os.environ.get("ST_THREADS", "2"))

# name -> (dir, encoder, decoder, joiner, extra kwargs). All int8, as shipped in the engine zips.
STREAMING = {
    "zip20M":     ("sherpa-onnx-streaming-zipformer-en-20M-2023-02-17", "encoder-epoch-99-avg-1.int8.onnx",
                   "decoder-epoch-99-avg-1.int8.onnx", "joiner-epoch-99-avg-1.int8.onnx", {}),
    "zip2023":    ("sherpa-onnx-streaming-zipformer-en-2023-06-26", "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                   "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                   "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx", {}),
    "nemo80":     ("sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8", "encoder.int8.onnx",
                   "decoder.int8.onnx", "joiner.int8.onnx", {}),
    "nemo480":    ("sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8", "encoder.int8.onnx",
                   "decoder.int8.onnx", "joiner.int8.onnx", {}),
    "nemo1040":   ("sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms-int8", "encoder.int8.onnx",
                   "decoder.int8.onnx", "joiner.int8.onnx", {}),
}
# Real profanity watch list for false-alarm runs, and the words spoken in the TTS test.
BAD = ["damn", "hell", "shit", "fuck", "fucking", "bitch", "bastard", "asshole", "ass", "crap", "dick", "piss",
       "pissed", "slut", "whore", "cunt", "retard", "idiot", "stupid", "moron", "dumbass", "jackass", "bullshit",
       "motherfucker", "douche", "prick", "wanker", "bollocks", "twat", "cock", "porn", "sex", "nazi",
       "kill yourself", "shut up", "go to hell", "son of a bitch", "piece of shit"]
TTS_WORDS = ["damn", "hell", "shit", "fuck", "bitch", "bastard", "asshole", "crap", "dick", "idiot", "moron",
             "bullshit", "motherfucker", "dumbass", "jackass", "shut up", "son of a bitch"]
KWS_DIR = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
PARAKEET = "sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8"
MOONSHINE = "sherpa-onnx-moonshine-base-en-int8"


def build_streaming(name, threads=THREADS):
    d, enc, dec, joi, extra = STREAMING[name]
    d = f"{MODELS}/{d}/"
    return so.OnlineRecognizer.from_transducer(
        tokens=d + "tokens.txt", encoder=d + enc, decoder=d + dec, joiner=d + joi,
        num_threads=threads, sample_rate=16000, feature_dim=80, decoding_method="greedy_search", **extra)


def build_parakeet(threads=THREADS):
    d = f"{MODELS}/{PARAKEET}/"
    return so.OfflineRecognizer.from_transducer(
        encoder=d + "encoder.int8.onnx", decoder=d + "decoder.int8.onnx", joiner=d + "joiner.int8.onnx",
        tokens=d + "tokens.txt", num_threads=threads, model_type="nemo_transducer")


def build_moonshine(threads=THREADS):
    d = f"{MODELS}/{MOONSHINE}/"
    return so.OfflineRecognizer.from_moonshine(
        preprocessor=d + "preprocess.onnx", encoder=d + "encode.int8.onnx",
        uncached_decoder=d + "uncached_decode.int8.onnx", cached_decoder=d + "cached_decode.int8.onnx",
        tokens=d + "tokens.txt", num_threads=threads)


def kws_keyword_lines(words):
    """BPE-tokenise arbitrary words/phrases the way sherpa-onnx-cli text2token does."""
    d = f"{MODELS}/{KWS_DIR}/"
    toks = so.text2token([w.upper() for w in words], tokens=d + "tokens.txt", tokens_type="bpe",
                         bpe_model=d + "bpe.model")
    return [" ".join(t) + " @" + w.upper().replace(" ", "_") for t, w in zip(toks, words)]


def build_kws(words, threshold=0.25, score=1.0, trailing_blanks=1, threads=THREADS, workdir="/tmp"):
    d = f"{MODELS}/{KWS_DIR}/"
    path = f"{workdir}/kw_{os.getpid()}_{abs(hash(tuple(words)))}.txt"
    with open(path, "w") as f:
        f.write("\n".join(kws_keyword_lines(words)) + "\n")
    suf = "chunk-16-left-64.int8.onnx"
    return so.KeywordSpotter(
        tokens=d + "tokens.txt", encoder=d + "encoder-epoch-12-avg-2-" + suf,
        decoder=d + "decoder-epoch-12-avg-2-" + suf, joiner=d + "joiner-epoch-12-avg-2-" + suf,
        keywords_file=path, num_threads=threads, keywords_score=score, keywords_threshold=threshold,
        num_trailing_blanks=trailing_blanks)


def norm(s):
    return re.sub(r"[^a-z' ]", "", s.lower()).strip()


_pat = {}


def find_words(text, words, partial=False):
    """Same rules as the engine's WordMatcher (whole word unless partial). Returns {word: count}."""
    out = {}
    t = text.lower()
    for w in words:
        w = w.lower()
        if not w:
            continue
        if partial:
            n = t.count(w)
        else:
            p = _pat.get(w)
            if p is None:
                p = _pat[w] = re.compile(r"(?<![\w])" + re.escape(w) + r"(?![\w])")
            n = len(p.findall(t))
        if n:
            out[w] = n
    return out


def rtext(r):
    return (r if isinstance(r, str) else r.text).strip()


class Det:
    """One speaker's fast-path detector. feed() returns new (word) events after each 20 ms chunk."""

    def feed(self, chunk):
        raise NotImplementedError

    def finish(self):
        return []


class AsrDet(Det):
    def __init__(self, rec, words, partial=False):
        self.rec, self.words, self.partial = rec, words, partial
        self.s = rec.create_stream()
        self.seen = {}
        self.text = ""

    def feed(self, chunk):
        self.s.accept_waveform(SR, chunk)
        ev, ran = [], False
        while self.rec.is_ready(self.s):
            self.rec.decode_stream(self.s)
            ran = True
        if ran:
            self.text = rtext(self.rec.get_result(self.s))
            for w, n in find_words(self.text, self.words, self.partial).items():
                if n > self.seen.get(w, 0):
                    ev += [w] * (n - self.seen.get(w, 0))
                self.seen[w] = max(n, self.seen.get(w, 0))
        return ev

    def finish(self):
        self.s.accept_waveform(SR, np.zeros(SR // 2, dtype=np.float32))  # tail padding flushes the model
        self.s.input_finished()
        while self.rec.is_ready(self.s):
            self.rec.decode_stream(self.s)
        self.text = rtext(self.rec.get_result(self.s))
        return self.text


class KwsDet(Det):
    def __init__(self, kws):
        self.kws = kws
        self.s = kws.create_stream()

    def feed(self, chunk):
        self.s.accept_waveform(SR, chunk)
        ev = []
        while self.kws.is_ready(self.s):
            self.kws.decode_stream(self.s)
            r = self.kws.get_result(self.s)
            if r:
                ev.append(r.lower().replace("_", " "))
                self.kws.reset_stream(self.s)
        return ev


def simulate(det, audio, hop=CHUNK):
    """Feed 20 ms chunks on a virtual real-time clock. Chunk i exists at (i+1)*20 ms; the detector can
    only start on it once it arrived and the previous chunk is done. Returns (events, cpu_seconds, per-step list).
    events = [(word, emit_time_s)]."""
    free, events, cpu = 0.0, [], 0.0
    n = len(audio) // hop
    for i in range(n):
        arrive = (i + 1) * hop / SR
        t0 = time.perf_counter()
        ev = det.feed(audio[i * hop:(i + 1) * hop])
        dt = time.perf_counter() - t0
        start = max(arrive, free)
        free = start + dt
        cpu += dt
        for w in ev:
            events.append((w, free))
    return events, cpu


def load_manifest(name):
    with open(f"{DATA}/{name}.json") as f:
        return json.load(f)


def load_audio(name):
    z = np.load(f"{DATA}/{name}.npz")
    return [z[k] for k in sorted(z.files, key=lambda x: int(x[1:]))]


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def load_clips(name):
    """Yields (manifest entry, float32 48 kHz audio) one clip at a time."""
    for it in load_manifest(name):
        yield it, np.load(f"{DATA}/proc/{name}/{it['id']}.npy").astype(np.float32) / 32768.0


def standin_words():
    return load_manifest("standin_words")


def make_detector(cand, words, recs):
    """cand: asr:<name> | kws:<score>:<threshold>. recs caches recognizers (they are expensive to build)."""
    kind, *rest = cand.split(":")
    if kind == "asr":
        if cand not in recs:
            recs[cand] = build_streaming(rest[0])
        return lambda: AsrDet(recs[cand], words)
    if kind == "kws":
        key = (cand, tuple(words))
        if key not in recs:
            recs[key] = build_kws(words, score=float(rest[0]), threshold=float(rest[1]))
        return lambda: KwsDet(recs[key])
    raise ValueError(cand)


class Confirm:
    """Double Check: run the offline models on the last ~3 s of audio and see whether they agree."""

    def __init__(self, threads=THREADS):
        self.p, self.m = build_parakeet(threads), build_moonshine(threads)

    @staticmethod
    def _run(rec, a):
        t0 = time.perf_counter()
        s = rec.create_stream()
        s.accept_waveform(SR, a)
        rec.decode_stream(s)
        return s.result.text.strip(), time.perf_counter() - t0

    def check(self, audio, t, word, window=3.0):
        a = audio[int(max(0.0, t - window) * SR):int(t * SR)]
        pt, pd = self._run(self.p, a)
        mt, md = self._run(self.m, a)
        return {"p_ok": bool(find_words(pt, [word])), "m_ok": bool(find_words(mt, [word])), "p_ms": pd * 1e3,
                "m_ms": md * 1e3, "p_text": pt, "m_text": mt}


def score_clip(events, occ, ref):
    """Match fast-path events to annotated word occurrences. Returns (per-occurrence results, false alarms)."""
    used, res = set(), []
    for o in occ:
        best = None
        for i, (w, t) in enumerate(events):
            if i in used or w != o["w"] or t < o["s"] or t > o["e"] + 4.0:
                continue
            best = i
            break
        if best is None:
            res.append({"w": o["w"], "hit": False})
        else:
            used.add(best)
            res.append({"w": o["w"], "hit": True, "lat": events[best][1] - o["e"], "t": events[best][1]})
    fa = [(w, t) for i, (w, t) in enumerate(events) if i not in used and not find_words(ref, [w])]
    return res, fa
