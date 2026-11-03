"""N speakers talking at the same time, real wall-clock, 2 worker threads sharing one recognizer.
usage: python bench_load.py <tag> <cand> <N> [seconds]       Worst case: every speaker talks 100% of the time."""
import sys, time, json, os, threading, collections, psutil, queue
from concurrent.futures import ThreadPoolExecutor
from common import *

tag, cand, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
os.makedirs(f"{OUT}/{tag}", exist_ok=True)
rss0 = psutil.Process().memory_info().rss
recs = {}
if cand.startswith("asr:"):
    recs[cand] = build_streaming(cand.split(":")[1], threads=1)
    mk = lambda: AsrDet(recs[cand], BAD)
else:
    _, s, th = cand.split(":")
    k = build_kws(BAD, score=float(s), threshold=float(th), threads=1)
    mk = lambda: KwsDet(k)
# each speaker talks continuously: back-to-back clips, a different offset per speaker
clips = [a for _, a in load_clips("pos_clean")]
need = int(secs * SR) + SR
speech = []
for sp in range(N):
    buf, i = [], sp * 7
    while sum(len(b) for b in buf) < need:
        buf.append(clips[i % len(clips)][int(0.5 * SR):]); i += 1
    speech.append(np.concatenate(buf)[:need])
dets = [mk() for _ in range(N)]
locks = [threading.Lock() for _ in range(N)]
pending = [collections.deque() for _ in range(N)]
lags, done_cnt = [], [0]
pool = ThreadPoolExecutor(2)
sched = [False] * N
mu = threading.Lock()


def drain(s):
    while True:
        with mu:
            if not pending[s]:
                sched[s] = False
                return
            idx, due = pending[s].popleft()
        dets[s].feed(speech[s][idx * CHUNK:(idx + 1) * CHUNK])
        lags.append(time.perf_counter() - due)


proc = psutil.Process()
c0, w0 = sum(proc.cpu_times()[:2]), time.perf_counter()
steps = int(secs / 0.02)
for i in range(steps):
    due = w0 + (i + 1) * 0.02
    while time.perf_counter() < due:
        time.sleep(0.0005)
    for s in range(N):
        with mu:
            pending[s].append((i, due))
            if not sched[s]:
                sched[s] = True
                pool.submit(drain, s)
end_feed = time.perf_counter()
pool.shutdown(wait=True)
drained = time.perf_counter()
wall = end_feed - w0
cpu = sum(proc.cpu_times()[:2]) - c0
L = np.array(lags) * 1000
half = len(L) // 2
out = {"cand": cand, "N": N, "secs": secs, "cpu_cores_used": cpu / (drained - w0), "cpu_pct_per_speaker_of_1core": 100 * cpu / (drained - w0) / N,
       "rtf_per_speaker": cpu / (secs * N), "lag_ms_p50": pct(L, 50), "lag_ms_p90": pct(L, 90), "lag_ms_p99": pct(L, 99),
       "lag_ms_last10pct_p50": pct(L[-len(L) // 10:], 50), "backlog_drain_s": drained - end_feed,
       "rss_mb": psutil.Process().memory_info().rss / 2**20, "rss_delta_mb": (psutil.Process().memory_info().rss - rss0) / 2**20}
json.dump(out, open(f"{OUT}/{tag}/load_{cand.replace(':', '_')}_{N}.json", "w"))
print(tag, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
