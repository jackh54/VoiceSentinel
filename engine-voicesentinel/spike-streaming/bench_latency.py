"""Detection latency + recall + Double Check on clips with a watch-word in the middle of a sentence.
usage: python bench_latency.py <tag> <cand> <set,set,...>   (cand: asr:zip2023 | kws:1.0:0.25 | today)"""
import sys, time, json, os
from common import *

tag, cand, sets = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
SIL_MS, POLL_MS = 0.5, 0.05    # plugin silence threshold, mean 100 ms poll granularity
os.makedirs(f"{OUT}/{tag}", exist_ok=True)
P = standin_words()
watch = sorted(set(P) | set(BAD))
recs = {}
conf = Confirm()
if cand == "today":
    prim = conf.p
for sname in sets:
    rows, fa_all = [], []
    t_wall = time.time(); audio_s = cpu_s = 0.0
    mk = None if cand == "today" else make_detector(cand, watch, recs)
    for it, audio in load_clips(sname):
        occ = [o for o in it.get("occ", []) if 0.2 <= o.get("pos", 0.5) <= 0.8]   # word in the middle of the sentence
        if cand == "today":
            sp = audio[int(0.5 * SR):int(it["speech_end"] * SR)]       # only speech frames are sent
            txt, stt = Confirm._run(conf.p, sp)
            found = find_words(txt, watch)
            for o in occ:
                if o["w"] in found:
                    word = o["w"]
                    mt, mdur = Confirm._run(conf.m, sp)
                    ok = bool(find_words(mt, [word]))
                    lat = (it["speech_end"] - o["e"]) + SIL_MS + POLL_MS + stt + mdur
                    rows.append({"w": word, "hit": True, "lat": lat, "stt_s": stt, "confirm_s": mdur, "p_ok": True, "m_ok": ok,
                                 "tail_s": it["speech_end"] - o["e"]})
                else:
                    rows.append({"w": o["w"], "hit": False})
            audio_s += it["dur"]; cpu_s += stt
            continue
        det = mk()
        events, cpu = simulate(det, audio)
        res, fa = score_clip(events, occ, it["text"])
        cpu_s += cpu; audio_s += it["dur"]
        for r in res:
            if r["hit"]:
                r.update(conf.check(audio, r["t"], r["w"]))
            r["id"] = it["id"]
            rows.append(r)
        for w, t in fa:
            d = conf.check(audio, t, w); d.update(w=w, id=it["id"]); fa_all.append(d)
    hits = [r for r in rows if r["hit"]]
    lat = [r["lat"] for r in hits]
    out = {"cand": cand, "set": sname, "n_occ": len(rows), "n_hit": len(hits), "recall": len(hits) / max(1, len(rows)),
           "lat_p50": pct(lat, 50), "lat_p90": pct(lat, 90), "lat_min": min(lat) if lat else None,
           "rtf": cpu_s / max(1e-9, audio_s), "audio_s": audio_s, "fa_events": len(fa_all),
           "dc_pk_agree": sum(r["p_ok"] for r in hits) / max(1, len(hits)),
           "dc_ms_agree": sum(r["m_ok"] for r in hits) / max(1, len(hits)),
           "dc_p_ms_p50": pct([r["p_ms"] for r in hits if "p_ms" in r], 50),
           "dc_m_ms_p50": pct([r["m_ms"] for r in hits if "m_ms" in r], 50),
           "rows": rows, "fa": fa_all, "wall_s": time.time() - t_wall}
    json.dump(out, open(f"{OUT}/{tag}/lat_{cand.replace(':', '_')}_{sname}.json", "w"))
    print(tag, cand, sname, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k not in ("rows", "fa")}, flush=True)
