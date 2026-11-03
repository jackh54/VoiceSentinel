"""Turns results/<tag>/*.json into the markdown tables used in the report. usage: python aggregate.py <tag> [fa_tag]"""
import glob, json, sys, os, collections
import numpy as np, jiwer
from common import OUT, pct

tag = sys.argv[1]
fa_tag = sys.argv[2] if len(sys.argv) > 2 else tag
L = lambda p: [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/{p}"))]
NAT = ["pos_clean", "pos_other"]
ms = lambda x: "-" if x is None or x != x else f"{x * 1000:.0f}"


def lat_rows():
    by = collections.defaultdict(dict)
    for r in L(f"{tag}/lat_*.json"):
        by[r["cand"]][r["set"]] = r
    return by


def pool(rs, key="lat"):
    rows = [x for r in rs for x in r["rows"]]
    hits = [x for x in rows if x["hit"]]
    return rows, hits


print("## WER (73 LibriSpeech dev-clean clips, Opus 24k round trip)\n")
print("| model | WER | word accuracy | RTF (1 thread-set as run) |\n|---|---|---|---|")
for f in L(f"{tag}/wer_*.json"):
    for k, v in f.items():
        print(f"| {k} | {v['wer'] * 100:.1f}% | {100 - v['wer'] * 100:.1f}% | {v['rtf']:.3f} |")

by = lat_rows()
print("\n## Detection latency, end of word to hit (ms), watch word mid-sentence\n")
print("| candidate | natural clean+other: recall | p50 | p90 | babble 10 dB: recall / p50 | hiss 15 dB: recall / p50 | TTS profanity: recall / p50 |\n|---|---|---|---|---|---|---|")
for cand, sets in by.items():
    nat = [sets[s] for s in NAT if s in sets]
    rows, hits = pool(nat)
    lat = [h["lat"] for h in hits]
    cell = lambda s: (f"{sets[s]['recall'] * 100:.0f}% / {ms(sets[s]['lat_p50'])}" if s in sets else "-")
    print(f"| {cand} | {len(hits) / max(1, len(rows)) * 100:.0f}% ({len(hits)}/{len(rows)}) | {ms(pct(lat, 50))} | {ms(pct(lat, 90))} | {cell('pos_babble10')} | {cell('pos_white15')} | {cell('tts')} |")

print("\n## Double Check on true hits (natural clips): fraction kept, and confirm time for a 3 s window\n")
print("| candidate | Parakeet agrees | Moonshine agrees | both agree | Parakeet ms p50/p90 | Moonshine ms p50/p90 |\n|---|---|---|---|---|---|")
for cand, sets in by.items():
    if cand == "today":
        continue
    rows, hits = pool([sets[s] for s in NAT if s in sets])
    n = max(1, len(hits))
    print(f"| {cand} | {sum(h['p_ok'] for h in hits) / n * 100:.0f}% | {sum(h['m_ok'] for h in hits) / n * 100:.0f}% | "
          f"{sum(h['p_ok'] and h['m_ok'] for h in hits) / n * 100:.0f}% | {pct([h['p_ms'] for h in hits], 50):.0f}/{pct([h['p_ms'] for h in hits], 90):.0f} | "
          f"{pct([h['m_ms'] for h in hits], 50):.0f}/{pct([h['m_ms'] for h in hits], 90):.0f} |")

print("\n## Today's pipeline (500 ms silence wait + 50 ms poll + Parakeet on whole utterance + Moonshine confirm)\n")
if "today" in by:
    for s, r in by["today"].items():
        hits = [h for h in r["rows"] if h["hit"]]
        print(f"- {s}: recall {r['recall'] * 100:.0f}% ({r['n_hit']}/{r['n_occ']}), latency p50 {ms(r['lat_p50'])} ms, p90 {ms(r['lat_p90'])} ms; "
              f"word-to-end-of-speech p50 {ms(pct([h['tail_s'] for h in hits], 50))} ms; Parakeet {ms(pct([h['stt_s'] for h in hits], 50))} ms + Moonshine {ms(pct([h['confirm_s'] for h in hits], 50))} ms")

print("\n## False alarms per hour of word-free speech (real profanity list, 38 entries, whole-word match)\n")
fa = collections.defaultdict(lambda: collections.defaultdict(list))
for r in L(f"{fa_tag}/fa_*.json"):
    fa[r["cand"]][r["set"]].append(r)
print("| candidate | speech hours | FA events | FA/hour | after Parakeet check | after Moonshine check | after both | WER clean | WER other |\n|---|---|---|---|---|---|---|---|---|")
for cand, sets in fa.items():
    hrs = sum(r["audio_s"] for s in ("fa_clean", "fa_other") for r in sets.get(s, [])) / 3600
    ev = [e for s in ("fa_clean", "fa_other") for r in sets.get(s, []) for e in r["fa"]]
    n = len(ev)
    pk = sum(e["p_ok"] for e in ev); mk = sum(e["m_ok"] for e in ev); both = sum(e["p_ok"] and e["m_ok"] for e in ev)
    wers = []
    for s in ("fa_clean", "fa_other"):
        refs = [x for r in sets.get(s, []) for x in r["refs"]]; hyps = [x for r in sets.get(s, []) for x in r["hyps"]]
        wers.append(f"{jiwer.wer(refs, hyps) * 100:.1f}%" if refs else "-")
    print(f"| {cand} | {hrs:.2f} | {n} | {n / max(hrs, 1e-9):.1f} | {pk / max(hrs, 1e-9):.1f}/h | {mk / max(hrs, 1e-9):.1f}/h | {both / max(hrs, 1e-9):.1f}/h | {wers[0]} | {wers[1]} |")
print("\n## Noise-only audio (15 min: room tone, hiss, rumble+clicks): words emitted by the streaming ASR\n")
for cand, sets in fa.items():
    if "noise" in sets:
        print(f"- {cand}: {sum(r['noise_words'] for r in sets['noise'])} words, {sum(len(r['fa']) for r in sets['noise'])} watch-list hits")

print("\n## Load: speakers talking non-stop, 2 worker threads, real time\n")
print("| candidate | speakers | CPU cores used (of 2) | CPU % of one core per speaker | RTF per speaker | chunk lag p50 / p90 / p99 (ms) | lag in last 10% (ms) | RSS MB |\n|---|---|---|---|---|---|---|---|")
for r in L(f"{tag}/load_*.json"):
    print(f"| {r['cand']} | {r['N']} | {r['cpu_cores_used']:.2f} | {r['cpu_pct_per_speaker_of_1core']:.1f}% | {r['rtf_per_speaker']:.3f} | "
          f"{r['lag_ms_p50']:.0f} / {r['lag_ms_p90']:.0f} / {r['lag_ms_p99']:.0f} | {r['lag_ms_last10pct_p50']:.0f} | {r['rss_mb']:.0f} |")

print("\n## TTS profanity recall by word (per candidate, p50 latency ms)\n")
words = None
tab = {}
for cand, sets in by.items():
    if "tts" not in sets or cand == "today":
        continue
    d = collections.defaultdict(list)
    for x in sets["tts"]["rows"]:
        d[x["w"]].append(x)
    tab[cand] = {w: (sum(h["hit"] for h in v) / len(v)) for w, v in d.items()}
    words = sorted(d)
if tab:
    print("| word | " + " | ".join(tab) + " |\n|---|" + "---|" * len(tab))
    for w in words:
        print(f"| {w} | " + " | ".join(f"{tab[c].get(w, 0) * 100:.0f}%" for c in tab) + " |")
