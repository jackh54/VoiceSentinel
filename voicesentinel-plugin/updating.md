---
description: >-
  Updating the VoiceSentinel plugin: what happens to your config, the speech
  engine and your word list.
---

# Updating

1. Stop the server.
2. Replace the old jar in `plugins/` with the new one.
3. Start the server.

Do not use Plugman or similar to reload the plugin.

## Your config

On start the plugin upgrades `config.yml` itself. It adds the blocks that are new and leaves your values alone. Your comments and layout stay as they are. The console says `Config migrated.` when something changed.

`config-version` shows where the file is. Version 15 is the current one for 3.8.0. Each step adds:

| Version | Adds |
|---------|------|
| 12 | `local_engine` and `consent` |
| 13 | `cloud` |
| 14 | `moderation` and `storage` |
| 15 | `audio_capture.live_detection` and `live_chunk_ms` |

Do not edit `config-version`.

Older installs also move their files into folders on the first start of a newer version: `wordlist.txt`, `messages.yml` and `languages.yml` go to `plugins/VoiceSentinel/config/`, `mute_history.db` to `data/`, and the web dashboard database to `web/`.

New defaults only apply to new installs. An upgraded config keeps what it had. For example, an old config keeps `partial_match: true` unless the word list is upgraded (see below).

## The speech engine

In [On-Server Mode](on-server-mode.md), the engine is pinned to a version in each plugin release. 3.8.0 uses engine 1.1.0. If the installed engine is older, the plugin downloads the new one on the next start. You do not need to do anything.

PUBLIC and CUSTOM modes do not use the engine. A self-hosted processor is updated separately.

## Your word list

If your `config/wordlist.txt` is the unedited default from 3.6.0 or 3.7.0, the plugin replaces it with the 3.8.0 default when it starts. At the same time it changes `partial_match: true` to `partial_match: false` in `config.yml`. Your old file is kept as `config/wordlist.txt.old`. The console says `Word list: Updated to the 3.8.0 default`.

If you edited the list in any way, it is left alone and `partial_match` is not changed. A list counts as edited if the file differs from the shipped one, ignoring line endings.

To move to the new list yourself:

1. Copy any words you want to keep.
2. Delete `plugins/VoiceSentinel/config/wordlist.txt`.
3. Set `partial_match: false` in `config.yml`.
4. Restart. The new default is created.
5. Add your words back.

See [Word lists](word-lists.md).

## What is new in 3.8.0

* [Live detection](on-server-mode.md#live-detection) in On-Server Mode.
* New default word list with whole-word matching and a filled moderate tier.
* Fewer, cleaner console and chat lines: one line per mute or flag.
* The update notice shows once a day.
* Fix: two connections could open to the processor at startup.

Earlier changes you may be updating across: tiers and escalation ([Moderation and escalation](moderation-and-escalation.md)), [Storage](storage.md), [Consent](consent.md), [VoiceSentinel Cloud](voicesentinel-cloud.md) and On-Server Mode itself.
