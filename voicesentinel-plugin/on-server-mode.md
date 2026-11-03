---
description: >-
  On-Server Mode runs the speech engine on your Minecraft server. No Docker, no
  separate processor. Covers Double Check, live detection, RAM and privacy.
---

# On-Server Mode

On-Server Mode (`processor_connection_mode: LOCAL`) runs the speech engine on the same machine as your Minecraft server. The plugin downloads the engine, starts it, and talks to it over `127.0.0.1`. You do not need Docker, Python or a separate processor.

## Turn it on

```yaml
processor_connection_mode: LOCAL

local_engine:
  profile: ENGLISH
  threads: 2
```

Restart the server. `/voicesentinel reload` also starts or stops the engine to match the mode.

On the first start the plugin downloads the engine and the speech models from GitHub (release `engine-v1.1.0`). Each file is checked against a built-in SHA-256 hash. Files go to `plugins/VoiceSentinel/engine/`. Later starts reuse them.

The console shows `Speech engine: Starting`, then `Speech engine: Ready`. Voice moderation starts when the engine is ready.

`server_key` is not used in this mode.

## Profiles

| Profile | Languages | Free RAM | Double Check | Live detection |
|---------|-----------|----------|--------------|----------------|
| **ENGLISH** (default) | English | about 1 GB | Yes, every hit | Yes |
| **MULTILANG** | Many | about 2.5 GB | No, one check | No |

`local_engine.threads` is the number of CPU threads the engine uses (1 to 16, default 2).

The RAM is on top of what your server already uses. The engine runs as its own process, outside the Minecraft JVM.

## Requirements

* Java 21 on the server (the engine starts with the same Java).
* Linux (x64 or aarch64), macOS (Intel or Apple Silicon) or Windows (x64).
* Internet access to GitHub on the first start.
* Free disk space for the engine and models.
* Enough free RAM for the profile you chose.

## Double Check

In the ENGLISH profile a second speech model checks every hit. The result is one of:

* **confirmed**: both models heard the word.
* **rejected**: the second model did not hear it.
* **unchecked**: no second check was made for this hit.

A rejected hit is an **unconfirmed** hit. Staff and [VoiceSentinel Cloud](voicesentinel-cloud.md) still see it, marked `[unconfirmed]`. It never warns, mutes, counts a strike or runs commands.

Turn this off with `moderation.require_confirmation: false`. Then rejected hits are acted on like any other. See [Moderation and escalation](moderation-and-escalation.md).

## Live detection

Without live detection the plugin waits until a player stops talking, then sends the whole sentence. With live detection (`audio_capture.live_detection: true`, the default) the plugin sends short pieces of audio while the player is still talking. A listed word is caught in the middle of the sentence.

* Every live hit goes through Double Check before anything happens.
* When a live hit mutes a player, the rest of that sentence is dropped.
* The normal end-of-sentence check still runs as a backstop for words the live check missed. A word that was already handled live is not punished twice.
* `audio_capture.live_chunk_ms` is how often audio is sent while talking (100 to 1000, default 200).

In our tests a mute word was caught about 0.7 seconds after it was said, on a modern CPU. Older or slower CPUs take longer.

Live detection needs the ENGLISH profile and engine 1.1 or newer. With MULTILANG, PUBLIC or CUSTOM it has no effect and the plugin works as before. The startup lines list `live detection` when it is on.

## Privacy

In On-Server Mode the audio never leaves your server. It goes to the engine on `127.0.0.1` and is converted to text there. The engine port is random and only listens on the local machine.

The plugin still contacts these hosts:

* GitHub, to download the engine.
* The PandaScript license server, to check your license and for [telemetry](configuration.md#telemetry-vendor-analytics).
* `cloud.pandascript.dev`, only if you turn on [VoiceSentinel Cloud](voicesentinel-cloud.md).

Players are told about moderation by the join notice or the [consent menu](consent.md).

## Files

| Path | What it is |
|------|------------|
| `plugins/VoiceSentinel/engine/` | Downloaded engine and models |
| `plugins/VoiceSentinel/engine/engine.log` | Engine output. Read this when the engine fails |

## Problems

See [Troubleshooting](troubleshooting.md#on-server-mode).
