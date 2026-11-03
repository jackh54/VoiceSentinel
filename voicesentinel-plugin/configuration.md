---
description: >-
  Plugin config.yml: license, processor mode (PUBLIC/LOCAL/CUSTOM), consent,
  moderation, mute ladder, audio capture and live detection, storage, Cloud,
  voice reports, web dashboard, Discord, telemetry and languages.
---

# Configuration

Everything here lives in **`plugins/VoiceSentinel/config.yml`**. Restart the server or run **`/voicesentinel reload`** after changes (unless noted). Word lists and messages are in **`plugins/VoiceSentinel/config/`**.

Values shown as defaults are the ones in the `config.yml` that ships with 3.8.0. An upgraded config keeps its own values. See [Updating](updating.md).

## License and processor connection

| Option | What it does |
| ------ | ------------ |
| **license-key** | Your license key. Required. The plugin will not start without it. |
| **processor\_connection\_mode** | **`PUBLIC`** (default) = shared processor pool, audio is sent to the pool. **`LOCAL`** = [On-Server Mode](on-server-mode.md), no Docker, audio stays on this machine. **`CUSTOM`** = your own processor at **`processor_websocket_url`**. |
| **processor\_websocket\_url** | WebSocket URL of **your** processor in CUSTOM mode, for example `ws://192.168.1.5:28472` or `wss://processor.example.com`. Not used in LOCAL or PUBLIC mode. |
| **processor\_discovery\_url** | Optional. Where the plugin fetches the PUBLIC pool directory. Not in the default file. Override only if directed. |
| **server\_key** | **CUSTOM:** must match **`server.server_key`** in the processor's `config.json` (16 to 256 alphanumeric). **PUBLIC:** leave blank. **LOCAL:** not used. |
| **server\_name** | Optional display name in Discord, alerts, Cloud and vendor telemetry (`server_label`). Set something readable (for example `Survival`) if you run several servers on one license. Empty = the server MOTD, then the server software name. |

{% hint style="info" %}
**CUSTOM:** copy **`server_key`** between plugin `config.yml` and processor **`config.json`** exactly. **PUBLIC** and **LOCAL** do not use it.
{% endhint %}

Example (self-hosted):

```yaml
license-key: "your-license-key"
processor_connection_mode: CUSTOM
processor_websocket_url: "ws://192.168.1.5:28472"
server_key: "MySecretKeyAtLeast16CharsLong"
```

## On-Server engine (`local_engine`)

Used only when **`processor_connection_mode: LOCAL`**. See [On-Server Mode](on-server-mode.md).

```yaml
local_engine:
  profile: ENGLISH
  threads: 2
```

| Option | Default | Description |
|--------|---------|-------------|
| **profile** | `ENGLISH` | `ENGLISH` = fast, checks every hit twice, live detection, about 1 GB free RAM. `MULTILANG` = many languages, one check, about 2.5 GB free RAM. |
| **threads** | `2` | CPU threads for the engine (1 to 16). |

## Core behaviour

| Option | Default | Description |
| ------ | ------- | ----------- |
| **enabled** | `true` | Master switch for sending audio to the processor and applying moderation. |
| **voice\_backend** | `AUTO` | `SVC` (Simple Voice Chat), `PLASMOVOICE`, or `AUTO` (use the one that is installed). Do not run both voice mods. |
| **alert\_staff** | `true` | Players with **`voicesentinel.alerts`** get in-game alerts when content is flagged. |
| **flag\_threshold** | `1` | Flags in a session before alerts, strikes and actions start. `1` = the first hit counts. An [unconfirmed](moderation-and-escalation.md#unconfirmed-hits) hit does not count. |
| **mute-duration-minutes** | `60` | Auto-mute length when the mute ladder is off. |
| **disable-listening-on-mute** | `false` | `true` = muted players cannot hear others either. `false` = they only cannot speak. |
| **health\_check\_interval** | `10000` | How often the plugin checks processor health, in milliseconds (minimum 5000). |

## Consent and privacy notice

The **`consent`** block sets whether joining counts as consent (**NOTICE**) or players accept in a menu (**GUI**). See [Consent](consent.md) for every option.

| Option | Default | Description |
|--------|---------|-------------|
| **consent.mode** | `NOTICE` | `NOTICE` or `GUI`. |
| **consent.on\_deny** | `IGNORE` | GUI mode. `IGNORE` = player stays, voice not processed. `KICK` = player is removed. |
| **consent.rules\_version** | `1` | Raise to ask everyone again. |
| **privacy-join-msg** | `true` | Show **privacy-msg** to players when they join. |
| **privacy-msg** | (text) | The notice. Supports `&` colour codes. |

## Moderation and escalation

Every hit is mild, moderate or severe. The `moderation` block sets what happens at each strike. Full explanation: [Moderation and escalation](moderation-and-escalation.md).

```yaml
moderation:
  require_confirmation: true
  escalation:
    enabled: true
    window_minutes: 10
    mild: [report, warn, mute]
    moderate: [warn, mute]
    severe: [mute]
  warn_message: "&cPlease keep voice chat respectful."
  appeal_message: "&7To appeal, contact staff."
  appeal_url: ""  # optional; makes appeal_message a clickable link
  network_strikes: true
```

| Option | Default | Description |
|--------|---------|-------------|
| **require\_confirmation** | `true` | A hit that Double Check rejects is shown to staff as unconfirmed and never warns or mutes. No effect on processors without Double Check. |
| **escalation.enabled** | `true` | `false` = always use the first step. |
| **escalation.window\_minutes** | `10` | Strikes older than this do not count. |
| **escalation.mild / moderate / severe** | see above | Steps for the 1st, 2nd, 3rd strike. The last step repeats. Steps: `report`, `warn`, `mute`. |
| **warn\_message** | `&cPlease keep voice chat respectful.` | Sent on a `warn` step. |
| **appeal\_message** | `&7To appeal, contact staff.` | Sent after every automatic mute, unless Cloud gives the player its own appeal link. |
| **appeal\_url** | `""` | Makes **appeal\_message** a clickable link. |
| **network\_strikes** | `true` | With MySQL/MariaDB [storage](storage.md), count strikes across every server on the database. |

## Mute ladder (LuckPerms)

Escalating durations for **repeat** auto-mutes. Requires **LuckPerms**.

```yaml
mute_ladder:
  enabled: true
  tiers:
    1: "5m"
    2: "15m"
    3: "30m"
    4: "1h"
    5: "6h"
  max_tier_duration: "24h"
  reset_after_days: 30  # 0 = never reset offense count
```

Manual **`/voicesentinel vcmute`** mutes ignore the ladder. **`mute-duration-minutes`** applies when the ladder is off.

## Word lists and matching

* **`config/wordlist.txt`** holds per-language **PROFANITY** (mild), **MODERATE** and **MUTE** (severe) sections. See [Word lists](word-lists.md).
* **`profanity_words`** and **`mute-list`** in `config.yml` are extra words. They are **added** to the default-language lists from `wordlist.txt`. Empty `[]` means only the file is used. The old key **`profanity_list`** is accepted like **`profanity_words`**.

After edits, run **`/voicesentinel reload`**.

| Option | Default | Description |
|--------|---------|-------------|
| **case\_sensitive** | `false` | Exact letter casing when matching words. |
| **partial\_match** | `false` | `false` = whole words only (`crap` does not flag `scrap`). `true` = a word also matches inside longer words. This catches more forms but flags innocent words. |

{% hint style="info" %}
New installs get `partial_match: false`. A config upgraded from an older version keeps `true` unless its word list is upgraded. See [Updating](updating.md).
{% endhint %}

## Audio capture and live detection

Controls when the plugin sends audio to the processor. Tuning these affects **latency** against **fragment count**.

```yaml
audio_capture:
  silence_threshold_ms: 500
  min_audio_duration_ms: 500
  # false = skip Simple Voice Chat / PlasmoVoice group chats (proximity still monitored).
  monitor_groups: true
  # true = with the On-Server engine (v1.1+), stream audio while a player talks so a listed word is caught mid-sentence.
  live_detection: true
  # How often (ms) live audio is sent while talking (100-1000).
  live_chunk_ms: 200
```

| Option | Default | Description |
|--------|---------|-------------|
| **silence\_threshold\_ms** | `500` | Ms of silence after speech before sending. **Lower** = faster response, more fragments. |
| **min\_audio\_duration\_ms** | `500` | Minimum utterance length (ms) before sending. **Lower** = shorter clips allowed (may add noise). |
| **monitor\_groups** | `true` | `false` = skip voice chat group chats. Proximity chat is still monitored. |
| **live\_detection** | `true` | With the On-Server engine (ENGLISH profile, engine 1.1 or newer), stream audio while a player talks. See [On-Server Mode](on-server-mode.md#live-detection). |
| **live\_chunk\_ms** | `200` | How often live audio is sent while talking (100 to 1000). |

See [Performance and latency](#performance-and-latency) below and [Processor configuration](../voicesentinel-processor/configuration.md).

## Storage

Where mute history, strikes and voice reports are kept. SQLite (default) or MySQL/MariaDB. Restart after changing it. See [Storage](storage.md).

```yaml
storage:
  type: SQLITE
  host: "localhost"
  port: 3306
  database: "voicesentinel"
  username: ""
  password: ""
  ssl: false  # true = encrypt the connection (certificate is not verified)
```

## VoiceSentinel Cloud

Incident history, a Discord bot with buttons, remote unmute and player appeals. Off until you turn it on. See [VoiceSentinel Cloud](voicesentinel-cloud.md).

```yaml
cloud:
  enabled: false
  upload_clips: true
  appeal_links: true
```

## Transcript buffer

Rolling **in-memory** transcripts per player. Used for the **web dashboard** moderation view and for **`/reportvoice`**. Independent of **`voice_report.enabled`**.

```yaml
transcript_buffer:
  enabled: true
  retention_seconds: 43200
  max_lines_per_player: 500
```

See [Voice reports & transcript buffer](voice-reports.md).

## Voice report (`voice_report`)

Player **`/reportvoice`** and staff **`/viewreport`** / **`/reportinbox`**. Key options:

| Option | Description |
| ------ | ----------- |
| **enabled** | Turns the player-facing report flow on or off (default `false`). |
| **interface** | **`command`** = `/reportvoice [player] [minutes]`. **`gui`** = chest menu to pick player and time range. |
| **default\_report\_window\_seconds** / **min\_** / **max\_** | Allowed time window for a report (clamped). |
| **report\_window\_options\_seconds** | Menu button lengths (seconds), clamped to min/max. |
| **processor\_evidence\_url** | HTTP base for processor evidence (for example `https://processor.example.com`). Empty = derived from the WebSocket URL. |
| **webhook\_url** | Optional Discord webhook for the same summary staff see in-game. |
| **staff\_chat\_full\_transcript** | `true` = staff also get the full plain text in chat (can be spammy). |
| **cooldown\_seconds** | Cooldown between reports, in seconds (default `30`). |
| **max\_concurrent\_reports** | Most reports handled at the same time (default `32`). |

## Web dashboard

Browser operator console. See [Web dashboard](web-dashboard.md).

```yaml
web_dashboard:
  enabled: false
  bind_address: "127.0.0.1"
  port: 8124
  path_prefix: "/voicesentinel/web"
  session_minutes: 480
  max_login_attempts_per_minute: 20
  setup_token_minutes: 15
  ip_allowlist: []
  live_monitoring:
    enabled: true
    live_transcript: true
    flagged_feed: true
    player_history: true
```

**live\_monitoring** controls the Live monitoring tab. Set any panel to `false` to hide it.

## Network sync

For several servers under one dashboard. Default is off.

| Option | Description |
|--------|-------------|
| **enabled** | `false` by default. |
| **role** | `MAIN` = hosts the dashboard. `CLIENT` = remote shard. `DISABLED` = alerts only, if **cross\_server\_notifications** is on. |
| **network\_id** | Shared by every server in one network. |
| **server\_id** | Unique per server. **Must be different on each server.** Also used to tell servers apart for [network strikes](moderation-and-escalation.md#network-strikes). |
| **main\_dashboard\_url** | URL of the MAIN dashboard. Same on every server. |
| **api\_key** | Shared key between servers. |
| **redis\_uri** | For example `redis://:pass@host:6379/0`. |
| **sync\_wordlists\_from\_main** | `true` = MAIN pushes word lists to CLIENT servers. |
| **max\_events\_per\_player** | Cap for synced transcript events per player (`3000`). |
| **cross\_server\_notifications** | `true` = staff alerts reach every networked server. |

## Custom commands

Console commands when someone is **auto-muted** or when a hit does not mute. Placeholders: **`{player}`**, **`{words}`**, **`{transcript}`**. See [Custom commands](custom-commands.md) for conditions, **Folia/Paper** thread behaviour and **command whitelist** notes.

```yaml
custom_commands:
  mute_commands: ["say {player} has been flagged for inappropriate language!"]
  profanity_commands: []
  keyword_commands: {}
  auto_mute_builtin:
    luckperms_voice_mute: true
    player_message: true
    staff_alert: true
    discord_webhook: true
    mute_event_log: true
    console_log: true
```

* **keyword\_commands**: per-word command lists. They run **first** when a mapped word is detected.
* **mute\_commands** / **profanity\_commands**: global fallbacks when there is **no** keyword mapping for that word.
* **auto\_mute\_builtin**: turns parts of the built-in auto-mute on or off. `luckperms_voice_mute: false` lets you mute with your own **mute\_commands** instead of LuckPerms.

## Discord (flagged content)

Under **`discord`**: **enabled**, **webhook\_url**, **username**, **avatar\_url**, **embed\_color**, **ping\_role\_id**, **timeout\_seconds**, **retry\_attempts**, **include\_transcript**, **include\_server\_info**, **include\_audio** (needs **`response.include_audio: true`** on the processor), plus **customization** titles and fields.

Test: **`/voicesentinel discordtest`** with **`all`**, **`flag`** or **`mute`**. See [Discord webhooks](discord-webhooks.md).

## Discord (mute / unmute)

Under **`discord_mute`**: separate webhook for mute and unmute events. **include\_transcript** can be **`flagged`**, **`muted`**, **`both`** or **`none`**. **include\_audio** also needs the processor **`response.include_audio`** flag.

## Telemetry (vendor analytics)

Every **15 minutes** (aligned with license revalidation) the plugin may send usage counters to the license server.

**Not sent:** transcripts, player names, flagged words.

**Sent:** counters since the last report (utterances, transcripts, flags, mutes, staff alerts, reports, reconnects, dropped audio), latency percentiles, the online player count, plugin and Minecraft version, server software, the processor mode, `server_name` and `network_sync.server_id`, the server IP, and a per-server hardware id (`hwid`). Reports are keyed by a SHA-256 fingerprint of your license key.

```yaml
telemetry:
  enabled: true
```

| Option | Default | Description |
|--------|---------|-------------|
| **telemetry.enabled** | `true` | Set **`false`** to opt out. |

The URL, retry count and backoff are **fixed in the plugin** (license API host + `/api/metrics`, 2 retries, 2 s then 5 s). Telemetry is best-effort. A failure never affects moderation, license validation or console warnings.

If you run a **network** of servers, set **`network_sync.server_id`** (and **`server_name`**) so vendor dashboards can tell shards apart under one license.

## Languages

```yaml
languages:
  enabled: ["en"]
  default: "en"
```

## Placeholder API

**placeholder\_api**: integration hook. Leave **`enabled: false`** unless you use that feature.

## Performance and latency

**Plugin (`config.yml`):**

```yaml
audio_capture:
  silence_threshold_ms: 300   # send sooner after player stops talking
  min_audio_duration_ms: 400  # allow shorter utterances
```

In On-Server Mode, live detection already catches listed words before the sentence ends. See [On-Server Mode](on-server-mode.md).

**Processor (`config.json`):** see [Processor configuration](../voicesentinel-processor/configuration.md). The keys are `transcription.beam_size`, `transcription.use_memory_input` and `processing.worker_count`.

`final_transcript` responses may include **`processing_time_ms`**, **`stt_time_ms`** and **`moderation_time_ms`** for monitoring.

## Other files

| File | Purpose |
| ---- | ------- |
| **config/wordlist.txt** | Word lists ([Word lists](word-lists.md)). |
| **config/messages.yml** | In-game and alert text (MiniMessage). |
| **config/languages.yml** | Language settings. |
| **consent.properties** | Players' answers in GUI [consent](consent.md) mode. |
| **data/mute\_history.db**, **data/voice\_reports.db** | SQLite [storage](storage.md) (default). |
| **web/web\_auth.db** | SQLite store for web dashboard accounts (created when used). |
| **engine/** | Downloaded [On-Server](on-server-mode.md) engine (LOCAL mode). |

Superusers can edit **word lists** from the **web dashboard**. Changes are written back to **`config/wordlist.txt`**.
