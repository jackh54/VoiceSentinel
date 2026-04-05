---
description: >-
  Plugin config.yml – license, processor connection (CUSTOM/PUBLIC), behaviour,
  mute ladder, transcript buffer, voice reports, web dashboard, Discord, and
  languages.
---

# Configuration

Everything here lives in **`plugins/VoiceSentinel/config.yml`**. Restart the server or run **`/voicesentinel reload`** after changes (unless noted).

## License and processor connection

| Option                          | What it does                                                                                                                                                                                                                                                                           |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **license-key**                 | Your license key. Required — the plugin will not start without it.                                                                                                                                                                                                                     |
| **processor\_connection\_mode** | **`CUSTOM`** (default) = you run your own processor. **`PUBLIC`** = plugin uses the **shared processor pool**. Keep the plugin updated for PUBLIC mode.                                                                                                                                |
| **processor\_websocket\_url**   | WebSocket URL of **your** processor when using CUSTOM, e.g. `ws://192.168.1.5:28472` or `wss://processor.example.com`. In PUBLIC mode the plugin still needs a valid URL shape; the effective endpoint comes from directory discovery.                                                 |
| **processor\_discovery\_url**   | Optional. Where the plugin fetches the PUBLIC pool directory. Default: PandaScript CDN directory URL. Override only if directed.                                                                                                                                                       |
| **server\_key**                 | **CUSTOM:** Must match **`server.server_key`** in the processor’s `config.json` (16–256 alphanumeric). **PUBLIC:** If left default/empty, the plugin uses an internal placeholder — the pool uses its own auth; you do **not** use your personal `server_key` for pool WebSocket auth. |
| **server\_name**                | Optional display name sent with telemetry (can show in dashboards/alerts).                                                                                                                                                                                                             |

{% hint style="info" %}
**CUSTOM:** Copy **`server_key`** between plugin `config.yml` and processor **`config.json`** exactly. **PUBLIC:** Self-hosted **`server_key`** is not used for pool WebSocket authentication the same way; follow current license/pool instructions.
{% endhint %}

Example (self-hosted):

```yaml
license-key: "your-license-key"
processor_connection_mode: CUSTOM
processor_websocket_url: "ws://192.168.1.5:28472"
server_key: "MySecretKeyAtLeast16CharsLong"
```

## Core behaviour

| Option                        | Description                                                                                              |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| **enabled**                   | `true` / `false`. Master switch for sending audio to the processor and applying moderation.              |
| **alert\_staff**              | If `true`, players with **`voicesentinel.alerts`** get in-game alerts when content is flagged.           |
| **flag\_threshold**           | How many flags in a session before auto-mute / custom commands run. Default `1` = first hit can trigger. |
| **mute-duration-minutes**     | Default auto-mute length when **not** using the mute ladder (e.g. `60` = 1 hour).                        |
| **disable-listening-on-mute** | If `true`, muted players cannot hear others (full mute). If `false`, they only cannot transmit.          |
| **health\_check\_interval**   | How often the plugin checks processor health (milliseconds). Default `10000`.                            |

## Mute ladder (LuckPerms)

Optional escalating durations for **repeat** auto-mutes. Requires **LuckPerms**.

```yaml
mute_ladder:
  enabled: false
  tiers:
    1: "5m"
    2: "15m"
    3: "30m"
    4: "1h"
    5: "6h"
  max_tier_duration: "24h"
  reset_after_days: 30
```

Manual mutes with an explicit duration **bypass** the ladder. **`mute-duration-minutes`** applies when the ladder is off or for non-ladder paths.

## Word lists

* **`wordlist.txt`** in `plugins/VoiceSentinel/` — per-language **PROFANITY** / **MUTE** sections (see [Word lists](word-lists.md)).
* **`profanity_words`** / **`mute-list`** in YAML — if non-empty, they **replace** the default-language lists sent to the processor for those categories. Empty `[]` means “use **wordlist.txt**”. Legacy alias: **`profanity_list`** is accepted like **`profanity_words`**.

After edits, run **`/voicesentinel reload`**.

Optional: **`case_sensitive`** — if `true`, word-list matching uses case-sensitive checks (default `false`).

## Transcript buffer

Rolling **in-memory** transcripts per player — used for the **web dashboard** moderation view and for **`/reportvoice`** data. Independent of whether **`voice_report.enabled`** is true.

```yaml
transcript_buffer:
  enabled: true
  retention_seconds: 43200
  max_lines_per_player: 500
```

See [Voice reports & transcript buffer](voice-reports.md).

## Voice report (`voice_report`)

Player **`/reportvoice`** and staff **`/viewreport`** / **`/reportinbox`**. Key options:

| Option                                                       | Description                                                                                                      |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **enabled**                                                  | Turns the player-facing report flow on or off.                                                                   |
| **interface**                                                | **`command`** = `/reportvoice [player] [minutes]`. **`gui`** = chest UI to pick player and time range.           |
| **default\_report\_window\_seconds** / **min\_** / **max\_** | Allowed time window for a report (clamped).                                                                      |
| **report\_window\_options\_seconds**                         | GUI button lengths (seconds), clamped to min/max.                                                                |
| **processor\_evidence\_url**                                 | HTTP base for processor evidence (e.g. `https://processor.example.com`). Empty = derived from the WebSocket URL. |
| **webhook\_url**                                             | Optional Discord webhook for the same summary staff see in-game.                                                 |
| **staff\_chat\_full\_transcript**                            | If `true`, staff also get the full plain text in chat (can be spammy).                                           |

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
```

## Custom commands

Console commands when someone is auto-muted or only profanity is flagged. Placeholders: **`{player}`**, **`{words}`**, **`{transcript}`**.

```yaml
custom_commands:
  mute_commands: []
  profanity_commands: []
```

See [Custom commands](custom-commands.md).

## Discord (flagged content)

Under **`discord`**: **enabled**, **webhook\_url**, **username**, **avatar\_url**, **embed\_color**, **ping\_role\_id**, **timeout\_seconds**, **retry\_attempts**, **include\_transcript**, **include\_server\_info**, **include\_audio** (requires **`response.include_audio: true`** on the processor), plus **customization** titles/fields.

Test: **`/voicesentinel discordtest`** with **`all`**, **`flag`**, or **`mute`**.

## Discord (mute / unmute)

Under **`discord_mute`**: separate webhook for mute/unmute events. **include\_transcript** can be **`flagged`**, **`muted`**, **`both`**, or **`none`**. **include\_audio** also requires the processor **`response.include_audio`** flag.

## Privacy join message

**privacy-join-msg** and **privacy-msg** control the notice shown when players join.

## Languages

```yaml
languages:
  enabled: ["en"]
  default: "en"
```

## Placeholder API

**placeholder\_api** – integration hook; leave **`enabled: false`** unless you use that feature.

## Other files

| File                 | Purpose                                                      |
| -------------------- | ------------------------------------------------------------ |
| **wordlist.txt**     | Shared profanity/mute lists ([Word lists](word-lists.md)).   |
| **messages.yml**     | In-game and alert text (MiniMessage).                        |
| **languages.yml**    | Extra language configuration when used.                      |
| **web/web\_auth.db** | SQLite store for web dashboard accounts (created when used). |

Superusers can edit **word lists** from the **web dashboard**; changes are written back to **`wordlist.txt`**.
