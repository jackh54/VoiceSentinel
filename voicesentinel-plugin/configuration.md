---
description: Plugin config.yml – license, processor URL, server key, behaviour, word lists, and optional overrides.
---

# Configuration

All of this lives in `plugins/VoiceSentinel/config.yml`. Restart the server or run `/voicesentinel reload` to apply changes.

## Must-have

| Option | What it does |
|--------|----------------|
| **license-key** | Your license key. Required or the plugin won’t start. |
| **processor_websocket_url** | Where the processor is. Example: `ws://localhost:28472` if it’s on the same machine, or `ws://processor.yourdomain.com` if it’s on another server. Use `wss://` if you put the processor behind HTTPS. |
| **server_key** | Secret that must match the processor’s `server.server_key` (in its `config.json`). 16–256 letters/numbers. Same key in both places or the processor will reject the connection. |

{% hint style="info" %}
**server_key** must be identical in the plugin’s config.yml and the processor’s config.json. Copy-paste the value to avoid typos.
{% endhint %}

Example:

```yaml
license-key: "your-license-key"
processor_websocket_url: "ws://192.168.1.5:28472"
server_key: "MySecretKeyAtLeast16Chars"
```

## Behaviour

- **enabled** – `true`/`false`. Master switch for moderation.
- **alert_staff** – If `true`, staff with `voicesentinel.alerts` get in-game alerts when something is flagged.
- **flag_threshold** – How many times someone has to be flagged (in that “session”) before auto-mute or custom commands run. Default `1` = first offence can trigger it.
- **mute-duration-minutes** – Default length of an auto-mute (e.g. `60` = 1 hour).
- **disable-listening-on-mute** – If `true`, muted players can’t hear others either (full mute). If `false`, they just can’t talk.

## Word lists

By default the plugin uses `wordlist.txt` in the same folder (and sends that to the processor). You can override that per-server in config:

```yaml
# Use only these words (ignores wordlist.txt for the default language)
profanity_words:
  - badword1
  - badword2
mute-list:
  - muteword1
```

Leave both as `[]` to use `wordlist.txt` (and any per-language lists) instead.

## Custom commands

When someone gets auto-muted or only profanity is detected, you can run console commands. Placeholders: `{player}`, `{words}`, `{transcript}`.

Example:

```yaml
custom_commands:
  mute_commands:
    - "say [Voice] {player} was auto-muted for: {words}"
  profanity_commands:
    - "say [Voice] Profanity from {player}: {words}"
```

Commands run as the server console. See [Custom commands](custom-commands.md) for more.

## Discord

Set **discord.enabled** to `true` and **discord.webhook_url** to your webhook URL to send flagged-content alerts to a channel. Use **discord_mute** for mute/unmute notifications (separate webhook if you want). Test with `/voicesentinel discordtest all` (or `flag` / `mute`).

## Privacy message

**privacy-join-msg** and **privacy-msg** control the notice players see when they join. Turn it off or edit the text to match your server’s policy.

## Other files

- **wordlist.txt** – Profanity and mute words per language (see [Word lists](word-lists.md)).
- **messages.yml** – In-game and alert text (MiniMessage).
- **languages.yml** – Language options if you use them.
