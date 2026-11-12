---
description: >-
  How VoiceSentinel decides what to do with a flagged word: tiers, escalation
  steps, strike window, unconfirmed hits, network strikes, the mute ladder and
  the appeal message.
---

# Moderation and escalation

Every flagged hit has a tier. Each player collects strikes per tier. The strike number decides what happens: report, warn or mute.

## Tiers

The tier comes from the section of [wordlist.txt](word-lists.md) the word is in.

| Tier | Section | Meaning |
|------|---------|---------|
| **mild** | `[XX-PROFANITY]` | Swearing that is not aimed at anyone |
| **moderate** | `[XX-MODERATE]` (optional) | Insults, harassment and threats |
| **severe** | `[XX-MUTE]` | Slurs, hate and telling someone to harm themselves |

If a hit has words from more than one tier, the highest tier wins (severe, then moderate, then mild).

## Escalation

This is the default `moderation` block:

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
  appeal_url: ""
  network_strikes: true
```

Each list is a ladder of steps. A player's first strike in a tier uses step 1, the second uses step 2, and so on. The last step repeats.

| Step | What it does |
|------|--------------|
| **report** | Staff alerts, Discord webhooks and [custom commands](custom-commands.md) only. Nothing is sent to the player. |
| **warn** | Sends `warn_message` to the player. Staff alerts and commands run too. |
| **mute** | Voice-mutes the player. The length comes from the [mute ladder](#mute-ladder). Needs LuckPerms only if `mutes.enforcement` is `LUCKPERMS`. |

Strikes are counted per player and per tier. Only strikes inside the last `window_minutes` count. A mild strike does not count towards a moderate or severe ladder.

With the defaults, a player who says a mild word is reported, then warned, then muted, all inside 10 minutes. A player who says a severe word is muted straight away.

With `escalation.enabled: false` every hit uses the first step of its tier.

A strike is counted only after `flag_threshold` is reached for the session. A warn needs the player to be online.

If the player has left by the time a mute runs, VoiceSentinel still mutes them when it knows their UUID. If it does not, the hit is only reported.

## Unconfirmed hits

In [On-Server Mode](on-server-mode.md) with the ENGLISH profile, a second model checks every hit (Double Check). With `require_confirmation: true` a hit the second model rejects is **unconfirmed**.

An unconfirmed hit:

* shows to staff as `[unconfirmed]` with its tier,
* is sent to Discord with `(unconfirmed)` added to the words,
* is stored in [Cloud](voicesentinel-cloud.md) with its state,
* does not count towards `flag_threshold`, and never counts a strike, warns, mutes or runs commands.

Set `require_confirmation: false` to act on rejected hits as well. This has no effect on processors that do not do the second check.

## Network strikes

With [MySQL or MariaDB storage](storage.md), strikes are saved in the shared database. `network_strikes: true` counts a player's strikes from every server that uses the same database. A player who is warned on one server is muted on the next.

Give each server its own `network_sync.server_id` (or `server_name`). VoiceSentinel uses it to tell the servers apart. Set `network_strikes: false` to count only strikes from this server.

With SQLite storage, strikes stay on the server. If the database cannot be reached, strikes are counted in memory until it is back.

## Voice mutes

Since 3.10 VoiceSentinel blocks muted players itself (`mutes.enforcement: NATIVE`), so LuckPerms is optional. Set `mutes.enforcement: LUCKPERMS` to have LuckPerms and the voice mod enforce it instead.

By default a muted player cannot talk by proximity, whisper or in groups, and can still hear others. The `mutes` block in [Configuration](configuration.md#voice-mutes-mutes) changes that: which of proximity, whisper and groups are blocked, groups they may still use, whether they stop hearing others, and whether LuckPerms also gets a deny node.

A muted player sees a "Voice muted" line with the time left (action bar by default) and is told, at most every 30 seconds, when they try to talk.

Staff tools: `/voicesentinel vcmute <player> <duration|perm> [reason]`, `unvcmute`, `pardon` (unmute, clear strikes, reset the ladder), `history` and `mutes`. See [Commands and permissions](commands-and-permissions.md). Every mute, unmute and strike can also be sent to your own database with [Data export](data-export.md).

## Mute ladder

The mute ladder sets how long an automatic mute lasts. A repeat offender gets a longer mute. It applies to every automatic mute, whichever tier caused it. Manual `/voicesentinel vcmute` ignores it.

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

The key is the offense number. After the last tier, mutes last `max_tier_duration`. `reset_after_days` clears the offense count after that many days. With `enabled: false`, mutes last `mute-duration-minutes`.

## Appeal message

After every automatic mute the player gets `moderation.appeal_message`. Set `appeal_url` to make it a clickable link.

If [VoiceSentinel Cloud](voicesentinel-cloud.md) is on and `cloud.appeal_links` is true, the player gets Cloud's own appeal link instead. A player never gets both.

Both messages are skipped when `custom_commands.auto_mute_builtin.player_message` is `false`.

## Console and staff messages

Each flag or mute writes one line to the console with the tier, the strike (for example `strike 2/3`) and whether it was caught live. Staff with `voicesentinel.alerts` get an alert with an `Action` line. The text is in `config/messages.yml`.
