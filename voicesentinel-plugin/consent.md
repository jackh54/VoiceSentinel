---
description: >-
  How players agree to voice moderation: a join notice or a confirm/deny menu,
  the /voiceconsent command and rules_version.
---

# Consent

The `consent` block sets how players agree to voice moderation.

| Mode | What happens |
|------|--------------|
| **NOTICE** (default) | Joining counts as consent. The player sees `privacy-msg` on join. |
| **GUI** | The player confirms or denies in a menu. Their voice is never processed unless they accept. |

In both modes `privacy-join-msg` and `privacy-msg` control the join notice.

## Config

```yaml
consent:
  mode: NOTICE
  # GUI mode, when a player denies. IGNORE = they stay, voice not processed. KICK = removed.
  # MUTE = they stay and can listen, but nobody hears them until they accept.
  on_deny: IGNORE
  # Raise this after changing your rules to ask everyone again.
  rules_version: 1
  title: "&8Voice chat rules"
  accept_name: "&aI accept"
  deny_name: "&cI do not accept"
  rules_name: "&bVoice chat rules"
  rules:
    - "&7Voice chat is moderated automatically."
    - "&7Your voice is converted to text and checked"
    - "&7against the server rules."
  accepted_message: "&aAccepted. Voice chat is enabled."
  denied_message: "&eDeclined. Your voice is not processed. Change this with /voiceconsent."
  muted_message: "&eDeclined. You can hear voice chat, but nobody can hear you. Change this with /voiceconsent."
  kick_message: "&cYou must accept the voice chat rules to play on this server."
  notice_mode_message: "&7Voice chat is moderated. Your voice is converted to text and checked."
```

Text uses `&` colour codes.

## GUI mode

A new player gets a menu with three items: accept, the rules (a book with the lines from `rules`), and deny.

* Pressing Escape is not an answer. The menu opens again until the player chooses.
* **Accept**: voice is moderated.
* **Deny** with `on_deny: IGNORE`: the player stays on the server. Their voice is not processed.
* **Deny** with `on_deny: KICK`: the player is removed with `kick_message`. They get the menu again on their next join.
* **Deny** with `on_deny: MUTE` (3.11+): the player stays and hears voice chat, but nobody hears them. They get `muted_message`. Players who have not answered yet are muted too, so nobody talks unmoderated while the menu is open. Works with Simple Voice Chat and PlasmoVoice.

Answers are saved in `plugins/VoiceSentinel/consent.properties`.

## /voiceconsent

`/voiceconsent` (alias `/vconsent`) needs no permission. In GUI mode it opens the menu again so a player can change their answer. In NOTICE mode it shows `notice_mode_message`.

## rules_version

Answers are stored with the `rules_version` they were given for. Raise the number after you change your rules. Everyone is asked again, including players who are online (a reload applies it).

## Check it

* `/voicesentinel test` shows the consent mode.
* `/voicesentinel debug <player>` shows whether a player's voice is being processed, and if bypass or a mute applies.
