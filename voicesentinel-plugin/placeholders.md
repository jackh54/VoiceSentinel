---
description: >-
  PlaceholderAPI placeholders from VoiceSentinel: mute state, time left,
  reason, strikes and consent.
---

# Placeholders

VoiceSentinel registers a [PlaceholderAPI](https://www.spigotmc.org/resources/placeholderapi.6245/) expansion with the identifier `voicesentinel`. Use the placeholders in scoreboards, tab lists, chat formats or any plugin that reads PlaceholderAPI.

The expansion registers by itself when PlaceholderAPI is installed. There is nothing to enable for it.

## Placeholders

| Placeholder | Output | Example |
|-------------|--------|---------|
| `%voicesentinel_muted%` | `true` if the player is voice muted, otherwise `false` | `true` |
| `%voicesentinel_mute_remaining%` | Time left on the mute, or `permanent`. Empty when not muted | `1h 29m` |
| `%voicesentinel_mute_reason%` | The mute reason. Empty when not muted or no reason was set | `Repeated offences` |
| `%voicesentinel_strikes%` | Number of strikes inside the strike window, as a whole number | `2` |
| `%voicesentinel_consent%` | `true` if voice moderation may process the player's voice, otherwise `false` | `true` |

`mute_remaining` is written as `Xd Yh`, `Xh Ym`, `Xm Ys` or `Xs`, using only the largest two units that apply.

{% hint style="info" %}
The values are read from memory, so they are safe to use in placeholders that refresh every tick. An unknown placeholder such as `%voicesentinel_foo%` is left as it is.
{% endhint %}

## Strike count

`%voicesentinel_strikes%` counts the player's strikes in the [moderation window](moderation-and-escalation.md). It is loaded when the player joins and goes up with each new strike. For a player who is not online it is `0`. A pardon resets it.

## Consent

`%voicesentinel_consent%` is `true` when [consent](consent.md) is not required, or the player has agreed. It is `false` in `GUI` mode for a player who has denied, not answered, or not answered the current rules.

## Placeholders in VoiceSentinel's own messages

```yaml
placeholder_api:
  enabled: false
```

With `placeholder_api.enabled: true` and PlaceholderAPI installed, VoiceSentinel also runs PlaceholderAPI over its own player and staff messages, so you can put other plugins' placeholders into your language file. This only affects those messages. The `voicesentinel` placeholders above work whether it is on or off.
