# Custom commands

When someone is auto-muted or only profanity is detected (no mute), you can run console commands. Those are defined in **config.yml** under **custom_commands**.

- **mute_commands** – Run when the player is **auto-muted** (flag_threshold hit and the phrase was in the MUTE list).
- **profanity_commands** – Run when they’re **flagged for profanity only** (flag_threshold hit but no mute).

Commands run as the **server console**. You can use placeholders; the plugin replaces them before running.

## Placeholders

| Placeholder | Replaced with |
|-------------|----------------|
| **{player}** | Player’s name. |
| **{words}** | The words that were detected, comma-separated. |
| **{transcript}** | The full transcript of what was said. |

## Examples

Announce in chat when someone is auto-muted:

```yaml
custom_commands:
  mute_commands:
    - "say [Voice] {player} was auto-muted for: {words}"
```

Give a LuckPerms mute that expires in 60 minutes:

```yaml
mute_commands:
  - "lp user {player} permission set my.custom.mute true 60m"
```

(You’d need a permission/track set up in LuckPerms that actually mutes them – this is just the command side.)

Log profanity to console:

```yaml
profanity_commands:
  - "say [Voice] Profanity from {player}: {words}"
```

Multiple commands run in order. If a command fails (e.g. wrong syntax), the rest still run. Use quotes in YAML when the command has spaces or special characters.
