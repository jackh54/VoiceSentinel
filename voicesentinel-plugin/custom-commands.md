# Custom commands

When someone is auto-muted or only profanity is detected (no mute), you can run console commands. Those are defined in **config.yml** under **custom_commands**.

- **mute_commands** – Run **once** after a **successful auto-mute** for an **online** player ( **`flag_threshold`** reached, mute path, built-in mute applied). If the player is offline or auto-mute cannot run (e.g. LuckPerms missing when required), these commands do not fire.
- **profanity_commands** – Run **once** when the hit is **profanity-only** (processor does **not** request a mute for that event), the response includes **profanity** detections, **`flag_threshold`** is reached, and at least one detected word is in your configured **`profanity_words`** list (or wordlist profanity section). If everything is classified as **mute-list** only, use **`mute_commands`** instead.

Commands run as the **server console** on the **global / main tick thread** (Paper and Folia), so they behave like commands you typed in the server console. Use placeholders; the plugin substitutes them, then dispatches the line.

If the server does not recognise the command (wrong name, plugin not loaded, or blocked), the plugin logs a **warning** that the custom command was not handled—check spelling and **command whitelist** plugins (many must allow the command for **console**).

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

Multiple commands run **in order**. If one line throws (syntax error, etc.), the plugin logs it and continues with the rest. Use **YAML quotes** when the command has spaces, `:` in URLs, or special characters.

Prefer the same form you would type in **console** without a leading slash (e.g. `say hello`, `ban SomePlayer 7d reason…`). Leading slashes are not required.

Word matching can use **`case_sensitive: true`** in **`config.yml`** (see [Configuration](configuration.md)).
