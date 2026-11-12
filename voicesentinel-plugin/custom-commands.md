# Custom commands

When someone is flagged and **`flag_threshold`** is reached, the plugin can run **console** commands from **config.yml** under **`custom_commands`**.

[Unconfirmed](moderation-and-escalation.md#unconfirmed-hits) hits never run commands.

## Order of precedence

1. **keyword\_commands**: if a detected word has a mapping here, **only** that keyword’s commands run for that word (once per session per keyword). Global mute/profanity commands are **not** used for mapped words. They run on every escalation step (report, warn and mute).
2. **mute\_commands**: fallback when an **auto-mute** happens and the detected word has **no** keyword mapping.
3. **profanity\_commands**: fallback when the hit does **not** mute (the escalation step is `report` or `warn`) and the word has **no** keyword mapping.

Keyword keys should match entries in **wordlist.txt**, **`mute-list`**, or **`profanity_words`**. Matching respects **`case_sensitive`** and **`partial_match`** (see [Configuration](configuration.md)).

## mute\_commands

Run **once** after a **successful auto-mute** (`flag_threshold` reached, the escalation step is `mute`, built-in mute applied). If auto-mute cannot run (for example `mutes.enforcement` is `LUCKPERMS` and LuckPerms is missing, while **`auto_mute_builtin.luckperms_voice_mute`** is `true`), these commands do not fire. With `luckperms_voice_mute: false`, the built-in mute is skipped and these commands are how you mute. See [Moderation and escalation](moderation-and-escalation.md).

## profanity\_commands

Run **once** when the hit does **not** mute (the escalation step is `report` or `warn`), the response includes profanity detections, **`flag_threshold`** is reached, and at least one detected word has **no** keyword mapping.

## keyword\_commands

Per-word command lists. Example:

```yaml
custom_commands:
  keyword_commands:
    kys:
      - "warn {player} Self-harm language detected"
    nazi:
      - "lp user {player} ban voice 7d"
  mute_commands:
    - "say [Voice] {player} was auto-muted for: {words}"
  profanity_commands:
    - "say [Voice] Profanity from {player}: {words}"
```

Each keyword runs **at most once per moderation session** for that word.

## Execution behaviour

Commands run as the **server console** on the **global / main tick thread** (Paper and Folia), so they behave like commands you typed in the server console. Use placeholders; the plugin substitutes them, then dispatches the line.

If the server does not recognise the command (wrong name, plugin not loaded, or blocked), the plugin logs a **warning** that the custom command was not handled. Check spelling and **command whitelist** plugins (many must allow the command for **console**).

## Placeholders

| Placeholder | Replaced with |
|-------------|----------------|
| **{player}** | Player’s name. |
| **{words}** | The words that were detected, comma-separated. |
| **{transcript}** | The full transcript of what was said. |

## Examples

Announce in chat when someone is auto-muted (global fallback):

```yaml
custom_commands:
  mute_commands:
    - "say [Voice] {player} was auto-muted for: {words}"
```

Log profanity to console (global fallback):

```yaml
profanity_commands:
  - "say [Voice] Profanity from {player}: {words}"
```

Multiple commands in one list run **in order**. If one line throws (syntax error, etc.), the plugin logs it and continues with the rest. Use **YAML quotes** when the command has spaces, `:` in URLs, or special characters.

Prefer the same form you would type in **console** without a leading slash (e.g. `say hello`, `ban SomePlayer 7d reason…`). Leading slashes are not required.
