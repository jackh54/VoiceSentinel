# Word lists

The plugin (and processor) use word lists to decide what counts as profanity vs. what triggers an auto-mute. You can use the built-in **wordlist.txt** or override it in **config.yml**.

## wordlist.txt

Location: `plugins/VoiceSentinel/wordlist.txt`.

Format: sections by language and type, then one word per line. Lines starting with `#` are comments.

- **PROFANITY** – Triggers alerts and (optionally) profanity_commands. No auto-mute by default.
- **MUTE** – Triggers auto-mute plus mute_commands and alerts.

Section header format: `[LANG-TYPE]`. Language is the ISO code (e.g. `EN`, `ES`). The processor detects language and uses the matching section.

Example:

```text
# English – profanity only (alerts / profanity_commands)
[EN-PROFANITY]
word1
word2

# English – auto-mute
[EN-MUTE]
word3
word4

# Spanish
[ES-PROFANITY]
word5
```

So if someone speaks English and says something in the MUTE list, they get auto-muted. If it’s only in the PROFANITY list, they get flagged and you can run profanity_commands (e.g. a warning) but no mute unless you add the word to MUTE too.

## Overriding in config.yml

If you don’t want to edit wordlist.txt and just want a short list for your server:

```yaml
profanity_words:
  - badword1
  - badword2
mute-list:
  - muteword1
```

Non-empty lists here **replace** the default-language lists from wordlist.txt for what gets sent to the processor. Leave them as `[]` to use wordlist.txt (and per-language lists) instead.

After changing wordlist.txt or these options, run `/voicesentinel reload` so the plugin picks it up.
