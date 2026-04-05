# Word lists

The plugin and processor use lists to classify **profanity** (flag / alert) vs **mute** (auto-mute path). You can use **`wordlist.txt`**, **`config.yml`**, and/or the **web dashboard** (superusers).

## wordlist.txt

Location: **`plugins/VoiceSentinel/wordlist.txt`**.

Format: sections by **language** and **type**, one word per line. Lines starting with **`#`** are comments.

- **PROFANITY** — Flags, alerts, optional **profanity_commands** (no auto-mute from the list alone).
- **MUTE** — Auto-mute path when thresholds are met, plus **mute_commands** and alerts.

Section headers: **`[LANG-TYPE]`** (e.g. **`[EN-PROFANITY]`**, **`[EN-MUTE]`**). The processor uses detected language to pick a section.

Example:

```text
[EN-PROFANITY]
word1

[EN-MUTE]
word2

[ES-PROFANITY]
word3
```

## Overriding in config.yml

```yaml
profanity_words:
  - badword1
mute-list:
  - muteword1
```

Non-empty lists **replace** the default-language content from **wordlist.txt** for what is sent to the processor. Empty **`[]`** means use **wordlist.txt**. The plugin also accepts the legacy key **`profanity_list`** as an alias for **`profanity_words`**.

After changes, run **`/voicesentinel reload`**.

## Web dashboard (superuser)

Superusers can open **Config** in the **web dashboard** to add/remove words per language and type. Updates are saved to **`wordlist.txt`** on disk. Reload the plugin (or rely on the implementation’s live update path) if you also edit files by hand on disk.

See [Web dashboard](web-dashboard.md) and [Configuration](configuration.md).
