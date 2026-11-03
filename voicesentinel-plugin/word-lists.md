# Word lists

The plugin and processor use word lists to decide what to flag, and how serious a hit is. You can use **`wordlist.txt`**, **`config.yml`** and the **web dashboard** (superusers).

## wordlist.txt

Location: **`plugins/VoiceSentinel/config/wordlist.txt`**.

The file is split into sections by **language** and **tier**. Put one word or phrase on each line. Lines starting with **`#`** are comments.

| Section | Tier | Meant for | Default action |
|---------|------|-----------|----------------|
| **`[XX-PROFANITY]`** | mild | Swearing that is not aimed at anyone | report, then warn, then mute |
| **`[XX-MODERATE]`** | moderate (optional) | Insults, harassment and threats aimed at a person | warn, then mute |
| **`[XX-MUTE]`** | severe | Slurs, hate and telling someone to harm themselves | mute straight away |

What each tier does, and how it escalates when a player repeats it, is set in the `moderation` block of **`config.yml`**. See [Moderation and escalation](moderation-and-escalation.md).

`XX` is the language code, for example **`[EN-PROFANITY]`** or **`[ES-MUTE]`**. A heading without a language (**`[MUTE]`**) means English. The default file has sections for en, es, fr, de, pt, it, ru, ja and ar. Add other languages with your own **`[XX-...]`** sections.

Example:

```text
[EN-PROFANITY]
word1

[EN-MODERATE]
word2

[EN-MUTE]
word3

[ES-PROFANITY]
word4
```

With the ENGLISH profile of [On-Server Mode](on-server-mode.md) and with a processor that detects the language, the section for the spoken language is used. With the MULTILANG profile the engine does not report the language, so every list applies.

## Whole-word matching

New installs match **whole words** (`partial_match: false` in **`config.yml`**). `crap` does not flag `scrap`, and `cock` does not flag `peacock`.

Because of this, the default list names the common forms of each word: `fuck`, `fucks`, `fucked`, `fucking` and so on. Add forms the same way.

With **`partial_match: true`**, a word matches anywhere inside another word. That catches more forms but also flags innocent words. If you use it, keep short entries out of the list.

## Phrases

A multi-word entry such as `kill yourself` matches those words in that order. [On-Server Mode](on-server-mode.md) can catch phrases mid-sentence with live detection.

## Overriding in config.yml

```yaml
profanity_words:
  - badword1
mute-list:
  - muteword1
```

Words in **`profanity_words`** and **`mute-list`** are **added** to the default-language lists from **`wordlist.txt`**. Empty **`[]`** means only the file is used. The plugin also accepts the old key **`profanity_list`** as an alias for **`profanity_words`**.

These two keys cover mild and severe words. For the moderate tier, use **`[XX-MODERATE]`** in **`wordlist.txt`**.

After changes, run **`/voicesentinel reload`**.

## Matching options

In **`config.yml`**:

* **`case_sensitive`** (default `false`): exact letter casing when comparing words.
* **`partial_match`** (default `false` on new installs): `false` = whole words only, `true` = substring match.

Per-word **custom commands** use the same rules. See [Custom commands](custom-commands.md).

## Upgrading the default list

3.8.0 ships a new default list: whole-word matching, a filled moderate tier, self-harm phrases and slurs as mute words. `nazi` and `hitler` on their own are no longer listed, so talking about history is fine. `heil hitler` and `sieg heil` are mute words.

If your **`wordlist.txt`** is the **unedited** default from 3.6.0 or 3.7.0, the plugin replaces it on start:

* the new default is written to **`config/wordlist.txt`**,
* your old file is kept as **`config/wordlist.txt.old`**,
* `partial_match: true` in **`config.yml`** is changed to `false`.

If you edited the list, it is left alone and `partial_match` stays as it was. To switch later, see [Updating](updating.md#your-word-list).

## Web dashboard (superuser)

Superusers can open **Config** in the **web dashboard** to add and remove words per language and tier. Updates are saved to **`config/wordlist.txt`** on disk. Run **`/voicesentinel reload`** if you also edit the file by hand.

See [Web dashboard](web-dashboard.md) and [Configuration](configuration.md).
