# Welcome

<figure><img src=".gitbook/assets/Pbanner.png" alt=""><figcaption></figcaption></figure>

**VoiceSentinel** moderates voice chat on **Paper** and **Folia** servers. It works with **Simple Voice Chat** and **PlasmoVoice**. The plugin turns what players say into text, checks it against your word lists, and warns, reports or mutes them.

Speech is processed in one of three ways (**`processor_connection_mode`** in the plugin):

* **LOCAL (On-Server Mode):** the speech engine runs on your Minecraft server. No Docker, and audio never leaves your server.
* **PUBLIC:** the shared PandaScript pool. No setup.
* **CUSTOM:** a processor you run yourself.

In On-Server Mode a second model double-checks every hit, and a listed word is caught while the player is still talking (live detection).

Each hit is mild, moderate or severe. Repeat offenders move up a ladder of report, warn and mute. Players can agree through a join notice or a menu. Several servers can share strikes through MySQL or MariaDB. The optional **VoiceSentinel Cloud** add-on adds incident history, a Discord bot and player appeals.

Start here:

* **Plugin (Paper / Folia):** [Installation](voicesentinel-plugin/installation.md)
* **No Docker:** [On-Server Mode](voicesentinel-plugin/on-server-mode.md)
* **Updating to 3.10.1:** [Updating](voicesentinel-plugin/updating.md)
* **Hear it work:** [Try it](voicesentinel-plugin/try-it.md)
* **Processor (Docker / bare metal):** [Installation](voicesentinel-processor/installation.md)
* **Legal:** [**https://www.pandascript.dev/voicesentinel**](https://www.pandascript.dev/voicesentinel)

Use the sidebar for the full table of contents.
