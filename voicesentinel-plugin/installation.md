---
description: >-
  Install the VoiceSentinel plugin on Paper/Folia and choose how speech is
  processed: PUBLIC pool, On-Server Mode (LOCAL) or your own processor (CUSTOM).
---

# Installation

You need a few things in place before the plugin will work:

* **Paper or Folia 1.21+** (Bukkit/Spigot is not supported)
* **Java 21**
* **Simple Voice Chat** or **PlasmoVoice** (the voice mod players use). Install one, not both.
* **LuckPerms** is optional. Voice mutes work without it. It is only needed if you set `mutes.enforcement: LUCKPERMS` or `mutes.mirror_luckperms: true`.
* A **license key**. See [Getting your license key](getting-your-license-key.md).

Drop **`VoiceSentinel.jar`** into **`plugins/`** and restart. On first start it creates **`plugins/VoiceSentinel/`** with **`config.yml`**. Word lists and messages are in **`plugins/VoiceSentinel/config/`** (`wordlist.txt`, `messages.yml`, `languages.yml`).

## Processor modes

VoiceSentinel turns speech into text with a processor. You choose where it runs with **`processor_connection_mode`**.

| Mode | Where speech is processed | Setup | Audio leaves your server |
|------|---------------------------|-------|--------------------------|
| **PUBLIC** (default) | Shared PandaScript pool | None | Yes, to the pool |
| **LOCAL** | On your Minecraft server (On-Server Mode) | None, no Docker. The engine downloads automatically on first start | No |
| **CUSTOM** | A processor you run yourself | Install the [processor](../voicesentinel-processor/installation.md), set URL and key | Only to your processor |

### LOCAL (On-Server Mode)

```yaml
processor_connection_mode: LOCAL
local_engine:
  profile: ENGLISH
  threads: 2
```

* **ENGLISH** needs about 1 GB of free RAM. It checks every hit twice and supports live detection.
* **MULTILANG** supports many languages, checks once, and needs about 2.5 GB of free RAM.

On first start the plugin downloads the engine and models by itself. This needs internet access and disk space. If the server cannot download, you can place the engine file by hand. See [On-Server Mode](on-server-mode.md#turn-it-on).

### PUBLIC pool

```yaml
processor_connection_mode: PUBLIC
```

The plugin finds pool endpoints from the directory (see **`processor_discovery_url`** in [Configuration](configuration.md)). You do not set a **`server_key`**. Keep the plugin on the latest version.

### CUSTOM (your own processor)

```yaml
processor_connection_mode: CUSTOM
processor_websocket_url: "ws://192.168.1.10:28472"
server_key: "MySecretKeyAtLeast16CharsLong"
```

Set **`processor_websocket_url`** to your processor (for example `ws://192.168.1.10:28472` or `wss://processor.example.com`). Set **`server_key`** to the **exact same** value as **`server.server_key`** in the processor's **`config.json`** (16 to 256 alphanumeric characters).

## Finish setup

{% stepper %}
{% step %}
### Add your license key

Put your **license key** in **`config.yml`**, or start the server without it and run **`/voicesentinel license <key>`**. See [Getting your license key](getting-your-license-key.md).
{% endstep %}

{% step %}
### Choose a mode

Set **`processor_connection_mode`** to **PUBLIC**, **LOCAL** or **CUSTOM** and fill in the settings for that mode.
{% endstep %}

{% step %}
### Choose how players consent

The default is a join notice. You can ask players to accept in a menu instead. See [Consent](consent.md).
{% endstep %}

{% step %}
### Restart

Restart the server. After the first start, **`/voicesentinel reload`** applies config changes. In LOCAL mode, wait for `Speech engine: Ready` in the console.

Run **`/voicesentinel test`** to check your whole setup: speech link, license, voice chat mod, engine and more.
{% endstep %}
{% endstepper %}

{% hint style="danger" %}
**Do not use Plugman (or similar) to load VoiceSentinel late.** It must load with the server so it can integrate with the voice mod.
{% endhint %}

## Optional

* **Web dashboard.** If you enable **`web_dashboard`** in **`config.yml`**, create accounts with **`voicesentinel webuser ...`** from the **server console or RCON** (not in-game). See [Web dashboard](web-dashboard.md).
* **VoiceSentinel Cloud.** Incident history, a Discord bot and appeals. See [VoiceSentinel Cloud](voicesentinel-cloud.md).
* **Shared storage.** Share mute history and strikes between servers with MySQL or MariaDB. See [Storage](storage.md).
* **Try it.** See [Try it](try-it.md).
