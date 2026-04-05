---
description: >-
  Install the VoiceSentinel plugin on Paper/Folia with LuckPerms and Simple
  Voice Chat.
---

# Installation

You need a few things in place before the plugin will work:

* **Paper or Folia 1.21+** (Bukkit/Spigot is not supported)
* **Java 17+**
* **LuckPerms** (required — voice mutes use it)
* **Simple Voice Chat** (the mod players use for voice)
* A **VoiceSentinel processor** you are allowed to use:
  * **CUSTOM** — your own processor (Docker, bare metal, or host), or
  * **PUBLIC** — the shared pool (set **`processor_connection_mode: PUBLIC`** in `config.yml` and keep the plugin updated)

Drop **`VoiceSentinel.jar`** into **`plugins/`** and restart. On first start it creates **`plugins/VoiceSentinel/`** with **`config.yml`**, **`messages.yml`**, and related files.

**Then finish setup:**

{% stepper %}
{% step %}
### Add your license key

Put your **license key** in **`config.yml`**. See [Getting your license key](getting-your-license-key.md).
{% endstep %}

{% step %}
### Choose CUSTOM or PUBLIC processor

* **CUSTOM (default):** Set **`processor_connection_mode: CUSTOM`**, set **`processor_websocket_url`** to your processor (e.g. `ws://192.168.1.10:28472` or `wss://processor.example.com`), and set **`server_key`** to the **exact same** value as **`server.server_key`** in the processor’s **`config.json`** (16–256 alphanumeric).
* **PUBLIC pool:** Set **`processor_connection_mode: PUBLIC`**. The plugin discovers pool endpoints from the directory (see **`processor_discovery_url`** in [Configuration](configuration.md)). You typically do **not** supply your own **`server_key`** for pool WebSocket auth the same way as self-hosted.
{% endstep %}

{% step %}
### Reload or restart

Restart the server, or run **`/voicesentinel reload`**, so the plugin loads the new settings.
{% endstep %}
{% endstepper %}

{% hint style="danger" %}
**Do not use Plugman (or similar) to load VoiceSentinel late.** It must load with the server so it can integrate with Simple Voice Chat.
{% endhint %}

## Optional: web dashboard

If you enable **`web_dashboard`** in **`config.yml`**, create accounts with **`voicesentinel webuser …`** from the **server console or RCON** (not in-game). See [Web dashboard](web-dashboard.md).
