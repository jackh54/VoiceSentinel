---
description: Install the VoiceSentinel plugin on Paper/Folia with LuckPerms and Simple Voice Chat.
---

# Installation

You need a few things in place before the plugin will work:

- **Paper or Folia 1.21+** (Bukkit/Spigot won’t cut it)
- **Java 17+**
- **LuckPerms** (required – the plugin uses it for voice mutes)
- **Simple Voice Chat** (the mod players use to talk)
- The **VoiceSentinel processor** running somewhere and reachable from the server (see the Processor docs)

Drop `VoiceSentinel.jar` into your server’s `plugins` folder and restart. On first start it’ll create `plugins/VoiceSentinel/` with config files.

**Then finish setup:**

{% stepper %}
{% step %}
## Add your license key

Put your **license key** in `config.yml`. See [Getting your license key](getting-your-license-key.md).
{% endstep %}

{% step %}
## Point the plugin at the processor

Set **processor_websocket_url** in `config.yml` to where the processor is (e.g. `ws://192.168.1.10:28472` or `ws://processor.yourserver.com`).
{% endstep %}

{% step %}
## Set the shared secret

Set **server_key** in `config.yml` to the **exact same** value as **server.server_key** in the processor’s `config.json` (16–256 letters/numbers). If they don’t match, the plugin won’t connect.
{% endstep %}

{% step %}
## Reload or restart

Restart the server, or run `/voicesentinel reload`, so the plugin picks up the config.
{% endstep %}
{% endstepper %}

{% hint style="danger" %}
**Do not use Plugman (or similar) to load VoiceSentinel.** It has to load with the server so it can hook into Simple Voice Chat. Loading it late will break that and cause issues.
{% endhint %}
