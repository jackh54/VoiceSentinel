---
description: >-
  VoiceSentinel Cloud: incident history, audio clips, a Discord bot with
  buttons, remote unmute and extend, and player appeals.
---

# VoiceSentinel Cloud

VoiceSentinel Cloud is an add-on service at [cloud.pandascript.dev](https://cloud.pandascript.dev). The plugin sends flagged moments to it. Staff review them in a web dashboard or in Discord.

It is off until you turn it on.

## What it does

* **Incidents.** Every flag and mute from every server on your license, in one feed. Each one has the transcript, the matched words, the tier and strike, and whether it was unconfirmed. You can search and filter by server.
* **Clips.** A short audio clip of the flagged moment, when `cloud.upload_clips` is on.
* **Discord bot.** Incidents are posted to a staff channel with **Unmute**, **Extend 1h** and **Dismiss** buttons. The server applies the action within seconds.
* **Remote unmute and extend.** Done from Discord or from the dashboard. The plugin polls for them. Unmute and extend need LuckPerms.
* **Appeals.** A muted player gets a private link that shows what was heard and the mute length. The player can write a message (up to 1000 characters). Staff approve or deny it in Discord or the dashboard. Approving an appeal unmutes the player on all your servers.

Incidents, clips and appeals are deleted after 30 days.

## Who gets it

* Licenses bought **before 3 November 2026** get Cloud free.
* Newer licenses add Cloud as an add-on from the [VoiceSentinel listing on BuiltByBit](https://builtbybit.com/resources/voicesentinel-voicechat-moderation.72862/).

There is no separate key. Cloud uses your VoiceSentinel `license-key`.

## Turn it on

{% stepper %}
{% step %}
### Enable it in the plugin

In `config.yml`:

```yaml
cloud:
  enabled: true
  upload_clips: true
  appeal_links: true
```

Run `/voicesentinel reload`.
{% endstep %}

{% step %}
### Sign in to the dashboard

Go to [cloud.pandascript.dev](https://cloud.pandascript.dev) and sign in with your license key. Generate a Discord link code. The code is valid for 15 minutes and works once.
{% endstep %}

{% step %}
### Add the bot to Discord

Add the VoiceSentinel bot to your Discord server with the link on the Cloud site. In your staff channel, run `/link` with the code. You can choose the channel and a staff role.

People with the staff role, or Manage Server, can use the buttons.
{% endstep %}
{% endstepper %}

`/vsstatus` in Discord shows whether the server is linked.

## Options

| Option | Default | What it does |
|--------|---------|--------------|
| **cloud.enabled** | `false` | Turns Cloud on. |
| **cloud.upload_clips** | `true` | Uploads an audio clip with each incident. Clips over 2 MB are not sent. |
| **cloud.appeal_links** | `true` | Gives muted players Cloud's appeal link. When off, they get `moderation.appeal_message`. |

Set `server_name` so your servers have readable names in the feed.

## Check the status

`/voicesentinel cloud` (needs `voicesentinel.admin`) shows whether Cloud is enabled, whether your license is entitled, the last contact, queued incidents and the last error.

If the license does not include Cloud, the console says so and links to the listing.

## Privacy

The plugin sends transcripts of flagged moments, the matched words, the player name and UUID, and clips if you leave them on. It does not send continuous voice, and nothing from players who are not flagged. In [On-Server Mode](on-server-mode.md) the audio is still processed on your server. Only flagged clips leave it, and only with `upload_clips` on.

Tell your players. See the [Cloud privacy policy](https://cloud.pandascript.dev/privacy) for what is stored, how long, and who can see it. You can write your own rules into the [consent menu](consent.md).
