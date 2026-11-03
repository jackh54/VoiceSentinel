# Discord webhooks

Send **flagged content** and **mute/unmute** events to Discord.

## Flagged content

In **`config.yml`** under **`discord`**:

- **`enabled`:** `true`
- **`webhook_url`:** channel webhook URL
- Optional: **`username`**, **`avatar_url`**, **`embed_color`**, **`ping_role_id`**, **`timeout_seconds`**, **`retry_attempts`**
- **`include_transcript`**, **`include_server_info`**
- **`include_audio`:** attach flagged audio to the webhook **only if** the processor has **`response.include_audio: true`** (plugin sends base64 audio from the processor when available)
- **`customization`:** embed title, field names, footer, etc.

Flagged-content alerts are sent when **`flag_threshold`** is reached, on every escalation step. A hit that Double Check rejected is still sent, with `(unconfirmed)` added to the words. See [Moderation and escalation](moderation-and-escalation.md#unconfirmed-hits).

Test without real voice: **`/voicesentinel discordtest flag`** or **`all`**.

## Mute / unmute

Under **`discord_mute`:** separate webhook for mute and unmute notifications. Automatic mutes only post here when **`custom_commands.auto_mute_builtin.discord_webhook`** is `true` (the default).

- **`include_transcript`:** **`flagged`**, **`muted`**, **`both`**, or **`none`**
- **`include_audio`:** same requirement as above: processor **`response.include_audio`**
- **`customization`** and **`unmute_customization`** for embed text

Test: **`/voicesentinel discordtest mute`**.

## Tips

- Treat webhook URLs as secrets.  
- For incident history, buttons and remote unmute in Discord, see [VoiceSentinel Cloud](voicesentinel-cloud.md). It uses its own bot, not these webhooks.  
- If nothing posts, verify URL, channel, and outbound HTTPS from the Minecraft host. Check server logs for webhook errors.  
- For audio attachments, enable **`response.include_audio`** on the **processor** and **`include_audio`** in the plugin’s Discord sections.
