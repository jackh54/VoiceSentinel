# Discord webhooks

You can send flagged content and mute events to Discord so staff get alerts in a channel.

## Flagged content

In **config.yml** under **discord**:

- Set **enabled** to `true`.
- Set **webhook_url** to the Discord webhook URL (channel → Integrations → Webhooks → copy URL).

When something is flagged, the processor sends the result to the plugin and the plugin can post an embed: player, transcript, flagged words, optional server name. You can tweak titles and field names under **discord.customization**.

Test it without triggering real voice: `/voicesentinel discordtest all` (or `flag` to test only the flagged-content webhook).

## Mute / unmute

**discord_mute** is a separate block for mute and unmute events. Same idea: **enabled**, **webhook_url**, and **customization** for embed text. Use it if you want mutes (auto or manual) and unmutes in a different channel or with different text.

Test: `/voicesentinel discordtest mute`.

## Tips

- Webhook URL is secret; don’t commit it or share it.
- If nothing appears in Discord, check the URL, that the webhook is for the right channel, and that the plugin can reach Discord (no firewall blocking outbound HTTPS). Check server logs for webhook errors.
