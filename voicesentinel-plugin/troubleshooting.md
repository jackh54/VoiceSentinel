# Troubleshooting (plugin)

**Plugin won’t start**  
- **license-key** missing or invalid.  
- No voice mod: install **Simple Voice Chat** or **PlasmoVoice**. Having both also stops the plugin. Install one.  
- **`voice_backend`** names a mod that is not installed. Set it to `AUTO` or install the mod.  
- **Paper/Folia 1.21+** and **Java 21** required.  
- **CUSTOM** mode: **server_key** empty or still the default placeholder. Set a real 16 to 256 character key matching the processor. (**PUBLIC** and **LOCAL** do not use it.)  
- **LuckPerms** is not needed to start. Without it the console says `LuckPerms not found: built-in voice mute is off` and automatic mutes do not happen.

**“Processor not connected” / no transcripts**  
- Run **`/voicesentinel test`**. It shows the mode and whether the speech link is connected.  
- **PUBLIC / CUSTOM:** the processor (or pool) must be reachable from the Minecraft host. For a self-hosted processor try **`curl http://host:port/health`**.  
- **CUSTOM:** **processor_websocket_url** must be correct (`ws://` or `wss://`). **server_key** in **`config.yml`** must match **`server.server_key`** in **`config.json`**. Check processor logs for auth failures.  
- **PUBLIC:** **processor_connection_mode** must be **`PUBLIC`**. Ensure the plugin version is supported by the pool. Check **`processor_discovery_url`** if you overrode it.  
- **LOCAL:** see [On-Server Mode](#on-server-mode) below.  
- Firewall: outbound TCP from the game server to the processor (or pool) must be allowed.

## On-Server Mode

**Engine does not download** (`Speech engine install failed`)  
- The server needs outbound HTTPS access to GitHub on the first start. Check firewall and proxy rules.  
- Check free disk space. The engine and models are stored in **`plugins/VoiceSentinel/engine/`**.  
- A file that fails its checksum is not used. Restart to try again.  
- Restart the server after fixing the cause.

**Engine keeps crashing** (`Speech engine keeps crashing. Voice moderation is off.`)  
- Check free RAM first. ENGLISH needs about 1 GB free, MULTILANG about 2.5 GB. The engine is a separate process outside the Minecraft heap, so this is RAM left over after the server.  
- Read **`plugins/VoiceSentinel/engine/engine.log`**.  
- The plugin restarts the engine up to five times in a row, then stops. Restart the server once the cause is fixed.  
- Switch from MULTILANG to ENGLISH if you do not need other languages.

**Hits are slow**  
- Older or slower CPUs take longer to check audio. The 0.7 second live detection time is for a modern CPU.  
- On a busy server the engine shares the CPU with Minecraft.

**Live detection is not working**  
- It needs **`processor_connection_mode: LOCAL`**, **`local_engine.profile: ENGLISH`** and **`audio_capture.live_detection: true`**.  
- MULTILANG, PUBLIC and CUSTOM modes do not do live detection. They check after the player stops talking.  
- The startup lines list `live detection` when it is on.

**“Unconfirmed” hits in staff alerts**  
- Double Check: the second model did not hear the word. The hit is shown to staff, but nobody is warned or muted and it does not count a strike.  
- To act on rejected hits anyway, set **`moderation.require_confirmation: false`**. See [Moderation and escalation](moderation-and-escalation.md#unconfirmed-hits).

**Web dashboard: “unauthorized” / cannot log in**  
- **`web_dashboard.enabled`** must be **`true`**.  
- Create users with **`voicesentinel webuser create …`** from **console/RCON**, then **`setuplink`** or **`setpassword`**.  
- **`ip_allowlist`:** if not empty and not `*`, your browser’s IP must be listed (or use SSH tunnel / bind to localhost + proxy).

**Transcripts empty in dashboard or /reportvoice**  
- Enable **`transcript_buffer.enabled`** in **`config.yml`**.  
- For **`/reportvoice`**, also set **`voice_report.enabled: true`** and grant **`voicesentinel.report`**.  
- Processor must be connected and transcribing.

**A player is not moderated**  
- Run **`/voicesentinel debug <player>`**. It shows consent, the bypass permission and voice mute state.  
- In **GUI** consent mode a player who has not accepted is never processed. See [Consent](consent.md).  
- **`voicesentinel.bypass`** exempts a player (for example through LuckPerms). Config changes need **`/voicesentinel reload`**. Permission changes usually apply live.

**Words are not flagged, or too many are flagged**  
- With **`partial_match: false`** only whole words match. Add the forms you need (`fucking`, `fucked`) to **`config/wordlist.txt`**.  
- With **`partial_match: true`** short words match inside other words (`crap` in `scrap`).  
- An old unedited default list is replaced on update. See [Updating](updating.md#your-word-list).

**Config overrides (profanity_words / mute-list)**  
- Save **`config.yml`** and run **`/voicesentinel reload`**.  
- Words in these lists are added to the default language from **`config/wordlist.txt`**. They do not replace it.

**Warn or mute does not happen**  
- Check the steps in **`moderation.escalation`**. A first mild strike is only a `report` by default.  
- Strikes only count after **`flag_threshold`** is reached.  
- Strikes older than **`escalation.window_minutes`** do not count.  
- A `warn` needs the player to be online. A `mute` needs LuckPerms.

**Storage database unreachable**  
- The console says `Could not reach the storage database`. Strikes are counted in memory and ladder mutes use the default length until it reconnects. Check **`storage`** host, port, database, username and password. See [Storage](storage.md).  
- With **`storage.type: MYSQL`** the **host**, **database** and **username** must be filled in.

**VoiceSentinel Cloud**  
- Run **`/voicesentinel cloud`**. `Entitled: no` means the license does not include Cloud. See [VoiceSentinel Cloud](voicesentinel-cloud.md#who-gets-it).  
- Set **`cloud.enabled: true`** and run **`/voicesentinel reload`**.  
- Remote unmute needs LuckPerms.

**Alerts or Discord**  
- Alerts: **alert_staff** and **`voicesentinel.alerts`**.  
- Discord: **discord.enabled** and URLs set. Test **`/voicesentinel discordtest all`**.  
- Optional **include_audio** on flagged/mute embeds requires **`response.include_audio: true`** on the **processor** and **`include_audio: true`** in the plugin Discord blocks.

**Mute ladder**  
- Requires **LuckPerms**. Enable **`mute_ladder.enabled`**. Durations use the same syntax as **`/voicesentinel vcmute`** (for example `30m`, `1h`).

**Custom commands (`mute_commands` / `profanity_commands`) never run**  
- **`mute_commands`:** only after a **successful auto-mute**. Check **LuckPerms** if the built-in LuckPerms mute is on and LuckPerms is missing.  
- **`profanity_commands`:** only when the hit does not mute (step `report` or `warn`), with **`flag_threshold`** met, and words with no keyword mapping (see [Custom commands](custom-commands.md)).  
- Unconfirmed hits never run commands.  
- **CommandWhitelist** (or similar): allow the command root for **console**.  
- Watch the server log for **“Custom command was not handled”** or dispatch errors after a flag.
