# Troubleshooting (plugin)

**Plugin won’t start**  
- **license-key** missing or invalid.  
- **LuckPerms** or **Simple Voice Chat** missing.  
- **Paper/Folia 1.21+** and **Java 17+** required.  
- **CUSTOM** mode: **server_key** empty or still the default placeholder — set a real 16–256 character key matching the processor (unless you intentionally use **PUBLIC** mode).

**“Processor not connected” / no transcripts**  
- Processor (or PUBLIC pool) must be reachable from the Minecraft host. Try **`curl http://host:port/health`** for a self-hosted processor (host/port from your setup).  
- **CUSTOM:** **processor_websocket_url** must be correct (`ws://` or `wss://`). **server_key** in **`config.yml`** must match **`server.server_key`** in **`config.json`**. Check processor logs for auth failures.  
- **PUBLIC:** **processor_connection_mode** must be **`PUBLIC`**. Ensure the plugin version is supported by the pool; check **`processor_discovery_url`** if you overrode it.  
- Firewall: outbound TCP from the game server to the processor (or pool) must be allowed.

**Web dashboard: “unauthorized” / cannot log in**  
- **`web_dashboard.enabled`** must be **`true`**.  
- Create users with **`voicesentinel webuser create …`** from **console/RCON**, then **`setuplink`** or **`setpassword`**.  
- **`ip_allowlist`:** if not empty and not `*`, your browser’s IP must be listed (or use SSH tunnel / bind to localhost + proxy).

**Transcripts empty in dashboard or /reportvoice**  
- Enable **`transcript_buffer.enabled`** in **`config.yml`**.  
- For **`/reportvoice`**, also set **`voice_report.enabled: true`** and grant **`voicesentinel.report`**.  
- Processor must be connected and transcribing.

**Bypass not working**  
- Grant **`voicesentinel.bypass`** (e.g. LuckPerms). Config changes need **`/voicesentinel reload`**; permission changes usually apply live.

**Config overrides (profanity_words / mute-list)**  
- Save **`config.yml`** and run **`/voicesentinel reload`**.  
- Non-empty YAML lists override the default language from **wordlist.txt**; **`[]`** means use **wordlist.txt**.

**Alerts or Discord**  
- Alerts: **alert_staff** and **`voicesentinel.alerts`**.  
- Discord: **discord.enabled** / URLs set; test **`/voicesentinel discordtest all`**.  
- Optional **include_audio** on flagged/mute embeds requires **`response.include_audio: true`** on the **processor** and **`include_audio: true`** in the plugin Discord blocks.

**Mute ladder**  
- Requires **LuckPerms**. Enable **`mute_ladder.enabled`**. Durations use the same syntax as **`/voicesentinel vcmute`** (e.g. `30m`, `1h`).

**Custom commands (`mute_commands` / `profanity_commands`) never run**  
- **`mute_commands`:** only after a **successful auto-mute** while the player is **online**. Check **LuckPerms** if the built-in LP mute is required and missing.  
- **`profanity_commands`:** only on **profanity-only** flags (not the auto-mute path), with **`flag_threshold`** met and words matching your **`profanity_words`** config (see [Custom commands](custom-commands.md)).  
- **CommandWhitelist** (or similar): allow the command root for **console**.  
- Watch the server log for **“Custom command was not handled”** or dispatch errors after a flag.
