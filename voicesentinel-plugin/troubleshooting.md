# Troubleshooting (plugin)

**Plugin won’t start**  
- License key missing or invalid. Set **license-key** in config.yml and get a valid key from Discord.  
- LuckPerms or Simple Voice Chat not installed. Both are required.  
- Server is not Paper/Folia 1.21+ or Java 17+.

**“Processor not connected” / no transcripts**  
- Processor must be running and reachable. Try opening `http://processor-host:28472/health` in a browser (replace host/port with yours).  
- **processor_websocket_url** must be correct: `ws://` (or `wss://` if you use HTTPS), host, and port (default 28472).  
- **server_key** in config.yml must match **server.server_key** in the processor’s config.json exactly. If they don’t match, the processor rejects the connection (check processor logs for auth_failed).  
- Firewall: the server must be able to open a TCP connection to the processor’s port.

**Bypass not working**  
- User needs the permission **voicesentinel.bypass**. Grant it with LuckPerms (e.g. `lp user <name> permission set voicesentinel.bypass true`).  
- After changing permissions, a reload is not required for permissions, but if you changed config, run `/voicesentinel reload`.

**Config override (profanity_words / mute-list) not applied**  
- Save config.yml and run `/voicesentinel reload`.  
- If you use wordlist.txt as well, non-empty **profanity_words** / **mute-list** replace the default-language list sent to the processor; empty lists mean “use wordlist.txt”.

**Alerts or Discord not working**  
- Alerts: **alert_staff** must be true and staff need **voicesentinel.alerts**.  
- Discord: **discord.enabled** true and **discord.webhook_url** set. Test with `/voicesentinel discordtest all`.
