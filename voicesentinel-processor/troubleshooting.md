# Troubleshooting (processor)

**Processor won’t start / crashes on start**  
- Check **config.json** is valid JSON (no trailing commas, quoted keys).  
- If you’re using Docker, ensure **config.json** is mounted and the path is correct (e.g. **CONFIG_PATH**).  
- First run can look “stuck” while the Whisper model downloads; check logs. Needs internet for that.

**Plugin says “Processor not connected”**  
- Processor must be running and reachable from the Minecraft server. From the server machine, try `curl http://processor-host:28472/health` (replace with your host/port).  
- **processor_websocket_url** in the plugin: use **ws://** for plain HTTP and **wss://** if you’re behind HTTPS. Host and port must match (default port 28472).  
- **server_key** in the plugin’s config.yml must match **server.server_key** in the processor’s config.json exactly. Wrong key = auth failure; check processor logs for **auth_failed**.  
- Firewall: the server must be able to open a TCP connection to the processor’s port.

**Transcription timeout**  
- Long or noisy audio can exceed **transcription.timeout_seconds**. Increase it (e.g. 60).  
- **audio.max_audio_length_ms** caps clip length (default 30 seconds). Longer clips are rejected before transcription.  
- If the host is underpowered, transcription can be slow; consider a smaller model (e.g. **base** instead of **large-v3**) or more CPU threads.

**Queue full / recordings dropped**  
- **processing.queue_max_size** limits how many jobs can wait. When full, new recordings are dropped. Increase the value or scale the processor.  
- Check **GET /stats** for **processing_queue_size** to see backlog.

**Negative or wrong processing time in alerts**  
- The app uses monotonic time and clamps to ≥ 0. If you still see wrong values, make sure you’re on the latest build. The plugin also clamps negative values to 0.

**Recording save / legal**  
- **recordings.save_mode** controls whether WAVs are stored. Check your local laws and privacy policy. Prefer **"none"** or **"flagged"** unless you have a clear need and compliance in place. **retention_days** should match your policy.
