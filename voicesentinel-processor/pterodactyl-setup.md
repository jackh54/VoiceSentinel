# Pterodactyl setup

You can run the VoiceSentinel processor as a Pterodactyl “game” server using the provided egg. The egg uses the **Python 3.11** yolk image, installs dependencies, and starts the processor. No need to clone the repo on the server yourself – the egg handles it.

## What the egg does

**On install (once):**

- Installs ffmpeg, curl, git.
- Clones the VoiceSentinel repo into a **temporary** folder, copies only **app/** and **requirements.txt** into the server root, then deletes the clone. So the server root has **app**, **config.json**, **requirements.txt**, **venv**, etc. – no permanent `voicesentinel` folder.
- Creates **config.json** from the example if you don’t have one.
- Creates **models** and **tmp** directories.

**On every start:**

- Tries to clone the repo again into a temp folder and overwrites **app/** and **requirements.txt** with the latest version (so you get updates on restart). **config.json** is never overwritten.
- Creates/updates the Python venv and installs pip packages (including PyTorch CPU and faster-whisper).
- Starts uvicorn with the port Pterodactyl gives you (e.g. **SERVER_PORT**).

So: one-time install gives you a working tree; every restart pulls the latest code and keeps your config.

## Importing the egg

1. In Pterodactyl: **Nests** → your nest → **Eggs** → **Import**.
2. Use the egg JSON from the repo: **processor-voicesentinel/egg-voice-sentinel-processor.json**.
3. Create a new server (or reinstall an existing one) and select the **VoiceSentinel Processor** egg and the **Python 3.11** Docker image.

## After the server is created

1. Run **Install** once. Wait for it to finish (it installs system packages, clones, copies files, and sets up the venv).
2. Edit **config.json** in the server root: set **server.server_key** (and optionally port, model, etc.). Use the same **server_key** in the plugin’s config.yml.
3. Start the server. The first start can take a while while pip installs and the Whisper model downloads. Later starts are faster.

## Startup command

The egg’s startup command does not use **cd /mnt/server** or any path like **voicesentinel/processor-voicesentinel**. It runs in whatever directory the panel uses as the server root (often the container home). If you ever see “No such file or directory” for a path, you’re likely using an old or edited startup – replace it with the one from the current egg so it matches.

**Already running an older egg?** See [Updating the egg](updating-the-egg.md) to update the egg, reinstall the server, and switch to the new startup command.

## Port and connection

The panel assigns a port (e.g. 28472 or something else). In the plugin set **processor_websocket_url** to that host and port, e.g. `ws://processor.yourdomain.com:28472`, and use the same **server_key** in both configs.
