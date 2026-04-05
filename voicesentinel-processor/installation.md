---
description: Install the VoiceSentinel processor manually, with Docker, or via Pterodactyl.
---

# Installation

You need Python 3.8+ (3.11 is a good choice), enough RAM (4GB minimum, 8GB+ better for heavier models), and a multi-core CPU. The processor runs as a standalone service: the Minecraft plugin connects to it over WebSocket and sends audio; the processor transcribes and checks word lists, then sends results back.

{% hint style="danger" %}
Basic Linux/Docker and server admin experience will help. If you’re not comfortable with that, get someone who is to set it up.
{% endhint %}

Pick one of the options below:

{% tabs %}
{% tab title="Manual (no Docker)" %}
Clone the repo and use the processor folder:

```bash
git clone https://github.com/jackh54/VoiceSentinel.git
cd VoiceSentinel/processor-voicesentinel
```

Copy the example config and edit it (at least **server_key** and optionally **port**):

```bash
cp config.example.json config.json
# Edit config.json – see Configuration
```

Use a virtualenv so you don’t pollute the system Python:

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -U pip setuptools wheel
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Run it (default port 28472):

```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 28472
```

Leave that running or run it under something like systemd/screen. The plugin will connect to `ws://this-machine:28472` (or whatever host/port you use).
{% endtab %}

{% tab title="Docker" %}
From the processor folder (`VoiceSentinel/processor-voicesentinel`):

```bash
cp config.example.json config.json
# Edit config.json as needed

docker compose up -d
```

That builds and starts the container. Port 28472 is exposed by default. Mount `config.json` (and optionally a recordings folder) if you want to persist config; see the compose file.
{% endtab %}

{% tab title="Pterodactyl" %}
Use the VoiceSentinel Processor egg so the panel runs the processor for you. Full steps: [Pterodactyl Setup](pterodactyl-setup.md).
{% endtab %}
{% endtabs %}

## After install

- **Self-hosted (plugin CUSTOM mode):** Set **`server.server_key`** in **`config.json`** and the **same** value as **`server_key`** in the plugin’s **`config.yml`** (16–256 alphanumeric). Mismatch → auth failure in processor logs.  
- **PUBLIC pool:** The Minecraft plugin uses **`processor_connection_mode: PUBLIC`**; you do **not** point it at your own **`config.json`** — follow license / pool setup instead.  
- Open the processor port on the firewall if the game server is remote.  
- For production, use a reverse proxy with **HTTPS** and **`wss://`**; see [Securing your processor](securing-your-processor.md).

{% hint style="success" %}
**CUSTOM:** When **`server_key`** and **`processor_websocket_url`** match your deployment, **`/voicesentinel stats`** on the server should show a healthy processor connection.
{% endhint %}
