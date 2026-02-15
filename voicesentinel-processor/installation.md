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

- Set **server.server_key** in `config.json` to a shared secret (16–256 alphanumeric). Use the **exact same** value in the plugin’s **server_key** in config.yml. If they don’t match, the plugin won’t authenticate.
- Open the port (e.g. 28472) on the host/firewall if the Minecraft server is on another machine.
- For production, put the processor behind a reverse proxy with HTTPS and use `wss://` in the plugin; see [Securing your processor](securing-your-processor.md).

{% hint style="success" %}
Once the processor is running and the plugin has the same **server_key** and correct **processor_websocket_url**, the plugin should connect. Check with `/voicesentinel stats` on the server.
{% endhint %}
