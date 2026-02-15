---
description: This assumes you have docker & docker compose already installed
---

# Running the processor

{% hint style="info" %}
Start the backend processor in the background using Docker Compose
{% endhint %}

```bash
docker compose up -d
```

{% hint style="info" %}
To view logs and/or check stats
{% endhint %}

```bash
# Logs:
docker compose logs -f

# Stats:
docker stats voicesentinel-processor
```

{% hint style="info" %}
To stop the processor
{% endhint %}

```bash
docker compose down
```
