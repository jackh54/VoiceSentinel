# VoiceSentinel Processor on Cloudflare Containers

Optional managed deployment for the VoiceSentinel processor. Self-hosting with Docker Compose remains the default path and is unchanged.

## Prerequisites

- Cloudflare account with Containers enabled (beta)
- [Wrangler](https://developers.cloudflare.com/workers/wrangler/install-and-update/) v4+
- Docker (for local image builds; `linux/amd64`)
- Node.js 18+

## Architecture

- A Worker routes all HTTP/WebSocket traffic to a **single warm container** instance (`getContainer(env.PROCESSOR, "main")`).
- The container runs the same FastAPI app as the Docker self-hosted path.
- Configuration is injected via Worker secrets and environment variables (no `config.json` in the container).
- Recordings and report-buffer data use **R2** when credentials are configured; otherwise the container uses ephemeral local disk (not recommended for production).

## Instance sizing

Default `wrangler.jsonc` uses `standard-4` (4 vCPU, 12 GiB RAM) for `compute_type: int8` Whisper workloads.

To scale up, replace `instance_type` with a custom block:

```jsonc
"instance_type": {
  "vcpu": 4,
  "memory_mib": 12288,
  "disk_mib": 20480
}
```

Custom types must satisfy Cloudflare limits (minimum 3 GiB memory per vCPU, max 4 vCPU / 12 GiB memory / 20 GB disk per instance).

## R2 setup

1. Create an R2 bucket (default name in `wrangler.jsonc`: `voicesentinel-processor`).
2. Create an R2 API token with read/write access to that bucket.
3. Set secrets on the Worker (see below). The Worker forwards R2 S3 credentials into the container as `VOICESENTINEL_R2_BUCKET`, `R2_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`.

Update `r2_buckets` and `vars.R2_BUCKET_NAME` in `wrangler.jsonc` if you use a different bucket name.

## Secrets and configuration

Set Worker secrets before deploy. These are forwarded into the container as environment variables for the shared config loader (`env > defaults`; no `config.json` on Cloudflare).

Required:

```bash
cd processor-voicesentinel/cloudflare
wrangler secret put SERVER_KEY
```

Recommended for public pool mode (matches README public-pool design):

```bash
wrangler secret put POOL_SERVER
# value: true
```

R2 (when using persistent recordings / report buffer):

```bash
wrangler secret put R2_ACCOUNT_ID
wrangler secret put R2_ACCESS_KEY_ID
wrangler secret put R2_SECRET_ACCESS_KEY
```

Optional tuning (same names as processor env overrides):

```bash
wrangler secret put TRANSCRIPTION_MODEL
wrangler secret put REPORT_BUFFER_ENABLED
wrangler secret put REPORT_BUFFER_SAVE_AUDIO
```

See `app/config_loader.py` for the full list of supported `SERVER_*`, `TRANSCRIPTION_*`, `RECORDINGS_*`, and `REPORT_BUFFER_*` environment variables.

## Build image with pre-baked Whisper model

Cloudflare container disk is ephemeral. Pre-bake the Whisper model at build time so cold starts do not download weights at runtime.

Self-hosted Docker builds are unaffected (`PREBAKE_WHISPER_MODEL` defaults to `false` in the root `Dockerfile`).

```bash
cd processor-voicesentinel/cloudflare
chmod +x build-image.sh
./build-image.sh
```

`wrangler deploy` builds from `../Dockerfile` automatically. For production, run `build-image.sh` first or ensure your deploy pipeline passes:

```bash
--build-arg PREBAKE_WHISPER_MODEL=true
--build-arg WHISPER_MODEL=Systran/faster-whisper-base
--platform linux/amd64
```

## Local development

```bash
cd processor-voicesentinel/cloudflare
npm install
wrangler dev
```

Then check health (Wrangler prints the local URL):

```bash
curl -sS "$WRANGLER_DEV_URL/health"
```

## Deploy

```bash
cd processor-voicesentinel/cloudflare
npm install
wrangler deploy
```

First deploy can take several minutes while the container image is provisioned globally.

## Cold start and keep-warm

Whisper model load is expensive. The Worker uses a singleton container (`"main"`) and `sleepAfter = "24h"` so the instance stays warm across idle periods instead of scaling to zero after every request.

If the container does stop (platform maintenance, long idle beyond `sleepAfter`, or deploy rollout), the next request pays a cold-start cost (container boot + model load unless pre-baked). Plan capacity accordingly; autoscaling beyond this single instance is out of scope for this deployment path.

## Plugin connection

Point the VoiceSentinel plugin processor URL at your Worker hostname (e.g. `wss://voicesentinel-processor.<your-subdomain>.workers.dev/ws/...`). Use the same `server_key` / pool auth values as documented in the main README.
