# Securing your processor

The processor speaks HTTP (for /health, /stats) and WebSocket (for the plugin). For production you should put it behind a reverse proxy with HTTPS so the plugin connects with **wss://** and traffic is encrypted.

## Nginx reverse proxy with SSL

Example: processor runs on the same machine as Nginx, listening on 28472. Nginx handles HTTPS and forwards to it.

Create a site config (e.g. `/etc/nginx/sites-available/voicesentinel-processor`):

```nginx
server {
    listen 80;
    server_name processor.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name processor.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/processor.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/processor.yourdomain.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://127.0.0.1:28472;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

The **Upgrade** and **Connection** headers are required for WebSocket. Without them the plugin’s WebSocket connection will fail.

Enable the site, test, reload Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/voicesentinel-processor /etc/nginx/sites-enabled/
nginx -t
sudo systemctl reload nginx
```

Then in the plugin set:

```yaml
processor_websocket_url: "wss://processor.yourdomain.com"
```

No port if you’re using 443. **server_key** stays the same in both plugin and processor config.

## SSL with Certbot

Get a certificate first (stop anything on port 80 if needed):

```bash
certbot certonly --standalone -d processor.yourdomain.com
```

Then point the Nginx config at the paths Certbot gives you (usually under `/etc/letsencrypt/live/...`). See [Certbot](certbot.md) for more.

## CORS

If you only use the processor from the Minecraft plugin (no browser calls to your processor URL), CORS is less critical. The default in config allows all origins. If you build a web dashboard that calls **/health** or **/stats** from a browser, set **cors.allow_origins** to your front-end URL (e.g. `["https://dashboard.yourdomain.com"]`) instead of `["*"]`.

## Firewall

Only expose 80/443 on the public interface. Let the processor listen on 127.0.0.1:28472 or a private IP so only Nginx can reach it.
