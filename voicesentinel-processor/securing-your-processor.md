---
description: >-
  This page goes over: setting up SSL using a reverse proxy, and proper cors
  configuration
---

# Securing your processor

### NGINX Reverse Proxy with SSL

```bash
sudo nano /etc/nginx/sites-available/<your-transcriber-url>
```

Here is a sample NGINX configuration for a reverse proxy, be sure to replace \<yourdomain> and \<yourip>.

```nginx
server {
    listen 80;
    server_name <yourdomain>;

    # Redirect all HTTP requests to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name <yourdomain>;

    # Path to your SSL certificate and key
    ssl_certificate /etc/letsencrypt/live/<yourdomain>/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/<yourdomain>/privkey.pem;

    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Proxy settings
    location / {
        proxy_pass http://<yourip>:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}


```

### Symlink the file

```bash
sudo ln -s /etc/nginx/sites-available/<your-transcriber-url> /etc/nginx/sites-enabled/
```

### Cors Configuration

VoiceSentinel includes all necessary methods and headers in the `config.example.json` file. Simply add your transcriber URL!&#x20;

{% hint style="warning" %}
Make sure to include http/https.
{% endhint %}

```json
"cors": {
  "allow_origins": ["<your-transcriber-url>"],
  "allow_credentials": true,
  "allow_methods": ["GET", "POST", "DELETE"],
  "allow_headers": [
    "Content-Type", 
    "Authorization", 
    "Accept",
    "User-Agent",
    "X-Requested-With"
  ]
}
```
