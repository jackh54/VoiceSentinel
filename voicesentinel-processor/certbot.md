# Certbot

Run the following command to create your SSL certificates

{% hint style="warning" %}
First, stop the nginx service if it's running, or anything running on port 80.
{% endhint %}

```bash
certbot certonly --standalone -d <yourdomain>
```

Now, test your nginx configuration

```bash
nginx -t
```

If all is good, restart nginx:

```bash
sudo systemctl restart nginx
```

