# Installation

{% hint style="danger" %}
You should have basic Linux/Docker knowledge and common sense. If you're lacking these, consult someone who does.
{% endhint %}

### Cloning the processor files

```bash
git clone https://github.com/jackh54/VoiceSentinel.git
cd VoiceSentinel
```

### Installing requirements

{% hint style="info" %}
This guide presumes installation on an Ubuntu Server, but it's also compatible with other operating systems. Make sure you know how to install it for your specific OS.
{% endhint %}

```bash
sudo apt install nginx certbot
```

```bash
curl -sSL https://get.docker.com/ | CHANNEL=stable bash
```

> This command is a quick way to install docker, if you'd like to do a manual installation see [Docker's Installation Documentation](https://docs.docker.com/engine/install/).
>
> Please note you might already have Docker installed, check with `docker -v`

Docker compose should automatically be installed with Docker, check with `docker compose version`
