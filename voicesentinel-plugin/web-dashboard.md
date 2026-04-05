---
description: Embedded web console — login, roles, first-time tour, activity log (superuser), and console-only web account management.
---

# Web dashboard

The plugin can serve a small **operator console** in the browser: live status, sessions, moderation (reports, mutes, transcripts), and statistics. Configuration is under **`web_dashboard`** in `config.yml`.

{% hint style="warning" %}
Bind to **`127.0.0.1`** unless you know what you’re doing. For remote access, use a **reverse proxy with HTTPS** and tighten **`ip_allowlist`**.
{% endhint %}

## Enabling it

```yaml
web_dashboard:
  enabled: true
  bind_address: "127.0.0.1"
  port: 8124
  path_prefix: "/voicesentinel/web"
  session_minutes: 480
  max_login_attempts_per_minute: 20
  setup_token_minutes: 15
  ip_allowlist: []   # [] or ["*"] = any IP; or list specific IPs
```

Open **`http://<bind>:<port><path_prefix>/`** (e.g. `http://127.0.0.1:8124/voicesentinel/web/`). If you use a proxy, use your public URL instead.

## Accounts and roles

* **Moderator** — can use the console for monitoring and moderation.
* **Superuser** — same as moderator, plus **Config** (word lists from the browser) and **Activity log** (audit of web sign-ins and dashboard actions). Legacy **`operator`** accounts in the database are treated like superusers in the UI.

Accounts are stored in **`plugins/VoiceSentinel/web/web_auth.db`**. Passwords are set with a one-time link or directly from the server console (see below).

## First-time tour

The first time a web user signs in, an optional **guided tour** runs once and is remembered for that account (moderators and superusers).

## Activity log (superuser only)

The **Activity log** tab lists recent events across **all** web users, for example: sign-in / sign-out, finishing the tour, marking a voice report reviewed, word list changes from the browser, and voice mute / unmute actions done through the web UI.

## Managing web users (console / RCON only)

These subcommands are **not** for in-game players; run them from the **server console** or **RCON** (permission **`voicesentinel.admin`**):

| Subcommand | Purpose |
|------------|---------|
| **`webuser list`** | Lists all dashboard accounts, role, and password state. |
| **`webuser create <name> [moderator\|superuser]`** | Creates an account (no password until you set one). |
| **`webuser delete <username>`** | Deletes the account and clears its active sessions and setup tokens. |
| **`webuser setpassword <name> <password>`** | Sets password (may appear in logs — prefer **`setuplink`** when possible). |
| **`webuser setrole <name> <moderator\|superuser>`** | Changes role. |
| **`webuser setuplink <name>`** | Prints a one-time URL to set a password in the browser. |

Example:

```text
voicesentinel webuser create Alice moderator
voicesentinel webuser setuplink Alice
```

## See also

* [Voice reports & transcript buffer](voice-reports.md) — how `/reportvoice` and the dashboard use the transcript buffer.
* [Configuration](configuration.md) — full `config.yml` overview.
