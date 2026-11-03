---
description: >-
  Where VoiceSentinel keeps mute history, strikes and voice reports: SQLite
  (default) or MySQL/MariaDB shared between servers.
---

# Storage

The `storage` block sets where mute history, strikes, the mute ladder count and the voice report log are kept.

```yaml
storage:
  type: SQLITE
  host: "localhost"
  port: 3306
  database: "voicesentinel"
  username: ""
  password: ""
  ssl: false  # true = encrypt the connection (certificate is not verified)
```

Restart the server after you change this. A reload is not enough.

## SQLite (default)

`type: SQLITE` keeps local files in the plugin folder:

| File | Content |
|------|---------|
| `plugins/VoiceSentinel/data/mute_history.db` | Mute history, strikes |
| `plugins/VoiceSentinel/data/voice_reports.db` | Voice report log |

Nothing else to set up. Each server has its own data.

## MySQL / MariaDB

`type: MYSQL` stores the same data in a MySQL or MariaDB database. Several servers can share one database. `MARIADB` is accepted as the same value.

1. Create an empty database and a user with full rights on it.
2. Fill in `host`, `port`, `database`, `username` and `password`.
3. Set `type: MYSQL` and restart.

The plugin creates its tables. `host`, `database` and `username` must not be empty, or the plugin warns and turns mute history and voice reports off.

Set `ssl: true` to encrypt the connection. The certificate is not verified.

Web dashboard accounts (`web/web_auth.db`) and consent answers (`consent.properties`) are always local files. They are not stored in the database.

## Network strikes

With a shared database, `moderation.network_strikes: true` counts a player's strikes across every server that uses it. Give every server its own `network_sync.server_id` (or `server_name`). See [Moderation and escalation](moderation-and-escalation.md#network-strikes).

## If the database is down

The plugin connects in the background and keeps retrying. Until it connects:

* strikes are counted in memory,
* ladder mutes use the default mute length,
* the report inbox is off.

Moderation does not wait for the database. A slow database is given up on after about two seconds.
