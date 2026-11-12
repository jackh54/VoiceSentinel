---
description: >-
  Optionally copy mutes, strikes, reports, transcripts and flagged audio into
  your own MySQL or MariaDB database, for dashboards, ticket bots and appeal
  sites.
---

# Data export

VoiceSentinel can copy moderation data into a MySQL or MariaDB database of your choice. Your own tools can then read it: dashboards, ticket bots, appeal sites.

* It is **off by default**. Turn it on with `data_export.enabled: true`.
* It uses its **own connection**, separate from [`storage`](storage.md). You can give another service read-only access without touching the tables VoiceSentinel itself uses.
* VoiceSentinel **never reads this database**. It only writes to it. Deleting rows or the whole database does not change how the plugin behaves.
* Rows are written in the background. If the database is slow or down, voice moderation is not affected: rows wait in a queue and are written once it is back. If the queue fills up (10,000 rows), new rows are dropped.
* Only MySQL and MariaDB are supported. `type` accepts `MYSQL` or `MARIADB`.

## Privacy

{% hint style="warning" %}
You are responsible for the player data you keep. Check your privacy policy and local law before you turn on transcripts or audio.
{% endhint %}

* `respect_consent: true` (the default) never stores **transcripts or audio** for players who have not agreed to your voice rules (see [Consent](consent.md)). Mutes, strikes, reports and staff actions are still stored for them, because they record what staff and the plugin did.
* Audio is only ever stored for **flagged** speech, as 16 kHz mono WAV, at most 60 seconds per clip.
* `retention_days` (default `30`) deletes older rows every night at 04:00 server time and when the server starts. `0` keeps rows forever. Each server only deletes rows with its own `server_id`, so servers sharing one export database can use different retention.

## Configuration

```yaml
data_export:
  enabled: false
  type: MYSQL  # MYSQL works for MySQL and MariaDB
  host: "localhost"
  port: 3306
  database: "voicesentinel_export"
  username: ""
  password: ""
  ssl: false
  table_prefix: "vs_"
  store:
    transcripts: FLAGGED   # OFF, FLAGGED or ALL
    audio_clips: FLAGGED   # OFF or FLAGGED (16 kHz mono WAV of flagged speech only)
    mutes: true            # mute, unmute, extend and expiry events
    strikes: true
    reports: true          # player voice reports
    staff_actions: true    # what staff did from commands and the dashboard
  retention_days: 30       # rows older than this are deleted every night; 0 = keep forever
  respect_consent: true    # never store transcripts or audio for players who have not consented
```

| Key | Meaning |
|-----|---------|
| `enabled` | Turns the export on. Default `false` |
| `type` | `MYSQL` (also for MariaDB) or `MARIADB` |
| `host`, `port`, `database` | Where to connect. `host`, `database` and `username` must not be empty |
| `username`, `password` | The database user |
| `ssl` | `true` encrypts the connection |
| `table_prefix` | Prepended to every table name. Letters, digits and `_` only (other characters are removed, at most 32 characters) |
| `store.transcripts` | `OFF`, `FLAGGED` (flagged lines only) or `ALL` |
| `store.audio_clips` | `OFF` or `FLAGGED`. `ALL` is read as `FLAGGED` |
| `store.mutes` | Mute, unmute, extend, expiry and pardon events |
| `store.strikes` | Strikes |
| `store.reports` | Player voice reports |
| `store.staff_actions` | Staff actions from commands and the dashboard |
| `retention_days` | Delete rows older than this many days. `0` keeps them forever |
| `respect_consent` | Skip transcripts and audio for players who have not consented |

If the settings are not usable, the plugin warns in the console and exports nothing.

Create the database and a user first, for example:

```sql
CREATE DATABASE voicesentinel_export CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'vs_export'@'%' IDENTIFIED BY 'change-me';
GRANT CREATE, SELECT, INSERT, DELETE ON voicesentinel_export.* TO 'vs_export'@'%';

-- A read-only user for your own tools:
CREATE USER 'vs_reader'@'%' IDENTIFIED BY 'change-me-too';
GRANT SELECT ON voicesentinel_export.* TO 'vs_reader'@'%';
```

VoiceSentinel creates its tables when it starts.

## Reading the data

* Rows are only ever added (and deleted by retention). They are never updated.
* To pick up new rows, remember the highest `id` you have seen and poll `WHERE id > ?`.
* Every row also has a `uuid` that never changes. Use it as a stable event id.
* `server_id` says which server wrote the row, so one export database can serve a whole network. It is `network_sync.server_id` when that is set. Otherwise it is an id generated once and saved in `plugins/VoiceSentinel/data/export-server-id`.
* All times are **UTC**, with millisecond precision.
* Player names are the names at the time of the event. Use `player_uuid` to follow a player across name changes. It can be empty when the player was never seen on the server.

## Schema (version 1)

The tables use the default prefix `vs_`. `vs_schema_version` holds the schema version: read `MAX(version)`. A change to the columns raises it.

### Columns in every event table

| Column | Type | Notes |
|--------|------|-------|
| `id` | BIGINT UNSIGNED, auto increment | Primary key; grows with every row |
| `uuid` | CHAR(36) | Unique event id |
| `server_id` | VARCHAR(64) | Server that wrote the row |
| `created_at` | DATETIME(3) | When it happened (UTC) |
| `player_uuid` | CHAR(36), nullable | The player the row is about |
| `player_name` | VARCHAR(64), nullable | Their name at the time |

Every event table has a unique index on `uuid` and indexes on `(player_uuid, created_at)`, `created_at` and `(server_id, created_at)`. All tables use InnoDB and utf8mb4.

### `vs_schema_version`

| Column | Type | Notes |
|--------|------|-------|
| `version` | INT | Primary key |
| `applied_at` | DATETIME(3) | When this version was first created (UTC) |
| `plugin_version` | VARCHAR(32), nullable | VoiceSentinel version that created it |

### `vs_transcripts`

What players said. `store.transcripts: FLAGGED` keeps flagged lines only; `ALL` keeps every line.

| Column | Type | Notes |
|--------|------|-------|
| `text` | MEDIUMTEXT | The transcript |
| `flagged` | TINYINT(1) | 1 if it matched your word lists |
| `flagged_words` | TEXT, nullable | JSON array of matched words, e.g. `["word1","word2"]` (at most 50) |
| `severity` | VARCHAR(16), nullable | `mild`, `moderate` or `severe` |
| `confirmation` | VARCHAR(16), nullable | `confirmed`, `rejected` or `unchecked`, when hits are double-checked |
| `language` | VARCHAR(16), nullable | Detected language code |
| `engine` | VARCHAR(32), nullable | Which kind of processor produced it: `LOCAL`, `PUBLIC` or `CUSTOM` |
| `is_live` | TINYINT(1) | 1 for a live hit while the player was still talking, 0 for a finished sentence |
| `session_id` | VARCHAR(64), nullable | Voice session |
| `utterance_id` | VARCHAR(64), nullable | Live utterance id |
| `processing_ms` | INT, nullable | How long processing took |

### `vs_audio_clips`

Flagged speech only.

| Column | Type | Notes |
|--------|------|-------|
| `transcript_id` | BIGINT UNSIGNED, nullable | `vs_transcripts.id` of the same speech, when that transcript was stored |
| `transcript_uuid` | CHAR(36), nullable | `vs_transcripts.uuid` of the same speech |
| `format` | VARCHAR(8) | Always `wav` |
| `sample_rate` | INT | Always 16000 |
| `channels` | TINYINT | Always 1 |
| `duration_ms` | INT | Up to 60000 |
| `size_bytes` | INT | Size of `audio` |
| `sha256` | CHAR(64) | Hex SHA-256 of `audio` |
| `audio` | MEDIUMBLOB | A complete WAV file (16-bit PCM) |

### `vs_mute_events`

| Column | Type | Notes |
|--------|------|-------|
| `action` | VARCHAR(16) | `MUTE`, `UNMUTE`, `EXTEND`, `EXPIRE` or `PARDON` |
| `duration_ms` | BIGINT, nullable | Mute length; empty for permanent mutes and for unmutes |
| `expires_at` | DATETIME(3), nullable | When the mute ends (UTC) |
| `reason` | VARCHAR(512), nullable | |
| `source` | VARCHAR(32), nullable | Where it came from: `auto`, `command`, `cloud`, `network` or `web` |
| `actor_uuid` | CHAR(36), nullable | Staff member, if a player did it |
| `actor_name` | VARCHAR(64), nullable | |

### `vs_strikes`

| Column | Type | Notes |
|--------|------|-------|
| `tier` | VARCHAR(16) | `mild`, `moderate` or `severe` |
| `strike_number` | INT | The player's strikes in this tier inside the window, including this one |
| `window_ms` | BIGINT | Strike window |
| `action` | VARCHAR(16), nullable | What the escalation did: `REPORT`, `WARN` or `MUTE` |
| `reason` | VARCHAR(512), nullable | The flagged words |
| `transcript_uuid` | CHAR(36), nullable | `vs_transcripts.uuid` of the speech that caused it |

### `vs_reports`

Player voice reports. `player_uuid` and `player_name` are the **reported** player.

| Column | Type | Notes |
|--------|------|-------|
| `reporter_uuid` | CHAR(36), nullable | |
| `reporter_name` | VARCHAR(64), nullable | |
| `window_seconds` | INT | How far back the report looked |
| `evidence` | MEDIUMTEXT, nullable | The transcript lines staff were shown |
| `report_log_id` | BIGINT, nullable | Id in VoiceSentinel's own report log (`/reportinbox`) |

### `vs_staff_actions`

What staff did. `player_uuid` and `player_name` are the **target** player, when there is one.

| Column | Type | Notes |
|--------|------|-------|
| `actor_uuid` | CHAR(36), nullable | Empty for the console and dashboard users |
| `actor_name` | VARCHAR(64), nullable | |
| `origin` | VARCHAR(16) | `COMMAND`, `CONSOLE` or `WEB` |
| `action` | VARCHAR(64) | e.g. `mute`, `unmute`, `pardon`, `report_reviewed` |
| `detail` | VARCHAR(1024), nullable | |

## Example queries

Flagged transcripts in the last 24 hours, newest first:

```sql
SELECT created_at, server_id, player_name, text, flagged_words, severity
FROM vs_transcripts
WHERE flagged = 1
  AND created_at >= UTC_TIMESTAMP() - INTERVAL 24 HOUR
ORDER BY created_at DESC;
```

Mute history for a player:

```sql
SELECT created_at, action, duration_ms, expires_at, reason, source, actor_name
FROM vs_mute_events
WHERE player_uuid = '069a79f4-44e9-4726-a5be-fca90e38aaf5'
ORDER BY created_at DESC;
```

New rows since your last poll:

```sql
SELECT * FROM vs_mute_events WHERE id > 1234 ORDER BY id LIMIT 500;
```
