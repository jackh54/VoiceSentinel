---
description: Transcript buffer, voice_report options, /reportvoice, /viewreport, /reportinbox, and optional processor report_buffer.
---

# Voice reports & transcript buffer

## Transcript buffer (`transcript_buffer`)

Rolling **in-memory** buffer of final transcripts per player.

* Powers **moderation / transcript** views in the **web dashboard** when enabled.  
* Supplies text for **`/reportvoice`** when the buffer is on — **independent** of **`voice_report.enabled`** (that flag only controls whether players may **submit** reports).

```yaml
transcript_buffer:
  enabled: true
  retention_seconds: 43200
  max_lines_per_player: 500
```

## Voice report (`voice_report`)

| Option | Purpose |
|--------|---------|
| **enabled** | Allows players with **`voicesentinel.report`** to use **`/reportvoice`** (or the GUI). |
| **interface** | **`command`** or **`gui`**. |
| **default_report_window_seconds** | Default lookback if not specified. |
| **min_report_window_seconds** / **max_report_window_seconds** | Clamp window length. |
| **report_window_options_seconds** | GUI preset lengths (seconds). |
| **processor_evidence_url** | Base URL for **`GET /report/evidence`** on the processor. Empty = derived from WebSocket URL. |
| **webhook_url** | Optional Discord notification with the same summary as staff. |
| **webhook_username**, **webhook_embed_color**, **webhook_timeout_seconds** | Discord payload tuning. |
| **staff_chat_full_transcript** | If `true`, staff also get full plain text in chat (noisy). |

Staff with **`voicesentinel.report.review`** get alerts, can use **`/viewreport`** (written book), and **`/reportinbox`** (list / mark reviewed — same logical inbox as the web dashboard).

## Processor: `report_buffer`

Optional **on-disk** evidence on the processor for the HTTP evidence API. If **`report_buffer.enabled`** is **`false`**, evidence calls return **404**; the book and dashboard still use the **local** transcript buffer.

Scoped by license and server key; use **HTTPS** on public hosts. Details: [Processor configuration](../voicesentinel-processor/configuration.md#report-buffer-optional).

## Permissions

| Permission | Use |
|------------|-----|
| **voicesentinel.report** | **`/reportvoice`** |
| **voicesentinel.report.review** | Alerts, **`/viewreport`**, **`/reportinbox`** |

## See also

* [Web dashboard](web-dashboard.md)  
* [Commands and permissions](commands-and-permissions.md)  
* [Configuration](configuration.md)
