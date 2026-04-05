---
description: Transcript buffer, /reportvoice, /viewreport, /reportinbox, and optional processor-side report evidence storage.
---

# Voice reports & transcript buffer

## Transcript buffer (`transcript_buffer`)

The plugin keeps a **rolling in-memory buffer** of final transcripts per player. This is controlled separately from the player-facing **`/reportvoice`** command:

* Turn **`transcript_buffer.enabled`** on to collect lines for **staff review in the web dashboard** and for reports—even if **`voice_report.enabled`** is `false`.
* **`/reportvoice`** uses this same buffer for transcript text (when the buffer is enabled).

Typical options:

```yaml
transcript_buffer:
  enabled: true
  retention_seconds: 43200   # how long lines are kept
  max_lines_per_player: 500
```

## Voice report (`voice_report`)

When **`voice_report.enabled`** is `true`, players with **`voicesentinel.report`** can use **`/reportvoice`** (or a chest GUI if **`voice_report.interface`** is `gui`) to flag recent voice context for staff.

Staff with **`voicesentinel.report.review`** receive the alert and can use **`/viewreport`** to open a **written book** with the same payload. **`/reportinbox`** lists pending reports and can mark them reviewed (aligned with the web dashboard inbox).

Optional **`voice_report.staff_chat_full_transcript`** dumps the full text in chat as well (can be noisy).

## Processor: report evidence (`report_buffer`)

If you want the plugin to pull **extra evidence** from the processor over HTTP, enable the processor’s **`report_buffer`** block in **`config.json`**. When it is **disabled**, evidence requests return **404** (the book and dashboard still use the local buffer).

When enabled, stored paths are scoped by license and server key; use **HTTPS** on public processors. See [Processor configuration](../voicesentinel-processor/configuration.md#report-buffer-optional) for the JSON keys.

## Permissions

| Permission | Typical use |
|------------|-------------|
| **`voicesentinel.report`** | Players: submit **`/reportvoice`**. |
| **`voicesentinel.report.review`** | Staff: alerts, **`/viewreport`**, **`/reportinbox`**. |

## See also

* [Web dashboard](web-dashboard.md)
* [Commands and permissions](commands-and-permissions.md)
