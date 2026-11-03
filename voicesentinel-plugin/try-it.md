---
description: Try VoiceSentinel on the public demo server.
---

# Try it

You can hear VoiceSentinel work on the public demo server, `try.pandascript.dev`.

1. Install Simple Voice Chat.
2. Join `try.pandascript.dev`.
3. Say a test word in a sentence.

| Word | Tier | What happens |
|------|------|--------------|
| `banana` | severe | You are muted for 1 minute. |
| `pineapple` | moderate | First time: a warning. Second time: muted for 1 minute. |
| `potato` | mild | First time: staff are alerted. Second: a warning. Third: muted for 1 minute. |

These words work on that server only. They are not in the default word list. Strikes reset after 3 minutes.

The demo runs [On-Server Mode](on-server-mode.md) with live detection, so a mute word is caught while you are still talking. See [Moderation and escalation](moderation-and-escalation.md) for how the steps work.
