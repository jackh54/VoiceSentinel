# Commands and permissions

Main command: **/voicesentinel** (aliases: **/vs**, **/voicesent**).

## Commands

| Command | What it does |
|--------|----------------|
| **/voicesentinel reload** | Reloads config and messages from disk. Use after editing config.yml or messages.yml. |
| **/voicesentinel stats** | Shows WebSocket connection status and basic processing stats. |
| **/voicesentinel vcmute &lt;player&gt; &lt;duration&gt;** | Voice-mutes a player. Duration examples: `30m`, `2h`, `1d`. |
| **/voicesentinel unvcmute &lt;player&gt;** | Removes their voice mute. |
| **/voicesentinel discordtest all** | Sends a test payload to all Discord webhooks. Use `flag` or `mute` to test only one type. |
| **/voicesentinel updatecheck** | Checks whether a newer plugin build is available (admin). |
| **/voicesentinel webuser list** | Console/RCON: list web dashboard accounts. |
| **/voicesentinel webuser delete &lt;username&gt;** | Console/RCON: remove a web account and its sessions. |
| **/voicesentinel webuser …** | Other subcommands: **`create`**, **`setpassword`**, **`setrole`**, **`setuplink`** (console/RCON only). See [Web dashboard](web-dashboard.md). |

### Voice reports

| Command | What it does |
|--------|----------------|
| **/reportvoice \[player\] \[minutes\]** (alias **/rvoice**) | Player reports recent voice context to staff (needs **`voicesentinel.report`**). May use a GUI if configured. |
| **/viewreport &lt;player&gt; \[minutes\]** (alias **/vr**) | Staff: opens a written book with report text (**`voicesentinel.report.review`**). |
| **/reportinbox** (aliases **/rinbox**, **/vreports**) | Staff: list reports / mark reviewed (**`voicesentinel.report.review`**). |

See [Voice reports & transcript buffer](voice-reports.md) for buffer settings and processor evidence.

Reload and stats need **voicesentinel.reload** / **voicesentinel.stats**. Vcmute/unvcmute need **voicesentinel.vcmute**. Discordtest, updatecheck, **webuser**, and the main command need **voicesentinel.admin** (or the right sub-permission).

## Permissions

| Permission | Default | What it does |
|------------|--------|----------------|
| **voicesentinel.admin** | op | Access to admin commands and tab completion. |
| **voicesentinel.bypass** | false | No voice moderation: their audio isn’t sent, and they never get muted or trigger alerts/commands. |
| **voicesentinel.alerts** | op | Receive in-game alerts when content is flagged. |
| **voicesentinel.reload** | op | Use `/voicesentinel reload`. |
| **voicesentinel.stats** | op | Use `/voicesentinel stats`. |
| **voicesentinel.vcmute** | op | Use vcmute and unvcmute. |
| **voicesentinel.report** | true | Use **`/reportvoice`** to submit a report. |
| **voicesentinel.report.review** | op | Receive report alerts, **`/viewreport`**, **`/reportinbox`**. |

Example (LuckPerms): give someone bypass so they’re never checked:

```text
/lp user TheirName permission set voicesentinel.bypass true
```

Give a staff role alerts:

```text
/lp group moderator permission set voicesentinel.alerts true
```
