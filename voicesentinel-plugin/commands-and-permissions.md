# Commands and permissions

Main command: **/voicesentinel** (aliases: **/vs**, **/voicesent**).

## Commands

| Command | What it does |
|--------|----------------|
| **/voicesentinel reload** | Reloads config and messages from disk. Use after editing config.yml or messages.yml. Also applies changes to consent, Cloud and the On-Server engine. Storage changes need a restart. |
| **/voicesentinel stats** | Shows WebSocket connection status and basic processing stats. |
| **/voicesentinel vcmute &lt;player&gt; &lt;duration&gt; \[reason\]** | Voice-mutes a player. Duration examples: `30m`, `2h`, `1d`, or `perm` for a permanent mute. The optional reason is stored with the mute. |
| **/voicesentinel unvcmute &lt;player&gt;** | Removes their voice mute. |
| **/voicesentinel pardon &lt;player&gt;** | Unmutes the player, clears their strikes and resets their [mute ladder](moderation-and-escalation.md#mute-ladder) step. The mute is logged as overturned. |
| **/voicesentinel history &lt;player&gt;** | Shows a player's active mute, strikes, ladder step and recent mutes. |
| **/voicesentinel mutes** | Lists active voice mutes (up to 50). |
| **/voicesentinel discordtest all** | Sends a test payload to all Discord webhooks. Use `flag` or `mute` to test only one type. |
| **/voicesentinel updatecheck** | Checks whether a newer plugin build is available (admin). |
| **/voicesentinel test** | Setup checker (admin). Lists the speech mode, speech link, consent mode, license, voice chat mod, LuckPerms (only flagged when your settings need it), On-Server engine, Discord webhook and [data export](data-export.md), then how many problems it found. **`/voicesentinel test webhook`** sends a test Discord message. |
| **/voicesentinel license &lt;key&gt;** | Activates your license key without a restart and saves it to `config.yml`. Console or admin. Only works while the plugin has no valid key. See [Getting your license key](getting-your-license-key.md). |
| **/voicesentinel debug &lt;player&gt;** | Shows why an online player is or is not moderated: consent, bypass permission, voice mute (admin). |
| **/voicesentinel cloud** | Shows [VoiceSentinel Cloud](voicesentinel-cloud.md) status: enabled, entitled, last contact, queued incidents, last error (admin). |
| **/voicesentinel webuser list** | Console/RCON: list web dashboard accounts. |
| **/voicesentinel webuser delete &lt;username&gt;** | Console/RCON: remove a web account and its sessions. |
| **/voicesentinel webuser …** | Other subcommands: **`create`**, **`setpassword`**, **`setrole`**, **`setuplink`** (console/RCON only). See [Web dashboard](web-dashboard.md). |

### Player command

| Command | What it does |
|--------|----------------|
| **/voiceconsent** (alias **/vconsent**) | Opens the consent menu so a player can review or change their answer. In NOTICE mode it shows the join notice message. No permission needed. See [Consent](consent.md). |

### Voice reports

| Command | What it does |
|--------|----------------|
| **/reportvoice \[player\] \[minutes\]** (alias **/rvoice**) | Player reports recent voice context to staff (needs **`voicesentinel.report`**). May use a GUI if configured. |
| **/viewreport &lt;player&gt; \[minutes\]** (alias **/vr**) | Staff: opens a written book with report text (**`voicesentinel.report.review`**). |
| **/reportinbox** (aliases **/rinbox**, **/vreports**) | Staff: list reports / mark reviewed (**`voicesentinel.report.review`**). |

See [Voice reports & transcript buffer](voice-reports.md) for buffer settings and processor evidence.

### Naming a player

**vcmute**, **unvcmute**, **pardon** and **history** work on offline players too. Give the name of anyone the server has seen before, the name with or without the Bedrock (Floodgate) `.` prefix, or their UUID. A mute on an offline player applies when they join. **debug** needs the player online.

Reload and stats need **voicesentinel.reload** / **voicesentinel.stats**. Vcmute/unvcmute need **voicesentinel.vcmute**. Pardon, history and mutes need **voicesentinel.pardon**, **voicesentinel.history** and **voicesentinel.mutes**. Discordtest, updatecheck, test, debug, cloud, license, **webuser**, and the main command need **voicesentinel.admin**.

## Permissions

| Permission | Default | What it does |
|------------|--------|----------------|
| **voicesentinel.admin** | op | Access to admin commands and tab completion. Also grants `reload`, `stats`, `vcmute`, `pardon`, `history` and `mutes` below. |
| **voicesentinel.bypass** | false | No voice moderation: their audio isn’t sent, and they never get muted or trigger alerts/commands. |
| **voicesentinel.alerts** | op | Receive in-game alerts when content is flagged. |
| **voicesentinel.reload** | op | Use `/voicesentinel reload`. |
| **voicesentinel.stats** | op | Use `/voicesentinel stats`. |
| **voicesentinel.vcmute** | op | Use vcmute and unvcmute. |
| **voicesentinel.pardon** | op | Use `/voicesentinel pardon`. |
| **voicesentinel.history** | op | Use `/voicesentinel history`. |
| **voicesentinel.mutes** | op | Use `/voicesentinel mutes`. |
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
