---
description: Update the VoiceSentinel Processor egg in Pterodactyl from older versions and apply the new startup command.
---

# Updating the egg

When a new version of the VoiceSentinel Processor egg is released (e.g. updated install script or startup command), use this process so your existing Pterodactyl server gets the new behavior.

## 1. Update the egg in Pterodactyl

1. Download or copy the latest egg JSON from the repo: **processor-voicesentinel/egg-voice-sentinel-processor.json** (from the [VoiceSentinel](https://github.com/jackh54/VoiceSentinel) repository).
2. In Pterodactyl: **Nests** → select the nest that has the VoiceSentinel Processor egg → **Eggs**.
3. Either **replace** the existing VoiceSentinel Processor egg with the new JSON (if your panel supports editing/re-importing the same egg), or **create a new egg** from the JSON and note its name. Exact steps depend on your Pterodactyl version; the goal is to have the updated egg available for servers.

## 2. Reinstall the processor server

After the egg is updated, **reinstall the processor server** so it uses the new install script and startup logic:

1. Open your **VoiceSentinel Processor** server in the panel.
2. Go to the **Settings** (or **Management**) tab.
3. Run **Reinstall** (or **Reinstall Server**). Confirm when prompted.
4. Wait for the reinstall to finish. The panel will run the **new** egg install script, so you get the updated file layout and behavior (e.g. no `voicesentinel` folder, correct paths).

{% hint style="warning" %}
Reinstalling **deletes** the server’s files and runs the install from scratch. **Back up config.json** (and any custom files) before reinstalling. After reinstall, upload or paste **config.json** back and set **server_key** and other options again.
{% endhint %}

## 3. Update the startup command

The egg’s startup command can change between versions (e.g. different paths or no `cd /mnt/server`). After reinstalling:

1. In the server’s **Startup** tab, check the **Startup Command**.
2. Replace it with the **startup command from the current egg**. You can copy it from the egg JSON (look for the `startup` field in **processor-voicesentinel/egg-voice-sentinel-processor.json**) or, if your panel has “Reset to default,” use that so the command matches the updated egg.
3. Save. Start the server and confirm it runs without path or “No such file or directory” errors.

If you keep an old or custom startup command, the server may point at wrong paths and fail to start. Always use the startup command that comes with the egg version you imported.

## Summary

1. **Update the egg** in your nest (re-import or replace with the latest egg JSON).
2. **Reinstall the processor server** so it uses the new install script; back up **config.json** first.
3. **Set the startup command** to the one from the updated egg (copy from egg JSON or reset to default).

Then restore **config.json**, set **server_key** and any other options, and start the server. For first-time setup, see [Pterodactyl Setup](pterodactyl-setup.md).
