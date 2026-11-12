# Getting your license key

License keys are issued through [key.pandascript.dev](https://key.pandascript.dev). Every mode needs a valid license. If you use the **PUBLIC** processor pool, keep the plugin up to date. For help, use the project [Discord](https://discord.gg/JAJyuzdgHZ).

Open **`plugins/VoiceSentinel/config.yml`** and set:

```yaml
license-key: "your-actual-key-here"
```

Save the file, and continue configuring.

You can also start the server without a key. VoiceSentinel stays loaded but does not moderate. Run **`/voicesentinel license <your-key>`** from the console or as an admin. The key is checked, saved to `config.yml` and moderation starts with no restart. To change a key that is already active, edit `config.yml` and restart.
