---
description: >-
  Listen to VoiceSentinel events from your own plugin: flags, voice mutes,
  unmutes and strikes. Event classes, getters, threads and build setup.
---

# Developer API

VoiceSentinel fires four Bukkit events that other plugins can listen to. Use them to log to your own system, reward or punish players, or cancel a detection.

The classes live in `dev.pandadevv.voicesentinel.api` and are kept readable in the release jar, so you can compile against the VoiceSentinel jar itself.

## Add VoiceSentinel to your project

There is no Maven repository. Compile against the jar you downloaded, and do not shade it into your plugin.

**Gradle**

```groovy
dependencies {
    compileOnly files("libs/VoiceSentinel.jar")
}
```

**Maven** (system scope)

```xml
<dependency>
    <groupId>dev.pandadevv</groupId>
    <artifactId>voicesentinel</artifactId>
    <version>3.10.1</version>
    <scope>system</scope>
    <systemPath>${project.basedir}/libs/VoiceSentinel.jar</systemPath>
</dependency>
```

**Maven** (install into your local repository once, then use `<scope>provided</scope>` and drop `systemPath`)

```bash
mvn install:install-file -Dfile=VoiceSentinel.jar -DgroupId=dev.pandadevv \
  -DartifactId=voicesentinel -Dversion=3.10.1 -Dpackaging=jar
```

In your `plugin.yml`, name VoiceSentinel as a dependency. The plugin's name is `VoiceSentinel`:

```yaml
depend: [VoiceSentinel]       # your plugin needs it
# or
softdepend: [VoiceSentinel]   # your plugin works without it
```

If you use `softdepend`, only touch the event classes after checking that VoiceSentinel is enabled, or the class loading will fail without it.

VoiceSentinel supports Folia (`folia-supported: true`), so your listeners may run on a region thread. See [Threads](developer-api.md#threads).

## Example

```java
import dev.pandadevv.voicesentinel.api.MuteSource;
import dev.pandadevv.voicesentinel.api.VoiceFlagEvent;
import dev.pandadevv.voicesentinel.api.VoiceMuteEvent;
import dev.pandadevv.voicesentinel.api.VoiceStrikeEvent;
import dev.pandadevv.voicesentinel.api.VoiceUnmuteEvent;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;

public class VoiceListener implements Listener {

    @EventHandler(priority = EventPriority.HIGH, ignoreCancelled = true)
    public void onFlag(VoiceFlagEvent event) {
        if (event.getPlayer().hasPermission("myplugin.voice.exempt")) {
            event.setCancelled(true); // VoiceSentinel ignores this detection
        }
    }

    @EventHandler
    public void onMute(VoiceMuteEvent event) {
        String until = event.getSource() == MuteSource.AUTO ? "auto (mute ladder)"
                : event.getDuration() == null ? "permanent" : event.getDuration().toString();
        System.out.println(event.getPlayerName() + " muted (" + event.getSource() + "): " + until);
    }

    @EventHandler
    public void onUnmute(VoiceUnmuteEvent event) {
        System.out.println(event.getPlayerName() + " unmuted: " + event.getCause());
    }

    @EventHandler
    public void onStrike(VoiceStrikeEvent event) {
        System.out.println(event.getPlayerId() + " strike " + event.getStrike() + " (" + event.getTier() + ")");
    }
}
```

Register it as usual with `getServer().getPluginManager().registerEvents(new VoiceListener(), this);`.

## Threads

| Event | Fired on |
|-------|----------|
| `VoiceFlagEvent` | The player's own thread: the main thread on Paper, the region thread that owns the player on Folia |
| `VoiceMuteEvent` | Player online: the player's own thread. Player not online on this server (non-automatic mutes only): an asynchronous thread |
| `VoiceUnmuteEvent` | Player online: the player's own thread, including when a timed mute runs out. Lifted or pardoned while the player is not online on this server: an asynchronous thread. A mute that runs out while the player is offline fires no event |
| `VoiceStrikeEvent` | An asynchronous thread |

Each event reports this through the standard Bukkit `event.isAsynchronous()`. Check it before you use the Bukkit API. On an asynchronous thread, do not touch worlds, entities or inventories directly. Hand the work to the scheduler instead (on Folia, the entity or global scheduler).

A `VoiceFlagEvent` is skipped for players with the `voicesentinel.bypass` permission. If a player leaves at the moment of a flag, the flag still counts but the event is not fired.

## VoiceFlagEvent

Fired when speech is flagged, before any alert, strike or mute. Cancellable: cancel it to ignore the detection.

| Method | Returns | Meaning |
|--------|---------|---------|
| `getPlayer()` | `Player` | The speaker. Never null, always online |
| `getTranscript()` | `String` | What was said |
| `getWords()` | `List<String>` | The words that matched. Read-only copy |
| `willMute()` | `boolean` | `true` if this detection is set to lead to a mute |
| `isCancelled()` / `setCancelled(boolean)` | `boolean` | Cancel to drop the detection: no alert, strike or mute follows |

## VoiceMuteEvent

Fired after a player is voice muted: automatically for flagged speech, or by staff, [VoiceSentinel Cloud](voicesentinel-cloud.md), the network or the dashboard. Not cancellable. Automatic mutes fire this event only if the player is still online on this server when the mute lands.

| Method | Returns | Meaning |
|--------|---------|---------|
| `getPlayer()` | `Player` | The player. Null only for a non-automatic mute of a player who is not online on this server |
| `getPlayerId()` | `UUID` | The player's UUID. Never null |
| `getPlayerName()` | `String` | The player's name |
| `getTranscript()` | `String` | The flagged speech. Null for mutes that were not automatic |
| `getWords()` | `List<String>` | The matched words. Empty for mutes that were not automatic |
| `getDuration()` | `Duration` | Length of the mute. For a Cloud Extend, the time added, not the new total. Null when permanent, and for automatic mutes (not known to this event) |
| `getReason()` | `String` | The reason. Nullable. Null for automatic mutes |
| `getSource()` | `MuteSource` | Where the mute came from |

`MuteSource` values:

| Value | Meaning |
|-------|---------|
| `AUTO` | VoiceSentinel muted the player for flagged speech |
| `MANUAL` | A staff member used `/vs vcmute` |
| `CLOUD` | A VoiceSentinel Cloud command |
| `NETWORK` | Another server on the network (network sync) |
| `DASHBOARD` | The web dashboard |

## VoiceUnmuteEvent

Fired after a voice mute ends, whether it was lifted, pardoned or ran out. Not cancellable.

| Method | Returns | Meaning |
|--------|---------|---------|
| `getPlayer()` | `Player` | The player. Null when they are not online on this server |
| `getPlayerId()` | `UUID` | The player's UUID |
| `getPlayerName()` | `String` | The player's name |
| `getCause()` | `VoiceUnmuteEvent.Cause` | Why the mute ended |

`Cause` values: `MANUAL`, `CLOUD`, `NETWORK`, `DASHBOARD`, `EXPIRED` (a timed mute ran out) and `PARDON`.

## VoiceStrikeEvent

Fired after VoiceSentinel records a strike for flagged speech. Not cancellable.

| Method | Returns | Meaning |
|--------|---------|---------|
| `getPlayer()` | `Player` | The player. Null when they are not online on this server |
| `getPlayerId()` | `UUID` | The player's UUID |
| `getTier()` | `String` | `mild`, `moderate` or `severe` |
| `getStrike()` | `int` | The player's strike count for this tier inside the strike window, including this one |

See [Moderation and escalation](moderation-and-escalation.md) for tiers and the strike window.
