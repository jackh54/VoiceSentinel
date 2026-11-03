package dev.pandadevv.voicesentinel.engine;

import java.util.HashMap;
import java.util.Map;

/** Lets each word through at most once per window. Times are audio-clock milliseconds. */
final class WordCooldown {
    private final long windowMs;
    private final Map<String, Long> last = new HashMap<>();

    WordCooldown(long windowMs) {
        this.windowMs = windowMs;
    }

    /** True and starts the window when the word is not cooling down. */
    boolean tryAcquire(String word, long nowMs) {
        Long previous = last.get(word);
        if (previous != null && nowMs - previous < windowMs) {
            return false;
        }
        last.put(word, nowMs);
        return true;
    }
}
