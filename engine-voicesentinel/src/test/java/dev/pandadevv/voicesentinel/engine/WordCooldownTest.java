package dev.pandadevv.voicesentinel.engine;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class WordCooldownTest {
    @Test
    void secondHitInsideTheWindowIsBlocked() {
        WordCooldown cooldown = new WordCooldown(2000);
        assertTrue(cooldown.tryAcquire("fuck", 1000));
        assertFalse(cooldown.tryAcquire("fuck", 2999));
    }

    @Test
    void wordIsAllowedAgainAfterTheWindow() {
        WordCooldown cooldown = new WordCooldown(2000);
        assertTrue(cooldown.tryAcquire("fuck", 1000));
        assertTrue(cooldown.tryAcquire("fuck", 3000));
    }

    @Test
    void wordsCoolDownIndependently() {
        WordCooldown cooldown = new WordCooldown(2000);
        assertTrue(cooldown.tryAcquire("fuck", 1000));
        assertTrue(cooldown.tryAcquire("shit", 1100));
        assertFalse(cooldown.tryAcquire("shit", 1200));
    }
}
