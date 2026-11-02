package dev.pandadevv.voicesentinel.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;

import java.util.List;

import org.junit.jupiter.api.Test;

class ModeratorTest {
    private static final Moderator.Words WORDS = new Moderator.Words(List.of("loser"), List.of("slur"));

    private static JsonObject run(String first, String second) {
        return Moderator.result(first, WORDS, false, false, second == null ? null : v -> second, "p", "s", "en");
    }

    @Test
    void confirmedHitStaysFlagged() {
        JsonObject out = run("you loser", "such a loser");
        assertTrue(out.get("flagged").getAsBoolean());
        assertEquals("confirmed", out.get("double_check").getAsString());
        assertEquals("confirmed", out.get("confirmation").getAsString());
    }

    @Test
    void secondModelDisagreeingStillEmitsTheHitMarkedRejected() {
        JsonObject out = run("you loser", "you looser");
        assertTrue(out.get("flagged").getAsBoolean());
        assertEquals("rejected", out.get("confirmation").getAsString());
        assertEquals("rejected", out.get("double_check").getAsString());
        assertEquals("loser", out.getAsJsonArray("bad_words").get(0).getAsString());
    }

    @Test
    void rejectedMuteWordIsReportedButNeverShouldMute() {
        JsonObject out = run("a slur here", "nothing");
        assertTrue(out.get("flagged").getAsBoolean());
        assertFalse(out.get("should_mute").getAsBoolean());
        assertEquals("slur", out.getAsJsonArray("mute_words").get(0).getAsString());
        assertEquals("rejected", out.get("confirmation").getAsString());
    }

    @Test
    void noSecondModelMeansUnchecked() {
        JsonObject out = run("you loser", null);
        assertTrue(out.get("flagged").getAsBoolean());
        assertEquals("unchecked", out.get("confirmation").getAsString());
        assertEquals("unchecked", run("hello there", "hello there").get("confirmation").getAsString());
    }

    @Test
    void muteWordsMuteOnlyWhenConfirmed() {
        assertTrue(run("a slur here", "slur").get("should_mute").getAsBoolean());
        assertFalse(run("a slur here", "nothing").get("should_mute").getAsBoolean());
    }

    @Test
    void wholeWordModeIgnoresWordsInsideWords() {
        assertTrue(WordMatcher.find("touch grass", List.of("ass"), false, false).isEmpty());
        assertEquals(List.of("ass"), WordMatcher.find("touch grass", List.of("ass"), true, false));
    }
}
