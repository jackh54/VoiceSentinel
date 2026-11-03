package dev.pandadevv.voicesentinel.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonObject;

import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;

class LiveDedupeTest {
    private static final Moderator.Words WORDS = new Moderator.Words(List.of("fuck", "shit"), List.of("slur"));

    private static JsonObject result(String transcript, String second) {
        return Moderator.result(transcript, WORDS, false, false, second == null ? null : v -> second, "p", "s", "en");
    }

    @Test
    void wordAlreadySentIsRemovedAndTheRestStaysFlagged() {
        JsonObject out = result("fuck that shit", "fuck that shit");
        LiveDedupe.strip(out, Set.of("fuck"));
        assertTrue(out.get("flagged").getAsBoolean());
        assertEquals("[\"shit\"]", out.get("bad_words").toString());
        assertEquals("[\"shit\"]", out.get("profanity_words").toString());
        assertFalse(out.get("should_mute").getAsBoolean());
    }

    @Test
    void nothingLeftMeansUnflaggedAndNoMute() {
        JsonObject out = result("you slur", "you slur");
        assertTrue(out.get("should_mute").getAsBoolean());
        LiveDedupe.strip(out, Set.of("slur"));
        assertFalse(out.get("flagged").getAsBoolean());
        assertFalse(out.get("should_mute").getAsBoolean());
        assertEquals("[]", out.get("bad_words").toString());
    }

    @Test
    void muteWordLeftOverStillMutes() {
        JsonObject out = result("fuck you slur", "fuck you slur");
        LiveDedupe.strip(out, Set.of("fuck"));
        assertTrue(out.get("should_mute").getAsBoolean());
    }

    @Test
    void rejectedResultNeverMutes() {
        JsonObject out = result("fuck slur", "nothing here");
        LiveDedupe.strip(out, Set.of("fuck"));
        assertFalse(out.get("should_mute").getAsBoolean());
    }

    @Test
    void rejectedHitIsNotRecordedSoTheFinalStillReportsTheWord() {
        Set<String> sent = java.util.Collections.synchronizedSet(new java.util.HashSet<>());
        assertTrue(LiveDedupe.claim(result("fuck", "nothing here"), sent));
        assertTrue(sent.isEmpty());
        JsonObject finalResult = result("fuck", "fuck");
        LiveDedupe.strip(finalResult, sent);
        assertTrue(finalResult.get("flagged").getAsBoolean());
        assertEquals("confirmed", finalResult.get("confirmation").getAsString());
    }

    @Test
    void laterConfirmedHitIsSentOnceAfterARejectedOne() {
        Set<String> sent = java.util.Collections.synchronizedSet(new java.util.HashSet<>());
        assertTrue(LiveDedupe.claim(result("fuck", "nothing here"), sent));
        assertTrue(LiveDedupe.claim(result("fuck", "fuck"), sent));
        assertEquals(Set.of("fuck"), sent);
        assertFalse(LiveDedupe.claim(result("fuck", "fuck"), sent));
    }

    @Test
    void emptySentSetLeavesTheResultAlone() {
        JsonObject out = result("fuck", "fuck");
        String before = out.toString();
        LiveDedupe.strip(out, Set.of());
        assertEquals(before, out.toString());
    }
}
