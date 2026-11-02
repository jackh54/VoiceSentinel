package dev.pandadevv.voicesentinel.engine;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Turns one transcript into a final_transcript message. When the second model disagrees the hit is
 * still sent, marked {@code confirmation: "rejected"} and never as should_mute, so the plugin can
 * show staff an unconfirmed flag without acting on it. {@code confirmation} is "confirmed",
 * "rejected" or "unchecked" (no second model, or nothing was flagged).
 */
final class Moderator {
    record Words(List<String> profanity, List<String> mute) {
    }

    private Moderator() {
    }

    static JsonObject result(
        String transcript,
        Words words,
        boolean partialMatch,
        boolean caseSensitive,
        Function<Void, String> confirmTranscript,
        String player,
        String sessionId,
        String language) {
        List<String> profanity = WordMatcher.find(transcript, words.profanity(), partialMatch, caseSensitive);
        List<String> mute = WordMatcher.find(transcript, words.mute(), partialMatch, caseSensitive);
        String check = "skipped";
        boolean rejected = false;
        if ((!profanity.isEmpty() || !mute.isEmpty()) && confirmTranscript != null) {
            String second = confirmTranscript.apply(null);
            List<String> all = new ArrayList<>(profanity);
            all.addAll(mute);
            List<String> agreed = WordMatcher.find(second, all, partialMatch, caseSensitive);
            rejected = agreed.isEmpty();
            if (!rejected) {
                profanity.retainAll(agreed);
                mute.retainAll(agreed);
            }
            check = rejected ? "rejected" : "confirmed";
        }
        boolean shouldMute = !mute.isEmpty() && !rejected;
        List<String> flagged = new ArrayList<>(profanity);
        flagged.addAll(mute);

        JsonObject out = new JsonObject();
        out.addProperty("type", "final_transcript");
        out.addProperty("transcript", transcript == null ? "" : transcript);
        out.addProperty("flagged", !flagged.isEmpty());
        out.addProperty("should_mute", shouldMute);
        out.add("bad_words", toJson(flagged));
        out.add("profanity_words", toJson(profanity));
        out.add("mute_words", toJson(mute));
        out.addProperty("player", player);
        out.addProperty("session_id", sessionId);
        out.addProperty("detected_language", language);
        out.addProperty("double_check", check);
        out.addProperty("confirmation", "skipped".equals(check) ? "unchecked" : check);
        return out;
    }

    private static JsonArray toJson(List<String> values) {
        JsonArray array = new JsonArray();
        values.forEach(array::add);
        return array;
    }
}
