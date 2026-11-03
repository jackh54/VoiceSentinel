package dev.pandadevv.voicesentinel.engine;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** Removes words already reported by a partial_hit from the end-of-utterance final_transcript. */
final class LiveDedupe {
    private LiveDedupe() {
    }

    static void strip(JsonObject result, Set<String> alreadySent) {
        if (alreadySent.isEmpty()) {
            return;
        }
        List<String> profanity = remaining(result.getAsJsonArray("profanity_words"), alreadySent);
        List<String> mute = remaining(result.getAsJsonArray("mute_words"), alreadySent);
        List<String> flagged = new ArrayList<>(profanity);
        flagged.addAll(mute);
        boolean rejected = "rejected".equals(result.get("confirmation").getAsString());
        result.add("profanity_words", toJson(profanity));
        result.add("mute_words", toJson(mute));
        result.add("bad_words", toJson(flagged));
        result.addProperty("flagged", !flagged.isEmpty());
        result.addProperty("should_mute", !mute.isEmpty() && !rejected);
    }

    private static List<String> remaining(JsonArray words, Set<String> alreadySent) {
        List<String> out = new ArrayList<>();
        words.forEach(word -> {
            if (!alreadySent.contains(word.getAsString())) {
                out.add(word.getAsString());
            }
        });
        return out;
    }

    private static JsonArray toJson(List<String> values) {
        JsonArray array = new JsonArray();
        values.forEach(array::add);
        return array;
    }
}
