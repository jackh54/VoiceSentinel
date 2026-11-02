package dev.pandadevv.voicesentinel.engine;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/** Same matching rules as the Python processor's ProfanityFilter. */
final class WordMatcher {
    private WordMatcher() {
    }

    static List<String> find(String transcript, List<String> words, boolean partialMatch, boolean caseSensitive) {
        List<String> hits = new ArrayList<>();
        if (transcript == null || transcript.isBlank() || words == null) {
            return hits;
        }
        String text = caseSensitive ? transcript : transcript.toLowerCase(Locale.ROOT);
        for (String raw : words) {
            String word = caseSensitive ? raw : raw.toLowerCase(Locale.ROOT);
            if (word.isBlank() || hits.contains(word)) {
                continue;
            }
            boolean match = partialMatch
                ? text.contains(word)
                : Pattern.compile("(?<![\\p{L}\\p{N}_])" + Pattern.quote(word) + "(?![\\p{L}\\p{N}_])",
                    Pattern.UNICODE_CHARACTER_CLASS).matcher(text).find();
            if (match) {
                hits.add(word);
            }
        }
        return hits;
    }
}
