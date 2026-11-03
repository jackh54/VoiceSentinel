package dev.pandadevv.voicesentinel.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class KwsTokenizerTest {
    private static KwsTokenizer tiny() {
        Map<String, Float> pieces = new HashMap<>();
        // Characters score lowest, longer pieces merge first.
        for (String c : "▁KILYOURSEFHW".split("")) {
            pieces.put(c, -100f);
        }
        pieces.put("▁K", -10f);
        pieces.put("IL", -9f);
        pieces.put("▁KIL", -8f);
        pieces.put("▁KILL", -7f);
        pieces.put("L", -100f);
        pieces.put("▁Y", -6f);
        pieces.put("OU", -5f);
        pieces.put("▁YOU", -4f);
        pieces.put("R", -100f);
        pieces.put("▁YOUR", -3f);
        pieces.put("SE", -2f);
        pieces.put("LF", -2.5f);
        pieces.put("SELF", -1f);
        return new KwsTokenizer(pieces);
    }

    @Test
    void singleWordMergesToTheBestPieces() {
        assertEquals(List.of("▁KILL"), tiny().encode("kill"));
    }

    @Test
    void phraseGetsOneSpaceMarkerPerWord() {
        assertEquals(List.of("▁KILL", "▁YOUR", "SELF"), tiny().encode("Kill  yourself"));
    }

    @Test
    void lineCarriesTheKeywordName() {
        assertEquals("▁KILL ▁YOUR SELF @w3", tiny().line("kill yourself", "w3"));
    }

    @Test
    void wordsWithUnknownCharactersAreSkipped() {
        assertNull(tiny().encode("k1ll"));
        assertNull(tiny().encode("   "));
        assertNull(tiny().line("kill!", "w0"));
    }

    @Test
    void realModelMatchesTheModelsOwnKeywordsFile() throws Exception {
        String dir = System.getenv("KWS_MODEL_DIR");
        assumeTrue(dir != null && Files.isDirectory(Path.of(dir)), "KWS_MODEL_DIR not set");
        KwsTokenizer tokenizer = KwsTokenizer.load(Path.of(dir));
        // Lines taken from keywords.txt that ships with the gigaspeech 3.3M model.
        assertEquals(List.of("▁HE", "LL", "O", "▁WORLD"), tokenizer.encode("hello world"));
        assertEquals(List.of("▁HE", "Y", "▁S", "I", "RI"), tokenizer.encode("hey siri"));
        assertEquals(List.of("▁LOVE", "▁AND", "▁P", "E", "A", "CE"), tokenizer.encode("love and peace"));
    }
}
