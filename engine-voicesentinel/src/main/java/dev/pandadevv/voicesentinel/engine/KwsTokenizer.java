package dev.pandadevv.voicesentinel.engine;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Turns a word or phrase into the pieces the keyword spotter expects ("KILL YOURSELF" becomes
 * "▁KILL ▁YOUR SELF"). Picks the best-scoring segmentation using the piece scores in the model's
 * bpe.model, the same result SentencePiece gives (the gigaspeech model is a unigram model).
 */
final class KwsTokenizer {
    private static final String SPACE = "\u2581";
    private static final int MAX_PIECE_CHARS = 24;

    private final Map<String, Float> pieces;
    private final Set<String> vocabulary;

    /** @param pieces pieces mapped to their score (higher is better) */
    KwsTokenizer(Map<String, Float> pieces) {
        this(pieces, pieces.keySet());
    }

    /** @param vocabulary the pieces the acoustic model knows; the output only ever uses these */
    KwsTokenizer(Map<String, Float> pieces, Set<String> vocabulary) {
        this.pieces = Map.copyOf(pieces);
        this.vocabulary = Set.copyOf(vocabulary);
    }

    /** Reads tokens.txt (which pieces the model knows) and bpe.model (their merge scores) from the kws folder. */
    static KwsTokenizer load(Path dir) throws IOException {
        Map<String, Float> scores = parseModel(Files.readAllBytes(dir.resolve("bpe.model")));
        Set<String> usable = new HashSet<>();
        for (String line : Files.readAllLines(dir.resolve("tokens.txt"), StandardCharsets.UTF_8)) {
            int cut = line.lastIndexOf(' ');
            String token = cut > 0 ? line.substring(0, cut) : line;
            if (!token.isEmpty() && !token.startsWith("<")) {
                usable.add(token);
            }
        }
        return new KwsTokenizer(scores, usable);
    }

    /** Minimal protobuf read of a SentencePiece ModelProto: repeated field 1 = {1: piece, 2: score, 3: type}. */
    static Map<String, Float> parseModel(byte[] model) {
        Map<String, Float> out = new HashMap<>();
        ByteBuffer in = ByteBuffer.wrap(model).order(ByteOrder.LITTLE_ENDIAN);
        while (in.hasRemaining()) {
            long tag = varint(in);
            int field = (int) (tag >>> 3);
            int wire = (int) (tag & 7);
            if (wire == 2) {
                int length = (int) varint(in);
                if (field == 1) {
                    readPiece(in.slice(in.position(), length).order(ByteOrder.LITTLE_ENDIAN), out);
                }
                in.position(in.position() + length);
            } else if (wire == 0) {
                varint(in);
            } else if (wire == 1) {
                in.position(in.position() + 8);
            } else if (wire == 5) {
                in.position(in.position() + 4);
            } else {
                throw new IllegalArgumentException("unsupported protobuf wire type " + wire);
            }
        }
        return out;
    }

    private static void readPiece(ByteBuffer in, Map<String, Float> out) {
        String piece = null;
        float score = 0f;
        long type = 1;
        while (in.hasRemaining()) {
            long tag = varint(in);
            int field = (int) (tag >>> 3);
            int wire = (int) (tag & 7);
            if (wire == 2) {
                byte[] bytes = new byte[(int) varint(in)];
                in.get(bytes);
                if (field == 1) {
                    piece = new String(bytes, StandardCharsets.UTF_8);
                }
            } else if (wire == 5) {
                float value = in.getFloat();
                if (field == 2) {
                    score = value;
                }
            } else if (wire == 0) {
                long value = varint(in);
                if (field == 3) {
                    type = value;
                }
            } else if (wire == 1) {
                in.position(in.position() + 8);
            }
        }
        if (piece != null && type == 1) {
            out.put(piece, score);
        }
    }

    private static long varint(ByteBuffer in) {
        long value = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            byte b = in.get();
            value |= (long) (b & 0x7f) << shift;
            if ((b & 0x80) == 0) {
                break;
            }
        }
        return value;
    }

    /** The pieces for a word or phrase, or null when it contains something the model cannot spell. */
    List<String> encode(String phrase) {
        if (phrase == null || phrase.isBlank()) {
            return null;
        }
        String text = SPACE + String.join(SPACE, phrase.trim().toUpperCase(Locale.ROOT).split("\\s+"));
        int n = text.length();
        // Viterbi: the segmentation with the best total score, like SentencePiece unigram encoding.
        double[] best = new double[n + 1];
        int[] from = new int[n + 1];
        java.util.Arrays.fill(best, Double.NEGATIVE_INFINITY);
        best[0] = 0;
        for (int end = 1; end <= n; end++) {
            for (int start = Math.max(0, end - MAX_PIECE_CHARS); start < end; start++) {
                Float score = vocabulary.contains(text.substring(start, end)) ? pieces.get(text.substring(start, end)) : null;
                if (score != null && best[start] > Double.NEGATIVE_INFINITY && best[start] + score > best[end]) {
                    best[end] = best[start] + score;
                    from[end] = start;
                }
            }
        }
        if (best[n] == Double.NEGATIVE_INFINITY) {
            return null;
        }
        List<String> out = new ArrayList<>();
        for (int end = n; end > 0; end = from[end]) {
            out.add(0, text.substring(from[end], end));
        }
        return out;
    }

    /** One keywords-file line ("▁KILL ▁YOUR SELF @name"), or null when the phrase cannot be tokenized. */
    String line(String phrase, String name) {
        List<String> tokens = encode(phrase);
        return tokens == null ? null : String.join(" ", tokens) + " @" + name;
    }
}
