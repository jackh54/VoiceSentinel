package dev.pandadevv.voicesentinel.engine;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.k2fsa.sherpa.onnx.LibraryUtils;

import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;

import java.io.ByteArrayOutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Local speech engine. Speaks the processor WebSocket protocol on 127.0.0.1 only. */
public final class EngineMain extends WebSocketServer {
    private static final double MIN_AUDIO_SECONDS = 0.3;

    private final String key;
    private final SpeechEngine speech;
    private final ThreadPoolExecutor worker = new ThreadPoolExecutor(
        1, 1, 0L, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<>(64));
    private final Map<WebSocket, ClientState> clients = new ConcurrentHashMap<>();

    private static final class ClientState {
        volatile boolean authenticated;
        volatile List<String> profanity = List.of();
        volatile Map<String, Moderator.Words> languages = Map.of();
        final Map<String, ByteArrayOutputStream> buffers = new ConcurrentHashMap<>();
    }

    private EngineMain(int port, String key, SpeechEngine speech) {
        super(new InetSocketAddress("127.0.0.1", port));
        this.key = key;
        this.speech = speech;
        setReuseAddr(true);
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> opts = new ConcurrentHashMap<>();
        for (int i = 0; i + 1 < args.length; i += 2) {
            opts.put(args[i].replaceFirst("^--", ""), args[i + 1]);
        }
        System.setProperty("sherpa_onnx.native.path", opts.get("native"));
        LibraryUtils.load();
        int threads = Integer.parseInt(opts.getOrDefault("threads", "2"));
        SpeechEngine speech = new SpeechEngine(Path.of(opts.get("models")), threads);
        // The key arrives on stdin so it never shows in a process listing.
        String key = new String(System.in.readNBytes(64)).trim();
        EngineMain server = new EngineMain(Integer.parseInt(opts.get("port")), key, speech);
        watchParent(opts.get("parent"));
        server.start();
    }

    private static void watchParent(String pid) {
        if (pid == null) {
            return;
        }
        ProcessHandle.of(Long.parseLong(pid)).ifPresent(parent ->
            parent.onExit().thenRun(() -> Runtime.getRuntime().halt(0)));
    }

    @Override
    public void onOpen(WebSocket conn, ClientHandshake handshake) {
        clients.put(conn, new ClientState());
    }

    @Override
    public void onClose(WebSocket conn, int code, String reason, boolean remote) {
        clients.remove(conn);
    }

    @Override
    public void onError(WebSocket conn, Exception ex) {
        System.err.println("Engine socket error: " + ex.getMessage());
    }

    @Override
    public void onStart() {
        System.out.println("ENGINE_READY");
    }

    @Override
    public void onMessage(WebSocket conn, String text) {
        ClientState state = clients.get(conn);
        if (state == null) {
            return;
        }
        try {
            JsonObject message = JsonParser.parseString(text).getAsJsonObject();
            JsonObject data = message.has("data") && message.get("data").isJsonObject()
                ? message.getAsJsonObject("data") : message;
            String type = message.get("type").getAsString();
            if ("auth".equals(type)) {
                state.authenticated = key.equals(string(data, "server_key", ""));
                conn.send(state.authenticated
                    ? "{\"type\":\"auth_success\"}"
                    : "{\"type\":\"auth_failed\",\"reason\":\"Invalid server key\"}");
            } else if ("ping".equals(type)) {
                JsonObject pong = new JsonObject();
                pong.addProperty("type", "pong");
                pong.addProperty("timestamp", data.has("timestamp") ? data.get("timestamp").getAsLong() : 0L);
                conn.send(pong.toString());
            } else if ("audio_chunk".equals(type) && state.authenticated) {
                handleAudio(conn, state, data);
            }
        } catch (Throwable e) {
            System.err.println("Engine message error: " + e);
        }
    }

    private void handleAudio(WebSocket conn, ClientState state, JsonObject data) {
        if (data.has("profanity_words")) {
            state.profanity = parseWords(data.get("profanity_words"));
        }
        if (data.has("language_word_lists") && data.get("language_word_lists").isJsonObject()) {
            state.languages = parseLanguages(data.getAsJsonObject("language_word_lists"));
        }
        String sessionId = string(data, "session_id", "unknown");
        String player = string(data, "player", sessionId);
        String language = string(data, "detected_language", "en");
        boolean partial = !data.has("partial_match") || data.get("partial_match").getAsBoolean();
        boolean caseSensitive = data.has("case_sensitive") && data.get("case_sensitive").getAsBoolean();
        byte[] audio = Base64.getDecoder().decode(string(data, "audio_data", ""));
        ByteArrayOutputStream buffer = state.buffers.computeIfAbsent(sessionId, id -> new ByteArrayOutputStream());
        if (!data.has("is_final") || !data.get("is_final").getAsBoolean()) {
            buffer.writeBytes(audio);
            return;
        }
        state.buffers.remove(sessionId);
        byte[] whole = buffer.size() == 0 ? audio : concat(buffer.toByteArray(), audio);
        // The multi-language model does not report what was spoken, so every list applies.
        Moderator.Words words = speech.canConfirm() ? wordsFor(state, language) : allWords(state);
        try {
            worker.execute(() -> process(conn, whole, words, partial, caseSensitive, player, sessionId, language));
        } catch (java.util.concurrent.RejectedExecutionException e) {
            System.err.println("Engine queue full, dropped clip for " + player);
        }
    }

    private static byte[] concat(byte[] a, byte[] b) {
        byte[] out = new byte[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static Moderator.Words wordsFor(ClientState state, String language) {
        String lang = language == null ? "" : language.toLowerCase(Locale.ROOT).split("[-_]")[0];
        Moderator.Words match = state.languages.get(lang);
        return match != null ? match : new Moderator.Words(state.profanity, List.of());
    }

    private static Moderator.Words allWords(ClientState state) {
        List<String> profanity = new ArrayList<>(state.profanity);
        List<String> mute = new ArrayList<>();
        state.languages.values().forEach(words -> {
            profanity.addAll(words.profanity());
            mute.addAll(words.mute());
        });
        profanity.removeAll(mute);
        return new Moderator.Words(profanity, mute);
    }

    private void process(WebSocket conn, byte[] wav, Moderator.Words words, boolean partial, boolean caseSensitive,
                         String player, String sessionId, String language) {
        long start = System.nanoTime();
        try {
            WavDecoder.Pcm pcm = WavDecoder.decode(wav);
            String transcript = pcm.seconds() < MIN_AUDIO_SECONDS ? "" : speech.transcribe(pcm);
            long sttMs = (System.nanoTime() - start) / 1_000_000L;
            JsonObject out = Moderator.result(transcript, words, partial, caseSensitive,
                speech.canConfirm() ? ignored -> speech.confirmTranscript(pcm) : null,
                player, sessionId, language);
            long totalMs = (System.nanoTime() - start) / 1_000_000L;
            out.addProperty("stt_time_ms", sttMs);
            out.addProperty("moderation_time_ms", totalMs - sttMs);
            out.addProperty("processing_time_ms", totalMs);
            if (out.get("flagged").getAsBoolean()) {
                out.addProperty("audio_data", Base64.getEncoder().encodeToString(wav));
            }
            System.out.println("clip " + Math.round(pcm.seconds() * 1000) + "ms took " + totalMs + "ms check="
                + out.get("double_check").getAsString());
            if (conn.isOpen()) {
                conn.send(out.toString());
            }
        } catch (Throwable e) {
            System.err.println("Engine processing error: " + e);
        }
    }

    private static String string(JsonObject data, String key, String fallback) {
        return data.has(key) && !data.get(key).isJsonNull() ? data.get(key).getAsString() : fallback;
    }

    private static List<String> wordsIn(JsonObject entry, String key) {
        String lower = key.toLowerCase(Locale.ROOT);
        return entry.has(key) ? parseWords(entry.get(key)) : entry.has(lower) ? parseWords(entry.get(lower)) : List.of();
    }

    private static List<String> parseWords(JsonElement element) {
        JsonElement value = element;
        if (value.isJsonPrimitive()) {
            value = JsonParser.parseString(value.getAsString());
        }
        List<String> words = new ArrayList<>();
        if (value.isJsonArray()) {
            JsonArray array = value.getAsJsonArray();
            array.forEach(word -> words.add(word.getAsString()));
        }
        return words;
    }

    private static Map<String, Moderator.Words> parseLanguages(JsonObject lists) {
        Map<String, Moderator.Words> out = new ConcurrentHashMap<>();
        for (String language : lists.keySet()) {
            if (!lists.get(language).isJsonObject()) {
                continue;
            }
            JsonObject entry = lists.getAsJsonObject(language);
            out.put(language.toLowerCase(Locale.ROOT).split("[-_]")[0],
                new Moderator.Words(wordsIn(entry, "PROFANITY"), wordsIn(entry, "MUTE")));
        }
        return out;
    }
}
