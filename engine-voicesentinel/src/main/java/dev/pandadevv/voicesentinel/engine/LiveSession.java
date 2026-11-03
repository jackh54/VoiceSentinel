package dev.pandadevv.voicesentinel.engine;

import com.google.gson.JsonObject;
import com.k2fsa.sherpa.onnx.OnlineStream;

import org.java_websocket.WebSocket;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * One player's live state. Every task for a session runs one at a time and in arrival order on the shared
 * lane pool, so the fields below that are marked lane-only need no locking.
 */
final class LiveSession {
    /** What the plugin sent with a chunk. */
    record Request(WebSocket conn, String player, String sessionId, String language, String utteranceId,
                   Moderator.Words words, boolean partialMatch, boolean caseSensitive) {
    }

    /** Runs the whole-utterance backstop once the live part of the utterance is over. */
    interface Backstop {
        void run(Request request, WavDecoder.Pcm whole, Set<String> alreadySent, boolean streamed,
                 List<CompletableFuture<Void>> pending);
    }

    private final LiveEngine engine;
    private final ArrayDeque<Runnable> queue = new ArrayDeque<>();
    private boolean draining;
    private final AtomicInteger queuedChunks = new AtomicInteger();
    private long lastDropLog;

    // lane-only
    private final WordCooldown cooldown = new WordCooldown(LiveEngine.COOLDOWN_MS);
    private long clockMs;
    private String utteranceId;
    private boolean streamed;
    private float[] utterance = new float[0];
    private int utteranceLength;
    private int sampleRate = 48000;
    private SampleRing ring;
    private OnlineStream stream;
    private List<String> streamWords;
    private List<String> keywordWords = List.of();
    private Set<String> sentWords = Collections.synchronizedSet(new HashSet<>());
    private List<CompletableFuture<Void>> pending = Collections.synchronizedList(new ArrayList<>());

    LiveSession(LiveEngine engine) {
        this.engine = engine;
    }

    private synchronized void lane(Runnable task) {
        queue.add(task);
        if (!draining) {
            draining = true;
            engine.lanes.execute(this::drain);
        }
    }

    private void drain() {
        while (true) {
            Runnable task;
            synchronized (this) {
                task = queue.poll();
                if (task == null) {
                    draining = false;
                    return;
                }
            }
            try {
                task.run();
            } catch (Throwable e) {
                System.err.println("Live session error: " + e);
            }
        }
    }

    /** Queues a streamed chunk; called from the WebSocket thread, so it only enqueues. */
    void chunk(Request request, String audioBase64) {
        if (queuedChunks.incrementAndGet() > LiveEngine.MAX_QUEUED_CHUNKS) {
            queuedChunks.decrementAndGet();
            logDrop("Live queue full, dropped chunk for " + request.player());
            return;
        }
        long queuedAt = System.nanoTime();
        lane(() -> {
            queuedChunks.decrementAndGet();
            boolean backedUp = (System.nanoTime() - queuedAt) / 1_000_000L > LiveEngine.MAX_BACKLOG_MS;
            if (backedUp) {
                logDrop("Live backlog over 1s, skipping spotting for " + request.player());
            }
            onChunk(request, audioBase64, backedUp);
        });
    }

    /** Queues the end of an utterance; the backstop runs after the last chunk has been spotted. */
    void finish(Request request, String audioBase64, Backstop backstop) {
        lane(() -> {
            WavDecoder.Pcm last = decode(audioBase64);
            if (!request.utteranceId().equals(utteranceId)) {
                begin(request.utteranceId());
            }
            if (last.samples().length > 0) {
                sampleRate = last.sampleRate();
                append(last.samples());
            }
            WavDecoder.Pcm whole = new WavDecoder.Pcm(java.util.Arrays.copyOf(utterance, utteranceLength), sampleRate);
            Set<String> sent = sentWords;
            List<CompletableFuture<Void>> waiting = pending;
            boolean wasStreamed = streamed;
            end();
            backstop.run(request, whole, sent, wasStreamed, waiting);
        });
    }

    /** Frees native state when the connection closes. */
    void close() {
        lane(this::end);
    }

    private void onChunk(Request request, String audioBase64, boolean backedUp) {
        if (!request.utteranceId().equals(utteranceId)) {
            begin(request.utteranceId());
        }
        WavDecoder.Pcm pcm = decode(audioBase64);
        float[] samples = pcm.samples();
        if (samples.length == 0) {
            return;
        }
        streamed = true;
        sampleRate = pcm.sampleRate();
        append(samples);
        if (backedUp) {
            return;
        }
        if (ring == null) {
            ring = new SampleRing(LiveEngine.RING_SECONDS * sampleRate);
        }
        ring.append(samples);
        clockMs += samples.length * 1000L / sampleRate;
        syncKeywords(request.words());
        if (stream == null) {
            return;
        }
        stream.acceptWaveform(samples, sampleRate);
        while (engine.isReady(stream)) {
            engine.decode(stream);
            String name = engine.hit(stream);
            if (!name.isEmpty()) {
                engine.reset(stream);
                onHit(request, name);
            }
        }
    }

    private void syncKeywords(Moderator.Words words) {
        List<String> all = new ArrayList<>(words.profanity());
        all.addAll(words.mute());
        if (stream != null && all.equals(streamWords)) {
            return;
        }
        if (stream != null) {
            stream.release();
            stream = null;
        }
        streamWords = all;
        List<String> used = new ArrayList<>();
        String text = engine.keywords(all, used);
        keywordWords = used;
        if (!used.isEmpty()) {
            stream = engine.createStream(text);
        }
    }

    private void onHit(Request request, String name) {
        int index;
        try {
            index = Integer.parseInt(name.substring(1));
        } catch (RuntimeException e) {
            return;
        }
        String word = index < keywordWords.size() ? keywordWords.get(index) : name;
        if (!cooldown.tryAcquire(word.toLowerCase(java.util.Locale.ROOT), clockMs)) {
            return;
        }
        float[] window = ring.last(LiveEngine.WINDOW_SECONDS * sampleRate);
        int rate = sampleRate;
        Set<String> sent = sentWords;
        CompletableFuture<Void> done = new CompletableFuture<>();
        pending.add(done);
        try {
            engine.doubleCheck.execute(() -> {
                try {
                    doubleCheck(request, new WavDecoder.Pcm(window, rate), sent);
                } catch (Throwable e) {
                    System.err.println("Live double check error: " + e);
                } finally {
                    done.complete(null);
                }
            });
        } catch (RejectedExecutionException e) {
            done.complete(null);
            logDrop("Live double check queue full, dropped hit for " + request.player());
        }
    }

    private void doubleCheck(Request request, WavDecoder.Pcm window, Set<String> sent) {
        long start = System.nanoTime();
        SpeechEngine speech = engine.speech;
        String transcript = speech.transcribe(window);
        long sttMs = (System.nanoTime() - start) / 1_000_000L;
        JsonObject out = Moderator.result(transcript, request.words(), request.partialMatch(),
            request.caseSensitive(), ignored -> speech.confirmTranscript(window),
            request.player(), request.sessionId(), request.language());
        if (!out.get("flagged").getAsBoolean()) {
            return;
        }
        long totalMs = (System.nanoTime() - start) / 1_000_000L;
        out.addProperty("type", "partial_hit");
        out.addProperty("source", "live");
        out.addProperty("utterance_id", request.utteranceId());
        out.addProperty("stt_time_ms", sttMs);
        out.addProperty("moderation_time_ms", totalMs - sttMs);
        out.addProperty("processing_time_ms", totalMs);
        out.addProperty("audio_data",
            Base64.getEncoder().encodeToString(WavEncoder.encode(window.samples(), window.sampleRate())));
        // Mark the words as reported before sending so the backstop never repeats them.
        out.getAsJsonArray("bad_words").forEach(word -> sent.add(word.getAsString()));
        System.out.println("partial_hit " + out.get("bad_words") + " took " + totalMs + "ms check="
            + out.get("double_check").getAsString());
        if (request.conn().isOpen()) {
            request.conn().send(out.toString());
        }
    }

    private void begin(String id) {
        end();
        utteranceId = id;
    }

    private void end() {
        if (stream != null) {
            stream.release();
            stream = null;
        }
        streamWords = null;
        ring = null;
        utterance = new float[0];
        utteranceLength = 0;
        streamed = false;
        utteranceId = null;
        sentWords = Collections.synchronizedSet(new HashSet<>());
        pending = Collections.synchronizedList(new ArrayList<>());
    }

    private void append(float[] samples) {
        int needed = utteranceLength + samples.length;
        if (needed > utterance.length) {
            utterance = java.util.Arrays.copyOf(utterance, Math.max(needed, utterance.length * 2));
        }
        System.arraycopy(samples, 0, utterance, utteranceLength, samples.length);
        utteranceLength = needed;
    }

    private static WavDecoder.Pcm decode(String audioBase64) {
        return WavDecoder.decode(Base64.getDecoder().decode(audioBase64));
    }

    private void logDrop(String message) {
        long now = System.currentTimeMillis();
        if (now - lastDropLog > 1000) {
            lastDropLog = now;
            System.err.println(message);
        }
    }
}
