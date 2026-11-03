package dev.pandadevv.voicesentinel.engine;

import com.k2fsa.sherpa.onnx.FeatureConfig;
import com.k2fsa.sherpa.onnx.KeywordSpotter;
import com.k2fsa.sherpa.onnx.KeywordSpotterConfig;
import com.k2fsa.sherpa.onnx.OnlineModelConfig;
import com.k2fsa.sherpa.onnx.OnlineStream;
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Live (mid-sentence) detection: one shared keyword spotter, a small pool that runs each session's audio in
 * order, and a separate pool for Double Check so a slow confirm never delays the live stream.
 */
final class LiveEngine implements AutoCloseable {
    static final int RING_SECONDS = 4;
    static final long COOLDOWN_MS = 2000;
    static final long MAX_BACKLOG_MS = 1000;
    static final int MAX_QUEUED_CHUNKS = 100;

    final SpeechEngine speech;
    final int windowMs;
    final ThreadPoolExecutor lanes;
    final ThreadPoolExecutor doubleCheck = new ThreadPoolExecutor(
        1, 1, 0L, TimeUnit.MILLISECONDS, new java.util.concurrent.ArrayBlockingQueue<>(16));
    private final KeywordSpotter spotter;
    private final KwsTokenizer tokenizer;

    private LiveEngine(SpeechEngine speech, KeywordSpotter spotter, KwsTokenizer tokenizer, int laneThreads,
                       int windowMs) {
        this.speech = speech;
        this.windowMs = windowMs;
        this.spotter = spotter;
        this.tokenizer = tokenizer;
        // Each session has at most one drain task queued, so this queue is bounded by the session count.
        this.lanes = new ThreadPoolExecutor(laneThreads, laneThreads, 0L, TimeUnit.MILLISECONDS,
            new LinkedBlockingQueue<>());
    }

    /** Live is only available when a kws folder sits next to the speech models and Double Check exists. */
    static LiveEngine open(Path models, SpeechEngine speech, float boost, float threshold, int laneThreads,
                          int windowMs) {
        Path dir = models.resolve("kws");
        if (!Files.isDirectory(dir) || !speech.canConfirm()) {
            return null;
        }
        try {
            KwsTokenizer tokenizer = KwsTokenizer.load(dir);
            // The native config insists on a keywords file even though every stream brings its own list.
            Path placeholder = Files.createTempFile("vs-kws", ".txt");
            placeholder.toFile().deleteOnExit();
            Files.writeString(placeholder, tokenizer.line("hello", "placeholder") + "\n", StandardCharsets.UTF_8);
            OnlineTransducerModelConfig transducer = OnlineTransducerModelConfig.builder()
                .setEncoder(dir.resolve("encoder.int8.onnx").toString())
                .setDecoder(dir.resolve("decoder.onnx").toString())
                .setJoiner(dir.resolve("joiner.int8.onnx").toString())
                .build();
            OnlineModelConfig model = OnlineModelConfig.builder().setTransducer(transducer)
                .setTokens(dir.resolve("tokens.txt").toString())
                .setNumThreads(1).setDebug(false).build();
            KeywordSpotterConfig config = KeywordSpotterConfig.builder()
                .setFeatureConfig(FeatureConfig.builder().setSampleRate(16000).setFeatureDim(80).build())
                .setOnlineModelConfig(model)
                .setKeywordsFile(placeholder.toString())
                .setKeywordsScore(boost).setKeywordsThreshold(threshold)
                .setNumTrailingBlanks(1).build();
            return new LiveEngine(speech, new KeywordSpotter(config), tokenizer, laneThreads, windowMs);
        } catch (IOException | RuntimeException | UnsatisfiedLinkError e) {
            System.err.println("Live detection unavailable: " + e);
            return null;
        }
    }

    /** Keyword lines for the words (named w0, w1, ...); words that cannot be tokenized are left out. */
    String keywords(List<String> words, List<String> usedWords) {
        StringBuilder text = new StringBuilder();
        for (String word : words) {
            String line = tokenizer.line(word, "w" + usedWords.size());
            if (line != null) {
                usedWords.add(word);
                text.append(line).append('\n');
            }
        }
        return text.toString();
    }

    synchronized OnlineStream createStream(String keywords) {
        return spotter.createStream(keywords);
    }

    // The shared spotter is only entered by one lane at a time; each stream belongs to a single session lane.
    synchronized boolean isReady(OnlineStream stream) {
        return spotter.isReady(stream);
    }

    synchronized void decode(OnlineStream stream) {
        spotter.decode(stream);
    }

    synchronized String hit(OnlineStream stream) {
        String keyword = spotter.getResult(stream).getKeyword();
        return keyword == null ? "" : keyword;
    }

    synchronized void reset(OnlineStream stream) {
        spotter.reset(stream);
    }

    @Override
    public void close() {
        lanes.shutdownNow();
        doubleCheck.shutdownNow();
        spotter.release();
    }
}
