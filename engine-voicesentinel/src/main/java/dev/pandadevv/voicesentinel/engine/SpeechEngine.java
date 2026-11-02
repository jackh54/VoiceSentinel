package dev.pandadevv.voicesentinel.engine;

import com.k2fsa.sherpa.onnx.OfflineModelConfig;
import com.k2fsa.sherpa.onnx.OfflineMoonshineModelConfig;
import com.k2fsa.sherpa.onnx.OfflineRecognizer;
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OfflineStream;
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig;

import java.nio.file.Files;
import java.nio.file.Path;

/** Primary recognizer plus an optional different model that confirms hits. */
final class SpeechEngine implements AutoCloseable {
    private final OfflineRecognizer primary;
    private final OfflineRecognizer confirm;

    SpeechEngine(Path modelDir, int threads) {
        Path transducer = modelDir.resolve("primary");
        this.primary = new OfflineRecognizer(config(transducerModel(transducer, threads)));
        Path moonshine = modelDir.resolve("confirm");
        this.confirm = Files.isDirectory(moonshine)
            ? new OfflineRecognizer(config(moonshineModel(moonshine, threads)))
            : null;
    }

    private static OfflineRecognizerConfig config(OfflineModelConfig model) {
        return OfflineRecognizerConfig.builder().setOfflineModelConfig(model).setDecodingMethod("greedy_search").build();
    }

    private static OfflineModelConfig transducerModel(Path dir, int threads) {
        OfflineTransducerModelConfig transducer = OfflineTransducerModelConfig.builder()
            .setEncoder(dir.resolve("encoder.int8.onnx").toString())
            .setDecoder(dir.resolve("decoder.int8.onnx").toString())
            .setJoiner(dir.resolve("joiner.int8.onnx").toString())
            .build();
        return OfflineModelConfig.builder().setTransducer(transducer)
            .setTokens(dir.resolve("tokens.txt").toString())
            .setModelType("nemo_transducer").setNumThreads(threads).build();
    }

    private static OfflineModelConfig moonshineModel(Path dir, int threads) {
        OfflineMoonshineModelConfig moonshine = OfflineMoonshineModelConfig.builder()
            .setPreprocessor(dir.resolve("preprocess.onnx").toString())
            .setEncoder(dir.resolve("encode.int8.onnx").toString())
            .setUncachedDecoder(dir.resolve("uncached_decode.int8.onnx").toString())
            .setCachedDecoder(dir.resolve("cached_decode.int8.onnx").toString())
            .build();
        return OfflineModelConfig.builder().setMoonshine(moonshine)
            .setTokens(dir.resolve("tokens.txt").toString()).setNumThreads(threads).build();
    }

    String transcribe(WavDecoder.Pcm pcm) {
        return run(primary, pcm);
    }

    boolean canConfirm() {
        return confirm != null;
    }

    String confirmTranscript(WavDecoder.Pcm pcm) {
        return run(confirm, pcm);
    }

    private static String run(OfflineRecognizer recognizer, WavDecoder.Pcm pcm) {
        OfflineStream stream = recognizer.createStream();
        try {
            stream.acceptWaveform(pcm.samples(), pcm.sampleRate());
            recognizer.decode(stream);
            return recognizer.getResult(stream).getText().trim();
        } finally {
            stream.release();
        }
    }

    @Override
    public void close() {
        primary.release();
        if (confirm != null) {
            confirm.release();
        }
    }
}
