package dev.pandadevv.voicesentinel.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DownsamplerTest {
    private static float[] tone(double hz, int rate, int samples) {
        float[] out = new float[samples];
        for (int i = 0; i < samples; i++) {
            out[i] = (float) Math.sin(2 * Math.PI * hz * i / rate);
        }
        return out;
    }

    private static double rms(float[] x, int from) {
        double sum = 0;
        for (int i = from; i < x.length; i++) {
            sum += x[i] * x[i];
        }
        return Math.sqrt(sum / (x.length - from));
    }

    @Test
    void keepsSpeechBandAndRemovesWhatWouldAlias() {
        WavDecoder.Pcm low = Downsampler.toModelRate(new WavDecoder.Pcm(tone(1000, 48000, 48000), 48000));
        WavDecoder.Pcm high = Downsampler.toModelRate(new WavDecoder.Pcm(tone(12000, 48000, 48000), 48000));
        assertEquals(16000, low.sampleRate());
        assertEquals(16000, low.samples().length);
        assertEquals(0.707, rms(low.samples(), 100), 0.02);
        assertTrue(rms(high.samples(), 100) < 0.01);
    }

    @Test
    void chunkedOutputMatchesOneShot() {
        float[] audio = tone(700, 48000, 48000);
        float[] whole = Downsampler.forRate(48000).process(audio);
        Downsampler streaming = Downsampler.forRate(48000);
        java.util.List<Float> parts = new java.util.ArrayList<>();
        for (int at = 0; at < audio.length; at += 9601) {
            for (float v : streaming.process(java.util.Arrays.copyOfRange(audio, at, Math.min(audio.length, at + 9601)))) {
                parts.add(v);
            }
        }
        float[] joined = new float[parts.size()];
        for (int i = 0; i < joined.length; i++) {
            joined[i] = parts.get(i);
        }
        assertArrayEquals(whole, joined, 1e-5f);
    }

    @Test
    void sixteenKilohertzAndOddRatesAreLeftAlone() {
        assertNull(Downsampler.forRate(16000));
        assertNull(Downsampler.forRate(44100));
        WavDecoder.Pcm pcm = new WavDecoder.Pcm(new float[10], 44100);
        assertEquals(pcm, Downsampler.toModelRate(pcm));
    }
}
