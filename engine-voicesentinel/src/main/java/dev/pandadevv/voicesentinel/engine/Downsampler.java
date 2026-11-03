package dev.pandadevv.voicesentinel.engine;

/**
 * Low-pass (windowed sinc FIR) and decimate by a whole factor, so 48 kHz voice chat becomes the 16 kHz the
 * models want without sherpa creating a resampler per stream. Streaming: keep one instance per audio stream.
 */
final class Downsampler {
    static final int MODEL_RATE = 16000;

    private final int factor;
    private final float[] taps;
    private float[] history;
    private int phase;

    private Downsampler(int factor) {
        this.factor = factor;
        int length = 32 * factor + 1;
        this.taps = new float[length];
        double cutoff = 0.45 / factor; // cycles per input sample, a little under the output Nyquist
        int mid = length / 2;
        double sum = 0;
        for (int i = 0; i < length; i++) {
            double x = i - mid;
            double sinc = x == 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
            double window = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (length - 1))
                + 0.08 * Math.cos(4 * Math.PI * i / (length - 1));
            taps[i] = (float) (sinc * window);
            sum += taps[i];
        }
        for (int i = 0; i < length; i++) {
            taps[i] /= (float) sum;
        }
        this.history = new float[length - 1];
    }

    /** A downsampler for this input rate, or null when the rate is already 16 kHz or not a multiple of it. */
    static Downsampler forRate(int rate) {
        return rate > MODEL_RATE && rate % MODEL_RATE == 0 ? new Downsampler(rate / MODEL_RATE) : null;
    }

    float[] process(float[] in) {
        int keep = history.length;
        float[] x = new float[keep + in.length];
        System.arraycopy(history, 0, x, 0, keep);
        System.arraycopy(in, 0, x, keep, in.length);
        int count = x.length > keep + phase ? (x.length - keep - phase + factor - 1) / factor : 0;
        float[] out = new float[count];
        int t = keep + phase;
        for (int j = 0; j < count; j++, t += factor) {
            float acc = 0;
            for (int k = 0; k < taps.length; k++) {
                acc += taps[k] * x[t - k];
            }
            out[j] = acc;
        }
        phase = t - x.length;
        history = java.util.Arrays.copyOfRange(x, x.length - keep, x.length);
        return out;
    }

    /** One-shot conversion of a whole clip to the model rate (unchanged when it cannot be done exactly). */
    static WavDecoder.Pcm toModelRate(WavDecoder.Pcm pcm) {
        Downsampler down = forRate(pcm.sampleRate());
        return down == null ? pcm : new WavDecoder.Pcm(down.process(pcm.samples()), MODEL_RATE);
    }
}
