package dev.pandadevv.voicesentinel.engine;

/** Fixed-size ring of the most recent samples. Not thread safe; one session lane owns it. */
final class SampleRing {
    private final float[] data;
    private int next;
    private int size;

    SampleRing(int capacity) {
        this.data = new float[capacity];
    }

    void append(float[] samples) {
        int from = Math.max(0, samples.length - data.length);
        for (int i = from; i < samples.length; i++) {
            data[next] = samples[i];
            next = (next + 1) % data.length;
        }
        size = Math.min(data.length, size + samples.length - from);
    }

    /** A copy of up to the last {@code count} samples, oldest first. */
    float[] last(int count) {
        int n = Math.min(count, size);
        float[] out = new float[n];
        int start = (next - n + data.length) % data.length;
        for (int i = 0; i < n; i++) {
            out[i] = data[(start + i) % data.length];
        }
        return out;
    }
}
