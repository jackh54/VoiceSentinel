package dev.pandadevv.voicesentinel.engine;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Reads the 16-bit PCM WAV files sent by the plugin. */
final class WavDecoder {
    record Pcm(float[] samples, int sampleRate) {
        double seconds() {
            return samples.length / (double) sampleRate;
        }
    }

    private WavDecoder() {
    }

    static Pcm decode(byte[] wav) {
        ByteBuffer in = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
        if (wav.length < 44 || in.getInt(0) != 0x46464952 || in.getInt(8) != 0x45564157) {
            throw new IllegalArgumentException("not a WAV file");
        }
        int channels = 1;
        int sampleRate = 48000;
        int bits = 16;
        int offset = 12;
        while (offset + 8 <= wav.length) {
            int id = in.getInt(offset);
            int size = in.getInt(offset + 4);
            int body = offset + 8;
            if (size < 0) {
                break;
            }
            if (id == 0x20746d66 && body + 16 <= wav.length) {
                channels = Math.max(1, in.getShort(body + 2));
                sampleRate = in.getInt(body + 4);
                bits = in.getShort(body + 14);
            } else if (id == 0x61746164) {
                if (bits != 16) {
                    throw new IllegalArgumentException("only 16-bit PCM is supported");
                }
                int available = Math.min(size, wav.length - body);
                int frames = available / 2 / channels;
                float[] out = new float[frames];
                for (int i = 0; i < frames; i++) {
                    int sum = 0;
                    for (int c = 0; c < channels; c++) {
                        sum += in.getShort(body + (i * channels + c) * 2);
                    }
                    out[i] = sum / (float) channels / 32768f;
                }
                return new Pcm(out, sampleRate);
            }
            offset = body + size + (size & 1);
        }
        throw new IllegalArgumentException("WAV has no data chunk");
    }
}
