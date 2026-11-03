package dev.pandadevv.voicesentinel.engine;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Writes mono 16-bit PCM WAV files (44-byte header), the same shape the plugin sends. */
final class WavEncoder {
    private WavEncoder() {
    }

    static byte[] encode(float[] samples, int sampleRate) {
        int dataBytes = samples.length * 2;
        ByteBuffer out = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        out.putInt(0x46464952).putInt(36 + dataBytes).putInt(0x45564157);
        out.putInt(0x20746d66).putInt(16).putShort((short) 1).putShort((short) 1)
            .putInt(sampleRate).putInt(sampleRate * 2).putShort((short) 2).putShort((short) 16);
        out.putInt(0x61746164).putInt(dataBytes);
        for (float sample : samples) {
            out.putShort((short) Math.max(-32768, Math.min(32767, Math.round(sample * 32768f))));
        }
        return out.array();
    }
}
