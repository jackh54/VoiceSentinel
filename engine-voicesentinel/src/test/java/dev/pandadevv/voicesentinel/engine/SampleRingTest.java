package dev.pandadevv.voicesentinel.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

class SampleRingTest {
    @Test
    void returnsEverythingBeforeItFills() {
        SampleRing ring = new SampleRing(5);
        ring.append(new float[] {1, 2, 3});
        assertArrayEquals(new float[] {1, 2, 3}, ring.last(10));
        assertArrayEquals(new float[] {2, 3}, ring.last(2));
    }

    @Test
    void keepsOnlyTheNewestSamplesOnceFull() {
        SampleRing ring = new SampleRing(5);
        ring.append(new float[] {1, 2, 3});
        ring.append(new float[] {4, 5, 6, 7});
        assertArrayEquals(new float[] {3, 4, 5, 6, 7}, ring.last(99));
        assertArrayEquals(new float[] {5, 6, 7}, ring.last(3));
    }

    @Test
    void anAppendLargerThanTheRingKeepsItsTail() {
        SampleRing ring = new SampleRing(3);
        ring.append(new float[] {1, 2, 3, 4, 5, 6, 7});
        assertArrayEquals(new float[] {5, 6, 7}, ring.last(3));
    }

    @Test
    void windowSpansTheWrapPoint() {
        SampleRing ring = new SampleRing(4);
        for (int i = 1; i <= 6; i++) {
            ring.append(new float[] {i});
        }
        assertArrayEquals(new float[] {4, 5, 6}, ring.last(3));
    }
}
