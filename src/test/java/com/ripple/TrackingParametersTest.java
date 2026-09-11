package com.ripple;

import org.junit.jupiter.api.Test;

import java.awt.Point;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class TrackingParametersTest {

    @Test
    void blobSearchRadiusAcceptsValuesBelowFive() {
        assertEquals(2, TrackingParameters.validateBlobSearchRadiusPixels("2", 15));
        assertEquals(1, TrackingParameters.validateBlobSearchRadiusPixels("1", 15));
        assertEquals(4, TrackingParameters.validateBlobSearchRadiusPixels("4.4", 15));
    }

    @Test
    void blobSearchRadiusRejectsNonPositiveValues() {
        assertEquals(15, TrackingParameters.validateBlobSearchRadiusPixels("0", 15));
        assertEquals(15, TrackingParameters.validateBlobSearchRadiusPixels("-3", 15));
        assertEquals(15, TrackingParameters.validateBlobSearchRadiusPixels("nan", 15));
    }

    @Test
    void localWindowAllowsValuesAboveFiftyOne() {
        assertEquals(101, TrackingParameters.normalizeLocalWindow(101, 500));
        assertEquals(10001, TrackingParameters.normalizeLocalWindow(20000, 0));
    }

    @Test
    void localWindowEnforcesOddSize() {
        assertEquals(11, TrackingParameters.normalizeLocalWindow(10, 500));
        assertEquals(3, TrackingParameters.normalizeLocalWindow(2, 500));
    }

    @Test
    void localWindowClipsToVideoFrameCount() {
        assertEquals(99, TrackingParameters.normalizeLocalWindow(101, 100));
        assertEquals(99, TrackingParameters.getLocalWindowMaximum(100));
    }

    @Test
    void computeLocalCorrectionRangeIsCenteredAndClipped() {
        TrackingParameters.LocalCorrectionRange range =
            TrackingParameters.computeLocalCorrectionRange(50, 11, 100);

        assertEquals(45, range.startFrame);
        assertEquals(55, range.endFrame);
        assertEquals(11, range.windowFrames);
        assertEquals(50, range.correctionFrame);
    }

    @Test
    void mergeLocalCorrectionChangesOnlyWindowFrames() {
        Map<Integer, Point> original = new HashMap<>();
        for (int frame = 0; frame < 20; frame++) {
            original.put(frame, new Point(frame, frame * 10));
        }

        Map<Integer, Point> optimized = new HashMap<>();
        for (int frame = 0; frame < 20; frame++) {
            optimized.put(frame, new Point(999, 999));
        }

        TrackingParameters.LocalCorrectionRange range =
            TrackingParameters.computeLocalCorrectionRange(10, 5, 20);

        Map<Integer, Point> merged = TrackingParameters.mergeLocalCorrection(
            original, optimized, range);

        for (int frame = 0; frame < 20; frame++) {
            Point expected = (frame >= range.startFrame && frame <= range.endFrame)
                ? new Point(999, 999)
                : new Point(frame, frame * 10);
            assertEquals(expected, merged.get(frame), "frame " + frame);
        }
    }

    @Test
    void mergeLocalCorrectionLeavesOtherTracksUntouchedByCallerContract() {
        Map<Integer, Point> trackA = new HashMap<>();
        Map<Integer, Point> trackB = new HashMap<>();
        trackA.put(0, new Point(1, 1));
        trackB.put(0, new Point(5, 5));

        Map<Integer, Point> optimized = new HashMap<>();
        optimized.put(0, new Point(9, 9));

        TrackingParameters.LocalCorrectionRange range =
            new TrackingParameters.LocalCorrectionRange(0, 0, 1, 0);

        Map<Integer, Point> mergedA = TrackingParameters.mergeLocalCorrection(
            trackA, optimized, range);

        assertEquals(new Point(9, 9), mergedA.get(0));
        assertEquals(new Point(5, 5), trackB.get(0));
    }

    @Test
    void shouldUseLocalCorrectionRequiresExistingTrackAndFrame() {
        Map<Integer, Point> track = new HashMap<>();
        track.put(0, new Point(1, 1));
        track.put(1, new Point(2, 2));

        assertTrue(TrackingParameters.shouldUseLocalCorrection(true, track, 1));
        assertFalse(TrackingParameters.shouldUseLocalCorrection(true, track, -1));
        assertFalse(TrackingParameters.shouldUseLocalCorrection(false, track, 1));
        assertFalse(TrackingParameters.shouldUseLocalCorrection(true, null, 1));
    }

    @Test
    void configurationRoundTripPreservesSmallSearchRadiusAndLargeLocalWindow() {
        java.util.Properties config = new java.util.Properties();
        config.setProperty("dis.blob.search.radius", "2");
        config.setProperty("correction.local.window", "101");
        config.setProperty("correction.local.mode", "true");

        assertEquals(2, TrackingParameters.validateBlobSearchRadiusPixels(
            config.getProperty("dis.blob.search.radius"), 15));
        assertEquals(101, TrackingParameters.normalizeLocalWindow(
            config.getProperty("correction.local.window"), 500, 11));
        assertTrue(Boolean.parseBoolean(config.getProperty("correction.local.mode")));
    }

    @Test
    void syntheticLocalCorrectionRegression() {
        Map<Integer, Point> track1 = new HashMap<>();
        Map<Integer, Point> track2 = new HashMap<>();
        for (int frame = 0; frame < 30; frame++) {
            track1.put(frame, new Point(100 + frame, 200));
            track2.put(frame, new Point(50, 50));
        }

        Map<Integer, Point> beforeTrack1 = TrackingParameters.deepCopyPoints(track1);
        Map<Integer, Point> beforeTrack2 = TrackingParameters.deepCopyPoints(track2);

        Map<Integer, Point> optimized = new HashMap<>();
        for (int frame = 0; frame < 30; frame++) {
            optimized.put(frame, new Point(999, 999));
        }

        TrackingParameters.LocalCorrectionRange range =
            TrackingParameters.computeLocalCorrectionRange(15, 9, 30);
        Map<Integer, Point> afterTrack1 = TrackingParameters.mergeLocalCorrection(
            track1, optimized, range);

        for (int frame = 0; frame < 30; frame++) {
            if (frame < range.startFrame || frame > range.endFrame) {
                assertEquals(beforeTrack1.get(frame), afterTrack1.get(frame));
            } else {
                assertEquals(new Point(999, 999), afterTrack1.get(frame));
            }
        }
        assertEquals(beforeTrack2, track2);
    }
}
