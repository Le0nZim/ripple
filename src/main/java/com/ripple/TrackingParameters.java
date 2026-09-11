package com.ripple;

import java.awt.Point;
import java.util.HashMap;
import java.util.Map;

/**
 * Shared validation and local-correction helpers for tracking parameters.
 * Used by the UI, configuration persistence, and backend request assembly.
 */
public final class TrackingParameters {

    public static final double MIN_BLOB_SEARCH_RADIUS = 1.0;
    public static final double MAX_BLOB_SEARCH_RADIUS = 500.0;
    public static final double DEFAULT_BLOB_SEARCH_RADIUS = 15.0;

    public static final int MIN_LOCAL_WINDOW = 3;
    public static final int MAX_LOCAL_WINDOW = 10_001;
    public static final int DEFAULT_LOCAL_WINDOW = 11;

    private TrackingParameters() {
    }

    /**
     * Validates blob search radius in pixels. Must be finite and positive.
     * Values below 5 are allowed; the backend uses integer pixel windows.
     */
    public static double validateBlobSearchRadius(String rawValue, double defaultValue) {
        if (rawValue == null || rawValue.trim().isEmpty()) {
            return clampBlobSearchRadius(defaultValue);
        }
        try {
            double value = Double.parseDouble(rawValue.trim());
            if (!Double.isFinite(value) || value <= 0) {
                return clampBlobSearchRadius(defaultValue);
            }
            return clampBlobSearchRadius(value);
        } catch (NumberFormatException ex) {
            return clampBlobSearchRadius(defaultValue);
        }
    }

    public static int validateBlobSearchRadiusPixels(String rawValue, int defaultValue) {
        return (int) Math.round(validateBlobSearchRadius(rawValue, defaultValue));
    }

    public static double clampBlobSearchRadius(double value) {
        return Math.max(MIN_BLOB_SEARCH_RADIUS, Math.min(MAX_BLOB_SEARCH_RADIUS, value));
    }

    /**
     * Normalizes the local correction window to an odd frame count within bounds.
     *
     * @param requestedWindow user-selected window size in frames
     * @param videoFrameCount total frames in the loaded video, or {@code <= 0} for unknown
     */
    public static int normalizeLocalWindow(int requestedWindow, int videoFrameCount) {
        int maxWindow = getLocalWindowMaximum(videoFrameCount);
        int window = Math.max(MIN_LOCAL_WINDOW, Math.min(maxWindow, requestedWindow));
        if (window % 2 == 0) {
            window++;
        }
        if (window > maxWindow) {
            window = maxWindow;
            if (window % 2 == 0) {
                window = Math.max(MIN_LOCAL_WINDOW, window - 1);
            }
        }
        return window;
    }

    public static int normalizeLocalWindow(String rawValue, int videoFrameCount, int defaultValue) {
        int parsed;
        try {
            parsed = Integer.parseInt(rawValue.trim());
        } catch (NumberFormatException | NullPointerException ex) {
            parsed = defaultValue;
        }
        return normalizeLocalWindow(parsed, videoFrameCount);
    }

    public static int getLocalWindowMaximum(int videoFrameCount) {
        if (videoFrameCount <= 0) {
            return MAX_LOCAL_WINDOW;
        }
        int capped = Math.min(MAX_LOCAL_WINDOW, videoFrameCount);
        if (capped % 2 == 0) {
            capped = Math.max(MIN_LOCAL_WINDOW, capped - 1);
        }
        return Math.max(MIN_LOCAL_WINDOW, capped);
    }

    /**
     * Computes the centered local correction frame range, clipped to valid trajectory bounds.
     */
    public static LocalCorrectionRange computeLocalCorrectionRange(
            int correctionFrame,
            int windowFrames,
            int totalFrames) {

        int window = normalizeLocalWindow(windowFrames, totalFrames);
        int halfWindow = window / 2;

        int maxFrame = Math.max(0, totalFrames - 1);
        int start = Math.max(0, correctionFrame - halfWindow);
        int end = Math.min(maxFrame, correctionFrame + halfWindow);

        return new LocalCorrectionRange(start, end, window, correctionFrame);
    }

    /**
     * Returns true when local correction should run instead of global re-optimization.
     */
    public static boolean shouldUseLocalCorrection(
            boolean localModeEnabled,
            Map<Integer, Point> existingTrack,
            int correctionFrame) {

        return localModeEnabled
                && correctionFrame >= 0
                && existingTrack != null
                && existingTrack.size() > 1;
    }

    /**
     * Merges optimized points into the original track for a local frame range only.
     * Frames outside the range and all other tracks remain unchanged by the caller.
     */
    public static Map<Integer, Point> mergeLocalCorrection(
            Map<Integer, Point> originalTrack,
            Map<Integer, Point> optimizedPoints,
            LocalCorrectionRange range) {

        Map<Integer, Point> merged = deepCopyPoints(originalTrack);
        if (optimizedPoints == null || range == null) {
            return merged;
        }
        for (int frame = range.startFrame; frame <= range.endFrame; frame++) {
            Point optimized = optimizedPoints.get(frame);
            if (optimized != null) {
                merged.put(frame, new Point(optimized));
            }
        }
        return merged;
    }

    public static Map<Integer, Point> deepCopyPoints(Map<Integer, Point> source) {
        Map<Integer, Point> copy = new HashMap<>();
        if (source == null) {
            return copy;
        }
        for (Map.Entry<Integer, Point> entry : source.entrySet()) {
            Point point = entry.getValue();
            if (point != null) {
                copy.put(entry.getKey(), new Point(point));
            }
        }
        return copy;
    }

    public static String formatLocalCorrectionRange(LocalCorrectionRange range) {
        if (range == null) {
            return "none";
        }
        return String.format(
                "frames %d-%d (window=%d, center=%d)",
                range.startFrame + 1,
                range.endFrame + 1,
                range.windowFrames,
                range.correctionFrame + 1);
    }

    public static final class LocalCorrectionRange {
        public final int startFrame;
        public final int endFrame;
        public final int windowFrames;
        public final int correctionFrame;

        public LocalCorrectionRange(int startFrame, int endFrame, int windowFrames, int correctionFrame) {
            this.startFrame = startFrame;
            this.endFrame = endFrame;
            this.windowFrames = windowFrames;
            this.correctionFrame = correctionFrame;
        }
    }
}
