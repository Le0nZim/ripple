package com.ripple;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Pure clip-splitting, RAM estimation, overlap, and switch-decision helpers.
 * Frame indices are 0-based and inclusive.
 */
public final class VideoBatchPlan {

    public static final int OVERLAP_FRAMES = Constants.BATCH_OVERLAP_FRAMES;
    public static final long RESERVE_BYTES = 4L * 1024 * 1024 * 1024;
    public static final double FLOW_BUDGET_FRACTION = 0.40;
    public static final int MIN_CLIP_FRAMES = 2;

    public enum SwitchAction {
        SEED_PROPAGATE,
        REOPTIMIZE_WITH_ANCHORS,
        SKIP
    }

    public enum ViewMode {
        DISABLED,
        ENTIRE_VIDEO,
        CLIP_PREVIEW,
        CLIP_WORK
    }

    private VideoBatchPlan() {
    }

    public static final class ClipRange {
        public final int index;
        public final int startFrame;
        public final int endFrame;

        public ClipRange(int index, int startFrame, int endFrame) {
            if (index < 0) {
                throw new IllegalArgumentException("clip index must be >= 0");
            }
            if (startFrame < 0 || endFrame < startFrame) {
                throw new IllegalArgumentException("invalid clip range: " + startFrame + "-" + endFrame);
            }
            this.index = index;
            this.startFrame = startFrame;
            this.endFrame = endFrame;
        }

        public int frameCount() {
            return endFrame - startFrame + 1;
        }

        public int pairCount() {
            return Math.max(0, frameCount() - 1);
        }

        public boolean contains(int globalFrame) {
            return globalFrame >= startFrame && globalFrame <= endFrame;
        }

        public int toLocal(int globalFrame) {
            return globalFrame - startFrame;
        }

        public int toGlobal(int localFrame) {
            return localFrame + startFrame;
        }

        public String displayRange1Based() {
            return (startFrame + 1) + "–" + (endFrame + 1);
        }

        public String flowToken() {
            return "f" + startFrame + "-" + endFrame;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (!(obj instanceof ClipRange)) {
                return false;
            }
            ClipRange other = (ClipRange) obj;
            return index == other.index && startFrame == other.startFrame && endFrame == other.endFrame;
        }

        @Override
        public int hashCode() {
            return Objects.hash(index, startFrame, endFrame);
        }

        @Override
        public String toString() {
            return "ClipRange{" + index + ", " + startFrame + "-" + endFrame + "}";
        }
    }

    public static int clampClipCount(int requested, int totalFrames) {
        int maxClips = Math.max(1, totalFrames - 1);
        if (totalFrames < MIN_CLIP_FRAMES) {
            return 1;
        }
        return Math.max(1, Math.min(maxClips, requested));
    }

    /**
     * Split {@code totalFrames} into {@code clipCount} overlapping clips.
     * Remainder flow-pairs are spread onto the later clips.
     */
    public static List<ClipRange> split(int totalFrames, int clipCount) {
        if (totalFrames <= 0) {
            return Collections.emptyList();
        }
        int n = clampClipCount(clipCount, totalFrames);
        if (n == 1 || totalFrames < MIN_CLIP_FRAMES) {
            return Collections.singletonList(new ClipRange(0, 0, totalFrames - 1));
        }

        int pairs = totalFrames - 1;
        int base = pairs / n;
        int remainder = pairs % n;
        List<ClipRange> clips = new ArrayList<>(n);
        int start = 0;
        for (int i = 0; i < n; i++) {
            int extra = (i >= n - remainder) ? 1 : 0;
            int clipPairs = base + extra;
            if (clipPairs < 1) {
                clipPairs = 1;
            }
            int end = Math.min(totalFrames - 1, start + clipPairs);
            clips.add(new ClipRange(i, start, end));
            start = end;
        }
        ClipRange last = clips.get(clips.size() - 1);
        if (last.endFrame != totalFrames - 1) {
            clips.set(clips.size() - 1, new ClipRange(last.index, last.startFrame, totalFrames - 1));
        }
        return Collections.unmodifiableList(clips);
    }

    /**
     * First (earliest) clip that contains the global frame. Overlap frames belong
     * to the earlier clip for navigation unless the caller already has a preference.
     */
    public static int clipIndexForFrame(List<ClipRange> clips, int globalFrame) {
        if (clips == null || clips.isEmpty()) {
            return -1;
        }
        for (ClipRange clip : clips) {
            if (clip.contains(globalFrame)) {
                return clip.index;
            }
        }
        if (globalFrame < clips.get(0).startFrame) {
            return clips.get(0).index;
        }
        return clips.get(clips.size() - 1).index;
    }

    public static ClipRange clipAt(List<ClipRange> clips, int index) {
        if (clips == null || index < 0 || index >= clips.size()) {
            return null;
        }
        return clips.get(index);
    }

    public static int overlapWithPrevious(ClipRange clip) {
        return clip == null || clip.index == 0 ? -1 : clip.startFrame;
    }

    public static int overlapWithNext(List<ClipRange> clips, ClipRange clip) {
        if (clip == null || clips == null || clip.index >= clips.size() - 1) {
            return -1;
        }
        return clip.endFrame;
    }

    public static boolean isBoundaryFrame(ClipRange clip, int globalFrame) {
        return clip != null && (globalFrame == clip.startFrame || globalFrame == clip.endFrame);
    }

    public static boolean hasInteriorAnnotations(Map<Integer, Point> points, ClipRange clip) {
        if (points == null || clip == null) {
            return false;
        }
        for (Integer frame : points.keySet()) {
            if (frame != null && frame > clip.startFrame && frame < clip.endFrame) {
                return true;
            }
        }
        return false;
    }

    public static boolean pointsDiffer(Point a, Point b) {
        if (a == null && b == null) {
            return false;
        }
        if (a == null || b == null) {
            return true;
        }
        return a.x != b.x || a.y != b.y;
    }

    /**
     * Decide how Switch should treat one track.
     *
     * @param hasInteriorAnnotations true if the destination clip already has points inside (s, e)
     * @param overlapChanged         true if a neighbor overlap position differs from the clip's
     */
    public static SwitchAction decideSwitchAction(boolean hasInteriorAnnotations, boolean overlapChanged) {
        if (!hasInteriorAnnotations) {
            return SwitchAction.SEED_PROPAGATE;
        }
        if (overlapChanged) {
            return SwitchAction.REOPTIMIZE_WITH_ANCHORS;
        }
        return SwitchAction.SKIP;
    }

    public static long bytesPerPair(int width, int height, String method, int disDownsample, boolean float16) {
        int w = Math.max(1, width);
        int h = Math.max(1, height);
        String normalized = method == null ? "raft" : method.toLowerCase();
        if ("dis".equals(normalized)) {
            int ds = Math.max(1, disDownsample);
            w = Math.max(1, w / ds);
            h = Math.max(1, h / ds);
        }
        int bytesPerValue = float16 ? 2 : 4;
        return (long) w * (long) h * 2L * bytesPerValue;
    }

    public static long estimateFlowBytes(int pairs, int width, int height, String method,
                                         int disDownsample, boolean float16) {
        return Math.max(0, pairs) * bytesPerPair(width, height, method, disDownsample, float16);
    }

    public static boolean shouldUseFloat16(long estimatedFlowBytes, long availableRamBytes) {
        long fourGb = 4L * 1024 * 1024 * 1024;
        long fiveHundredMb = 500L * 1024 * 1024;
        return estimatedFlowBytes >= fiveHundredMb || (availableRamBytes > 0 && availableRamBytes < fourGb);
    }

    public static long flowBudgetBytes(long availableRamBytes) {
        if (availableRamBytes <= 0) {
            return 0;
        }
        long fractional = (long) (availableRamBytes * FLOW_BUDGET_FRACTION);
        if (availableRamBytes <= RESERVE_BYTES) {
            return Math.max(0, fractional);
        }
        return Math.max(0, Math.min(fractional, availableRamBytes - RESERVE_BYTES));
    }

    public static int suggestedClipCount(int totalFrames, int width, int height, String method,
                                         int disDownsample, boolean float16, long availableRamBytes) {
        int maxClips = clampClipCount(Integer.MAX_VALUE, totalFrames);
        if (totalFrames < MIN_CLIP_FRAMES) {
            return 1;
        }
        int pairs = Math.max(1, totalFrames - 1);
        long bytesPair = bytesPerPair(width, height, method, disDownsample, float16);
        if (bytesPair <= 0) {
            return 1;
        }
        long budget = flowBudgetBytes(availableRamBytes);
        if (budget <= 0) {
            return maxClips;
        }
        long maxPairs = Math.max(1, budget / bytesPair);
        long needed = (pairs + maxPairs - 1) / maxPairs;
        return clampClipCount((int) Math.min(Integer.MAX_VALUE, needed), totalFrames);
    }

    public static boolean shouldPromptSplit(int totalFrames, long estimatedFullFlowBytes) {
        return shouldPromptSplit(totalFrames, estimatedFullFlowBytes, 0L);
    }

    /**
     * Prompt after open when the video is long, the full-flow estimate is large,
     * or the estimate does not fit in this computer's safe optical-flow budget.
     */
    public static boolean shouldPromptSplit(int totalFrames, long estimatedFullFlowBytes, long availableRamBytes) {
        if (totalFrames < MIN_CLIP_FRAMES) {
            return false;
        }
        if (totalFrames >= Constants.BATCH_PROMPT_MIN_FRAMES
            || estimatedFullFlowBytes >= Constants.BATCH_PROMPT_FLOW_BYTES) {
            return true;
        }
        long budget = flowBudgetBytes(availableRamBytes);
        return budget > 0 && estimatedFullFlowBytes > budget;
    }

    public static int clampSlice1Based(int slice, ClipRange clip, int totalSlices) {
        int min = 1;
        int max = Math.max(1, totalSlices);
        if (clip != null) {
            min = clip.startFrame + 1;
            max = Math.min(totalSlices, clip.endFrame + 1);
        }
        return Math.max(min, Math.min(max, slice));
    }

    public static TrackingParameters.LocalCorrectionRange clipLocalWindow(
            TrackingParameters.LocalCorrectionRange range, ClipRange clip) {
        if (range == null) {
            return null;
        }
        if (clip == null) {
            return range;
        }
        int start = Math.max(range.startFrame, clip.startFrame);
        int end = Math.min(range.endFrame, clip.endFrame);
        if (end < start) {
            start = clip.startFrame;
            end = clip.endFrame;
        }
        return new TrackingParameters.LocalCorrectionRange(
            start, end, end - start + 1, range.correctionFrame);
    }

    /**
     * Copy overlap-frame positions from source tracks into destination tracks.
     * Existing destination points on other frames are left intact.
     */
    public static void syncOverlapFrame(
            Map<String, Map<Integer, Point>> sourceTracks,
            Map<String, Map<Integer, Point>> destTracks,
            int overlapFrame) {
        if (sourceTracks == null || destTracks == null || overlapFrame < 0) {
            return;
        }
        for (Map.Entry<String, Map<Integer, Point>> entry : sourceTracks.entrySet()) {
            String trackId = entry.getKey();
            Map<Integer, Point> source = entry.getValue();
            if (trackId == null || source == null) {
                continue;
            }
            Point point = source.get(overlapFrame);
            if (point == null) {
                continue;
            }
            Map<Integer, Point> dest = destTracks.computeIfAbsent(trackId, k -> new LinkedHashMap<>());
            dest.put(overlapFrame, new Point(point));
        }
    }

    /**
     * Merge clip-local track maps into one full-video map. On overlap frames,
     * the later-listed clip in {@code clipTracks} wins when its {@code lastSavedMs}
     * is greater or equal; otherwise the earlier clip wins.
     */
    public static Map<String, Map<Integer, Point>> mergeClipTracks(
            List<Map<String, Map<Integer, Point>>> clipTracks,
            List<Long> lastSavedMs) {
        Map<String, Map<Integer, Point>> merged = new LinkedHashMap<>();
        if (clipTracks == null) {
            return merged;
        }
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < clipTracks.size(); i++) {
            order.add(i);
        }
        order.sort((a, b) -> {
            long ta = savedAt(lastSavedMs, a);
            long tb = savedAt(lastSavedMs, b);
            int cmp = Long.compare(ta, tb);
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        for (int index : order) {
            Map<String, Map<Integer, Point>> clip = clipTracks.get(index);
            if (clip == null) {
                continue;
            }
            for (Map.Entry<String, Map<Integer, Point>> trackEntry : clip.entrySet()) {
                String trackId = trackEntry.getKey();
                Map<Integer, Point> points = trackEntry.getValue();
                if (trackId == null || points == null) {
                    continue;
                }
                Map<Integer, Point> dest = merged.computeIfAbsent(trackId, k -> new LinkedHashMap<>());
                for (Map.Entry<Integer, Point> frameEntry : points.entrySet()) {
                    if (frameEntry.getKey() != null && frameEntry.getValue() != null) {
                        dest.put(frameEntry.getKey(), new Point(frameEntry.getValue()));
                    }
                }
            }
        }
        return merged;
    }

    public static Map<String, Map<Integer, Point>> splitTracksForClip(
            Map<String, Map<Integer, Point>> fullTracks, ClipRange clip) {
        Map<String, Map<Integer, Point>> split = new LinkedHashMap<>();
        if (fullTracks == null || clip == null) {
            return split;
        }
        for (Map.Entry<String, Map<Integer, Point>> entry : fullTracks.entrySet()) {
            if (entry.getKey() == null || entry.getValue() == null) {
                continue;
            }
            Map<Integer, Point> clipped = new LinkedHashMap<>();
            for (Map.Entry<Integer, Point> frameEntry : entry.getValue().entrySet()) {
                Integer frame = frameEntry.getKey();
                if (frame != null && clip.contains(frame) && frameEntry.getValue() != null) {
                    clipped.put(frame, new Point(frameEntry.getValue()));
                }
            }
            if (!clipped.isEmpty()) {
                split.put(entry.getKey(), clipped);
            }
        }
        return split;
    }

    public static Map<String, Map<Integer, Point>> boundaryOnly(
            Map<String, Map<Integer, Point>> tracks, ClipRange clip) {
        Map<String, Map<Integer, Point>> boundary = new LinkedHashMap<>();
        if (tracks == null || clip == null) {
            return boundary;
        }
        for (Map.Entry<String, Map<Integer, Point>> entry : tracks.entrySet()) {
            if (entry.getKey() == null || entry.getValue() == null) {
                continue;
            }
            Map<Integer, Point> points = new LinkedHashMap<>();
            Point first = entry.getValue().get(clip.startFrame);
            Point last = entry.getValue().get(clip.endFrame);
            if (first != null) {
                points.put(clip.startFrame, new Point(first));
            }
            if (last != null) {
                points.put(clip.endFrame, new Point(last));
            }
            if (!points.isEmpty()) {
                boundary.put(entry.getKey(), points);
            }
        }
        return boundary;
    }

    public static String formatRam(long bytes) {
        if (bytes < 0) {
            return "unknown";
        }
        double gb = bytes / (1024.0 * 1024.0 * 1024.0);
        if (gb >= 10) {
            return String.format("%.0f GB", gb);
        }
        if (gb >= 1) {
            return String.format("%.1f GB", gb);
        }
        double mb = bytes / (1024.0 * 1024.0);
        return String.format("%.0f MB", mb);
    }

    public static String suggestMessage(int totalFrames, long estimatedFullBytes, long availableBytes, int suggestedClips) {
        return String.format(
            "Optical flow for all %,d frames would use about %s. This computer has about %s free. Recommended: %d clip%s.",
            totalFrames,
            formatRam(estimatedFullBytes),
            formatRam(availableBytes),
            suggestedClips,
            suggestedClips == 1 ? "" : "s");
    }

    public static String clipAnnotationFileName(String videoBaseName, int clipIndex) {
        return String.format("%s%s%02d%s",
            videoBaseName == null ? "video" : videoBaseName,
            Constants.BATCH_ANNOTATION_INFIX,
            clipIndex + 1,
            Constants.ANNOTATION_SUFFIX);
    }

    public static String manifestFileName(String videoBaseName) {
        return (videoBaseName == null ? "video" : videoBaseName) + Constants.BATCH_MANIFEST_SUFFIX;
    }

    public static boolean flowFileMatchesClip(String filename, ClipRange clip) {
        if (filename == null || clip == null) {
            return false;
        }
        return filename.contains("_" + clip.flowToken() + "_")
            || filename.contains("_" + clip.flowToken() + ".");
    }

    private static long savedAt(List<Long> lastSavedMs, int index) {
        if (lastSavedMs == null || index < 0 || index >= lastSavedMs.size() || lastSavedMs.get(index) == null) {
            return 0L;
        }
        return lastSavedMs.get(index);
    }
}
