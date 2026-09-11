package com.ripple;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Finds review issues on tracks so N/P can jump to the next problem frame.
 */
public final class ReviewNavigator {

    public enum Kind {
        GAP,
        OCCLUSION_START,
        SEED_ONLY
    }

    public static final class Issue {
        public final String trackId;
        public final int frame;
        public final Kind kind;

        public Issue(String trackId, int frame, Kind kind) {
            this.trackId = trackId;
            this.frame = frame;
            this.kind = kind;
        }

        public String statusLabel() {
            String kindLabel;
            switch (kind) {
                case GAP:
                    kindLabel = "gap";
                    break;
                case OCCLUSION_START:
                    kindLabel = "occlusion";
                    break;
                case SEED_ONLY:
                    kindLabel = "seed only";
                    break;
                default:
                    kindLabel = kind.name().toLowerCase();
            }
            return trackId + " " + kindLabel + " at frame " + (frame + 1);
        }
    }

    private ReviewNavigator() {
    }

    /**
     * Collect issues for the selected track, or every track when {@code selectedTrackId} is null.
     * Frames inside an occlusion segment are not reported as gaps.
     */
    public static List<Issue> findIssues(
            Map<String, Map<Integer, Point>> tracks,
            Map<String, List<int[]>> occlusions,
            Map<String, Boolean> optimized,
            String selectedTrackId) {
        List<Issue> issues = new ArrayList<>();
        if (tracks == null || tracks.isEmpty()) {
            return issues;
        }
        List<String> trackIds = new ArrayList<>();
        if (selectedTrackId != null && tracks.containsKey(selectedTrackId)) {
            trackIds.add(selectedTrackId);
        } else {
            trackIds.addAll(tracks.keySet());
            trackIds.sort(Comparator.naturalOrder());
        }
        for (String trackId : trackIds) {
            collectTrackIssues(trackId, tracks.get(trackId),
                occlusions == null ? null : occlusions.get(trackId),
                optimized == null ? Boolean.FALSE : optimized.getOrDefault(trackId, Boolean.FALSE),
                issues);
        }
        issues.sort(Comparator.comparingInt((Issue issue) -> issue.frame)
            .thenComparing(issue -> issue.trackId)
            .thenComparing(issue -> issue.kind.name()));
        return issues;
    }

    public static Issue next(List<Issue> issues, int currentFrame0) {
        if (issues == null || issues.isEmpty()) {
            return null;
        }
        for (Issue issue : issues) {
            if (issue.frame > currentFrame0) {
                return issue;
            }
        }
        return issues.get(0);
    }

    public static Issue previous(List<Issue> issues, int currentFrame0) {
        if (issues == null || issues.isEmpty()) {
            return null;
        }
        for (int i = issues.size() - 1; i >= 0; i--) {
            if (issues.get(i).frame < currentFrame0) {
                return issues.get(i);
            }
        }
        return issues.get(issues.size() - 1);
    }

    public static int indexOf(List<Issue> issues, Issue issue) {
        if (issues == null || issue == null) {
            return -1;
        }
        for (int i = 0; i < issues.size(); i++) {
            Issue candidate = issues.get(i);
            if (candidate.trackId.equals(issue.trackId)
                && candidate.frame == issue.frame
                && candidate.kind == issue.kind) {
                return i;
            }
        }
        return -1;
    }

    private static void collectTrackIssues(
            String trackId,
            Map<Integer, Point> points,
            List<int[]> occlusionSegments,
            Boolean optimized,
            List<Issue> issues) {
        if (trackId == null || points == null || points.isEmpty()) {
            return;
        }
        List<Integer> frames = new ArrayList<>(points.keySet());
        Collections.sort(frames);
        int minFrame = frames.get(0);
        int maxFrame = frames.get(frames.size() - 1);
        if (!Boolean.TRUE.equals(optimized) && points.size() == 1) {
            issues.add(new Issue(trackId, minFrame, Kind.SEED_ONLY));
        }
        if (occlusionSegments != null) {
            for (int[] segment : occlusionSegments) {
                if (segment == null || segment.length < 2) {
                    continue;
                }
                issues.add(new Issue(trackId, segment[0], Kind.OCCLUSION_START));
            }
        }
        for (int frame = minFrame + 1; frame < maxFrame; frame++) {
            if (points.containsKey(frame) || isOccluded(occlusionSegments, frame)) {
                continue;
            }
            issues.add(new Issue(trackId, frame, Kind.GAP));
        }
    }

    static boolean isOccluded(List<int[]> occlusionSegments, int frame) {
        if (occlusionSegments == null) {
            return false;
        }
        for (int[] segment : occlusionSegments) {
            if (segment == null || segment.length < 2) {
                continue;
            }
            if (frame >= segment[0] && frame <= segment[1]) {
                return true;
            }
        }
        return false;
    }
}
