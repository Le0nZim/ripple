package com.ripple;

import org.junit.jupiter.api.Test;

import java.awt.Point;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ReviewNavigatorTest {

    @Test
    void findsInteriorGapAndSkipsOccludedFrames() {
        Map<String, Map<Integer, Point>> tracks = new HashMap<>();
        Map<Integer, Point> points = new HashMap<>();
        points.put(0, new Point(1, 1));
        points.put(4, new Point(2, 2));
        tracks.put("Track1", points);

        Map<String, List<int[]>> occlusions = new HashMap<>();
        List<int[]> segments = new ArrayList<>();
        segments.add(new int[]{2, 3, 0});
        occlusions.put("Track1", segments);

        List<ReviewNavigator.Issue> issues = ReviewNavigator.findIssues(tracks, occlusions, null, "Track1");
        assertEquals(2, issues.size());
        assertEquals(ReviewNavigator.Kind.GAP, issues.get(0).kind);
        assertEquals(1, issues.get(0).frame);
        assertEquals(ReviewNavigator.Kind.OCCLUSION_START, issues.get(1).kind);
        assertEquals(2, issues.get(1).frame);
    }

    @Test
    void flagsUnoptimizedSeedOnlyTracks() {
        Map<String, Map<Integer, Point>> tracks = new HashMap<>();
        Map<Integer, Point> points = new HashMap<>();
        points.put(7, new Point(8, 9));
        tracks.put("Track2", points);
        Map<String, Boolean> optimized = new HashMap<>();
        optimized.put("Track2", false);

        List<ReviewNavigator.Issue> issues = ReviewNavigator.findIssues(tracks, null, optimized, null);
        assertEquals(1, issues.size());
        assertEquals(ReviewNavigator.Kind.SEED_ONLY, issues.get(0).kind);
        assertEquals(7, issues.get(0).frame);
    }

    @Test
    void nextAndPreviousWrapAround() {
        List<ReviewNavigator.Issue> issues = new ArrayList<>();
        issues.add(new ReviewNavigator.Issue("Track1", 2, ReviewNavigator.Kind.GAP));
        issues.add(new ReviewNavigator.Issue("Track1", 8, ReviewNavigator.Kind.OCCLUSION_START));

        ReviewNavigator.Issue next = ReviewNavigator.next(issues, 8);
        assertEquals(2, next.frame);
        ReviewNavigator.Issue previous = ReviewNavigator.previous(issues, 2);
        assertEquals(8, previous.frame);
        assertEquals(0, ReviewNavigator.indexOf(issues, next));
        assertTrue(next.statusLabel().contains("frame 3"));
    }

    @Test
    void selectedTrackFiltersOtherTracks() {
        Map<String, Map<Integer, Point>> tracks = new HashMap<>();
        Map<Integer, Point> a = new HashMap<>();
        a.put(0, new Point(1, 1));
        a.put(2, new Point(1, 2));
        tracks.put("Track1", a);
        Map<Integer, Point> b = new HashMap<>();
        b.put(0, new Point(3, 3));
        tracks.put("Track2", b);

        List<ReviewNavigator.Issue> issues = ReviewNavigator.findIssues(tracks, null, null, "Track1");
        assertEquals(1, issues.size());
        assertEquals("Track1", issues.get(0).trackId);
        assertEquals(ReviewNavigator.Kind.GAP, issues.get(0).kind);
    }
}
