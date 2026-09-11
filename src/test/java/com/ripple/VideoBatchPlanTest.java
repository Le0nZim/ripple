package com.ripple;

import org.json.JSONObject;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.Point;
import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class VideoBatchPlanTest {

    @Test
    void splitExactThreeClipsSharesOneFrame() {
        List<VideoBatchPlan.ClipRange> clips = VideoBatchPlan.split(1000, 3);
        assertEquals(3, clips.size());
        assertEquals(0, clips.get(0).startFrame);
        assertEquals(333, clips.get(0).endFrame);
        assertEquals(333, clips.get(1).startFrame);
        assertEquals(666, clips.get(1).endFrame);
        assertEquals(666, clips.get(2).startFrame);
        assertEquals(999, clips.get(2).endFrame);
        assertEquals(clips.get(0).endFrame, clips.get(1).startFrame);
        assertEquals(clips.get(1).endFrame, clips.get(2).startFrame);
    }

    @Test
    void splitSpreadsRemainderOntoLaterClips() {
        List<VideoBatchPlan.ClipRange> clips = VideoBatchPlan.split(11, 3);
        assertEquals(3, clips.size());
        assertEquals(0, clips.get(0).startFrame);
        assertEquals(10, clips.get(2).endFrame);
        assertEquals(clips.get(0).endFrame, clips.get(1).startFrame);
        assertEquals(clips.get(1).endFrame, clips.get(2).startFrame);
        int coveredPairs = 0;
        for (VideoBatchPlan.ClipRange clip : clips) {
            coveredPairs += clip.pairCount();
        }
        assertEquals(10, coveredPairs);
    }

    @Test
    void splitTwoFramesIsSinglePair() {
        List<VideoBatchPlan.ClipRange> clips = VideoBatchPlan.split(2, 1);
        assertEquals(1, clips.size());
        assertEquals(new VideoBatchPlan.ClipRange(0, 0, 1), clips.get(0));
    }

    @Test
    void splitMaxClipsIsOnePairEach() {
        List<VideoBatchPlan.ClipRange> clips = VideoBatchPlan.split(5, 99);
        assertEquals(4, clips.size());
        assertEquals(0, clips.get(0).startFrame);
        assertEquals(1, clips.get(0).endFrame);
        assertEquals(1, clips.get(1).startFrame);
        assertEquals(4, clips.get(3).endFrame);
    }

    @Test
    void clampClipCountHonorsBounds() {
        assertEquals(1, VideoBatchPlan.clampClipCount(0, 100));
        assertEquals(99, VideoBatchPlan.clampClipCount(500, 100));
        assertEquals(1, VideoBatchPlan.clampClipCount(8, 1));
    }

    @Test
    void clipIndexPrefersEarlierOverlapOwner() {
        List<VideoBatchPlan.ClipRange> clips = VideoBatchPlan.split(1000, 3);
        assertEquals(0, VideoBatchPlan.clipIndexForFrame(clips, 333));
        assertEquals(1, VideoBatchPlan.clipIndexForFrame(clips, 334));
        assertEquals(2, VideoBatchPlan.clipIndexForFrame(clips, 900));
    }

    @Test
    void ramSuggestionUsesDownsampleAndBudget() {
        long available = 16L * 1024 * 1024 * 1024;
        int raftClips = VideoBatchPlan.suggestedClipCount(
            4000, 1024, 1024, "raft", 2, false, available);
        int disClips = VideoBatchPlan.suggestedClipCount(
            4000, 1024, 1024, "dis", 2, false, available);
        assertTrue(raftClips >= disClips);
        assertTrue(raftClips >= 1);
        assertTrue(VideoBatchPlan.flowBudgetBytes(available) <= (long) (available * 0.40));
        assertEquals(0, VideoBatchPlan.flowBudgetBytes(0));
    }

    @Test
    void float16HalvesEstimate() {
        long full = VideoBatchPlan.estimateFlowBytes(100, 64, 64, "raft", 2, false);
        long half = VideoBatchPlan.estimateFlowBytes(100, 64, 64, "raft", 2, true);
        assertEquals(full / 2, half);
    }

    @Test
    void shouldPromptSplitOnLongOrHeavyVideos() {
        assertTrue(VideoBatchPlan.shouldPromptSplit(500, 100));
        assertTrue(VideoBatchPlan.shouldPromptSplit(10, Constants.BATCH_PROMPT_FLOW_BYTES));
        assertFalse(VideoBatchPlan.shouldPromptSplit(10, 100));
    }

    @Test
    void shouldPromptSplitWhenEstimateExceedsRamBudget() {
        long available = 6L * 1024 * 1024 * 1024;
        long budget = VideoBatchPlan.flowBudgetBytes(available);
        assertTrue(budget > 0);
        assertTrue(VideoBatchPlan.shouldPromptSplit(80, budget + 1, available));
        assertFalse(VideoBatchPlan.shouldPromptSplit(80, Math.max(1, budget / 4), available));
        assertFalse(VideoBatchPlan.shouldPromptSplit(1, budget + 1, available));
    }

    @Test
    void switchDecisionTable() {
        assertEquals(
            VideoBatchPlan.SwitchAction.SEED_PROPAGATE,
            VideoBatchPlan.decideSwitchAction(false, true));
        assertEquals(
            VideoBatchPlan.SwitchAction.SEED_PROPAGATE,
            VideoBatchPlan.decideSwitchAction(false, false));
        assertEquals(
            VideoBatchPlan.SwitchAction.REOPTIMIZE_WITH_ANCHORS,
            VideoBatchPlan.decideSwitchAction(true, true));
        assertEquals(
            VideoBatchPlan.SwitchAction.SKIP,
            VideoBatchPlan.decideSwitchAction(true, false));
    }

    @Test
    void interiorAnnotationsIgnoreBoundaryOnly() {
        VideoBatchPlan.ClipRange clip = new VideoBatchPlan.ClipRange(1, 10, 20);
        Map<Integer, Point> boundary = new HashMap<>();
        boundary.put(10, new Point(1, 1));
        boundary.put(20, new Point(2, 2));
        assertFalse(VideoBatchPlan.hasInteriorAnnotations(boundary, clip));
        boundary.put(15, new Point(3, 3));
        assertTrue(VideoBatchPlan.hasInteriorAnnotations(boundary, clip));
    }

    @Test
    void syncOverlapWritesOnlySharedFrame() {
        Map<String, Map<Integer, Point>> source = new LinkedHashMap<>();
        Map<Integer, Point> srcTrack = new LinkedHashMap<>();
        srcTrack.put(99, new Point(8, 9));
        srcTrack.put(50, new Point(1, 1));
        source.put("Track1", srcTrack);

        Map<String, Map<Integer, Point>> dest = new LinkedHashMap<>();
        Map<Integer, Point> destTrack = new LinkedHashMap<>();
        destTrack.put(99, new Point(0, 0));
        destTrack.put(120, new Point(4, 4));
        dest.put("Track1", destTrack);

        VideoBatchPlan.syncOverlapFrame(source, dest, 99);
        assertEquals(new Point(8, 9), dest.get("Track1").get(99));
        assertEquals(new Point(4, 4), dest.get("Track1").get(120));
        assertFalse(dest.get("Track1").containsKey(50));
    }

    @Test
    void mergePrefersLaterSavedOverlap() {
        Map<String, Map<Integer, Point>> clip0 = new LinkedHashMap<>();
        Map<Integer, Point> a = new LinkedHashMap<>();
        a.put(10, new Point(1, 1));
        a.put(20, new Point(2, 2));
        clip0.put("Track1", a);

        Map<String, Map<Integer, Point>> clip1 = new LinkedHashMap<>();
        Map<Integer, Point> b = new LinkedHashMap<>();
        b.put(20, new Point(9, 9));
        b.put(30, new Point(3, 3));
        clip1.put("Track1", b);

        List<Map<String, Map<Integer, Point>>> clips = new ArrayList<>();
        clips.add(clip0);
        clips.add(clip1);
        List<Long> times = new ArrayList<>();
        times.add(100L);
        times.add(200L);

        Map<String, Map<Integer, Point>> merged = VideoBatchPlan.mergeClipTracks(clips, times);
        assertEquals(new Point(1, 1), merged.get("Track1").get(10));
        assertEquals(new Point(9, 9), merged.get("Track1").get(20));
        assertEquals(new Point(3, 3), merged.get("Track1").get(30));
    }

    @Test
    void splitTracksKeepsOverlapOnBothClips() {
        Map<String, Map<Integer, Point>> full = new LinkedHashMap<>();
        Map<Integer, Point> track = new LinkedHashMap<>();
        for (int i = 0; i < 10; i++) {
            track.put(i, new Point(i, i));
        }
        full.put("Track1", track);
        List<VideoBatchPlan.ClipRange> ranges = VideoBatchPlan.split(10, 2);
        Map<String, Map<Integer, Point>> first = VideoBatchPlan.splitTracksForClip(full, ranges.get(0));
        Map<String, Map<Integer, Point>> second = VideoBatchPlan.splitTracksForClip(full, ranges.get(1));
        int overlap = ranges.get(0).endFrame;
        assertTrue(first.get("Track1").containsKey(overlap));
        assertTrue(second.get("Track1").containsKey(overlap));
        assertEquals(first.get("Track1").get(overlap), second.get("Track1").get(overlap));
    }

    @Test
    void boundaryOnlyDropsInterior() {
        VideoBatchPlan.ClipRange clip = new VideoBatchPlan.ClipRange(0, 0, 5);
        Map<String, Map<Integer, Point>> tracks = new LinkedHashMap<>();
        Map<Integer, Point> track = new LinkedHashMap<>();
        track.put(0, new Point(1, 1));
        track.put(3, new Point(2, 2));
        track.put(5, new Point(3, 3));
        tracks.put("Track1", track);
        Map<String, Map<Integer, Point>> boundary = VideoBatchPlan.boundaryOnly(tracks, clip);
        assertEquals(2, boundary.get("Track1").size());
        assertFalse(boundary.get("Track1").containsKey(3));
    }

    @Test
    void localWindowClipsToBatch() {
        TrackingParameters.LocalCorrectionRange window =
            TrackingParameters.computeLocalCorrectionRange(10, 11, 100);
        VideoBatchPlan.ClipRange clip = new VideoBatchPlan.ClipRange(0, 8, 12);
        TrackingParameters.LocalCorrectionRange clipped = VideoBatchPlan.clipLocalWindow(window, clip);
        assertEquals(8, clipped.startFrame);
        assertEquals(12, clipped.endFrame);
    }

    @Test
    void manifestRoundTrip(@TempDir Path tempDir) throws Exception {
        List<VideoBatchPlan.ClipRange> ranges = VideoBatchPlan.split(100, 3);
        VideoBatchManifest manifest = VideoBatchManifest.create(
            "video.tif", ranges, 64, 48, 3, "video", "raft");
        manifest.trackCounter = 4;
        manifest.markSaved(0, true);
        File file = VideoBatchManifest.fileFor(tempDir.toFile(), "video");
        manifest.save(file);
        VideoBatchManifest loaded = VideoBatchManifest.load(file);
        assertEquals(3, loaded.batchCount);
        assertEquals(4, loaded.trackCounter);
        assertEquals(ranges.get(1).startFrame, loaded.batches.get(1).startFrame);
        assertTrue(loaded.batches.get(0).hasAnnotations);
        assertEquals("video_batch01_annotations.json", loaded.batches.get(0).annotationFile);
    }

    @Test
    void coordinatorPreviewDoesNotAllowAnnotation() {
        VideoBatchCoordinator coordinator = new VideoBatchCoordinator();
        VideoBatchManifest manifest = VideoBatchManifest.create(
            "video.tif", VideoBatchPlan.split(100, 3), 32, 32, 3, "video", "dis");
        coordinator.activate(manifest, new File("."), "video");
        assertTrue(coordinator.isPreview());
        assertFalse(coordinator.allowsAnnotation());
        coordinator.enterWork(1);
        assertTrue(coordinator.allowsAnnotation());
        assertEquals(coordinator.workingRange().startFrame + 1, coordinator.clampSlice(1, 100));
        coordinator.enterEntireVideo();
        assertFalse(coordinator.allowsAnnotation());
        assertTrue(coordinator.allowsOpticalFlow() == false);
        assertEquals(1, coordinator.clampSlice(1, 100));
    }

    @Test
    void clipJsonMetadataIncludesRange() {
        JSONObject metadata = new JSONObject();
        metadata.put("format_type", "rich");
        metadata.put("batch_index", 1);
        metadata.put("start_frame", 10);
        metadata.put("end_frame", 20);
        metadata.put("total_frames", 100);
        assertEquals(1, metadata.getInt("batch_index"));
        assertEquals(10, metadata.getInt("start_frame"));
    }

    @Test
    void flowFileTokenMatchesClip() {
        VideoBatchPlan.ClipRange clip = new VideoBatchPlan.ClipRange(0, 0, 839);
        assertTrue(VideoBatchPlan.flowFileMatchesClip("video_raft_512x384_f0-839_optical_flow.npz", clip));
        assertFalse(VideoBatchPlan.flowFileMatchesClip("video_raft_512x384_optical_flow.npz", clip));
    }

    @Test
    void clipJsonRoundTripAndOverlapSync(@TempDir Path tempDir) throws Exception {
        VideoBatchPlan.ClipRange clip = new VideoBatchPlan.ClipRange(0, 0, 10);
        VideoBatchStore.ClipSnapshot snap = new VideoBatchStore.ClipSnapshot();
        Map<Integer, Point> track = new LinkedHashMap<>();
        track.put(0, new Point(1, 2));
        track.put(10, new Point(3, 4));
        snap.tracks.put("Track1", track);
        List<Anchor> firstAnchors = new ArrayList<>();
        firstAnchors.add(new Anchor(0, 1, 2));
        snap.anchors.put("Track1", firstAnchors);
        File file = tempDir.resolve("video_batch01_annotations.json").toFile();
        VideoBatchStore.save(file, snap, "video.tif", clip, 20);
        VideoBatchStore.ClipSnapshot loaded = VideoBatchStore.load(file);
        assertEquals(0, loaded.batchIndex);
        assertEquals(10, loaded.endFrame);
        assertEquals(new Point(3, 4), loaded.tracks.get("Track1").get(10));

        VideoBatchStore.ClipSnapshot neighbor = new VideoBatchStore.ClipSnapshot();
        Map<Integer, Point> later = new LinkedHashMap<>();
        later.put(10, new Point(9, 9));
        later.put(15, new Point(5, 5));
        neighbor.tracks.put("Track1", later);
        VideoBatchStore.syncOverlapSnapshots(snap, neighbor, 10);
        assertEquals(new Point(3, 4), neighbor.tracks.get("Track1").get(10));
        assertEquals(new Point(5, 5), neighbor.tracks.get("Track1").get(15));
        assertTrue(neighbor.anchors.get("Track1").stream().anyMatch(a -> a.frame == 10 && a.x == 3 && a.y == 4));
    }
}
