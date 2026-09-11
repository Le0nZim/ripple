package com.ripple;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.Color;
import java.awt.Point;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class VideoBatchStoreTest {

    @Test
    void roundTripsRichClipMetadata(@TempDir Path tempDir) throws Exception {
        VideoBatchStore.ClipSnapshot snap = new VideoBatchStore.ClipSnapshot();
        HashMap<Integer, Point> points = new HashMap<>();
        points.put(10, new Point(4, 5));
        points.put(20, new Point(6, 7));
        snap.tracks.put("Track1", points);
        List<Anchor> anchors = new ArrayList<>();
        anchors.add(new Anchor(10, 4, 5));
        snap.anchors.put("Track1", anchors);
        snap.colors.put("Track1", new Color(10, 20, 30, 200));
        snap.optimized.put("Track1", true);
        snap.completed.put("Track1", true);
        snap.timeMs.put("Track1", 1234L);
        snap.smoothing.put("Track1", true);
        List<int[]> occlusions = new ArrayList<>();
        occlusions.add(new int[]{12, 14, 2});
        snap.occlusions.put("Track1", occlusions);
        snap.trimRange.put("Track1", new int[]{10, 20});
        HashMap<Integer, Point> untrimmed = new HashMap<>();
        untrimmed.put(8, new Point(1, 2));
        snap.untrimmedAnnotations.put("Track1", untrimmed);
        List<Anchor> untrimmedAnchors = new ArrayList<>();
        untrimmedAnchors.add(new Anchor(8, 1, 2));
        snap.untrimmedAnchors.put("Track1", untrimmedAnchors);

        File file = tempDir.resolve("clip.json").toFile();
        VideoBatchStore.save(file, snap, "cells.tif", new VideoBatchPlan.ClipRange(0, 0, 40), 100);

        VideoBatchStore.ClipSnapshot loaded = VideoBatchStore.load(file);
        assertEquals(new Point(6, 7), loaded.tracks.get("Track1").get(20));
        assertEquals(1, loaded.anchors.get("Track1").size());
        assertEquals(10, loaded.colors.get("Track1").getRed());
        assertTrue(loaded.optimized.get("Track1"));
        assertTrue(loaded.completed.get("Track1"));
        assertEquals(1234L, loaded.timeMs.get("Track1"));
        assertTrue(loaded.smoothing.get("Track1"));
        assertArrayEquals(new int[]{12, 14, 2}, loaded.occlusions.get("Track1").get(0));
        assertArrayEquals(new int[]{10, 20}, loaded.trimRange.get("Track1"));
        assertEquals(new Point(1, 2), loaded.untrimmedAnnotations.get("Track1").get(8));
        assertEquals(new Anchor(8, 1, 2), loaded.untrimmedAnchors.get("Track1").get(0));
    }

    @Test
    void loadsLegacyClipFilesWithoutRichFields(@TempDir Path tempDir) throws Exception {
        String legacy = "{\n"
            + "  \"metadata\": {\"batch_index\": 1, \"start_frame\": 10, \"end_frame\": 20},\n"
            + "  \"tracks\": [{\n"
            + "    \"track_id\": \"Track3\",\n"
            + "    \"annotations\": [{\"frame\": 12, \"x\": 3, \"y\": 4}],\n"
            + "    \"optimized\": false\n"
            + "  }]\n"
            + "}\n";
        File file = tempDir.resolve("legacy.json").toFile();
        Files.write(file.toPath(), legacy.getBytes(StandardCharsets.UTF_8));

        VideoBatchStore.ClipSnapshot loaded = VideoBatchStore.load(file);
        assertEquals(1, loaded.batchIndex);
        assertEquals(new Point(3, 4), loaded.tracks.get("Track3").get(12));
        assertTrue(loaded.occlusions.isEmpty());
        assertFalse(loaded.completed.getOrDefault("Track3", false));
        assertFalse(loaded.smoothing.getOrDefault("Track3", false));
    }
}
