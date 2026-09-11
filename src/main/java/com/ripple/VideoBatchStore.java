package com.ripple;

import org.json.JSONArray;
import org.json.JSONObject;

import java.awt.Color;
import java.awt.Point;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Clip annotation JSON I/O and overlap bookkeeping.
 */
public final class VideoBatchStore {

    public static final class ClipSnapshot {
        public final Map<String, Map<Integer, Point>> tracks = new LinkedHashMap<>();
        public final Map<String, List<Anchor>> anchors = new LinkedHashMap<>();
        public final Map<String, Color> colors = new LinkedHashMap<>();
        public final Map<String, Boolean> optimized = new LinkedHashMap<>();
        public int batchIndex = -1;
        public int startFrame = -1;
        public int endFrame = -1;
        public int totalFrames = 0;

        public boolean hasAnnotations() {
            for (Map<Integer, Point> points : tracks.values()) {
                if (points != null && !points.isEmpty()) {
                    return true;
                }
            }
            return false;
        }
    }

    private VideoBatchStore() {
    }

    public static ClipSnapshot load(File file) throws Exception {
        ClipSnapshot snapshot = new ClipSnapshot();
        if (file == null || !file.exists()) {
            return snapshot;
        }
        String content = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
        if (content.trim().isEmpty()) {
            return snapshot;
        }
        JSONObject root = new JSONObject(content);
        JSONObject metadata = root.optJSONObject("metadata");
        if (metadata != null) {
            snapshot.batchIndex = metadata.optInt("batch_index", -1);
            snapshot.startFrame = metadata.optInt("start_frame", -1);
            snapshot.endFrame = metadata.optInt("end_frame", -1);
            snapshot.totalFrames = metadata.optInt("total_frames", 0);
        }
        JSONArray tracks = root.optJSONArray("tracks");
        if (tracks == null) {
            return snapshot;
        }
        for (int i = 0; i < tracks.length(); i++) {
            JSONObject track = tracks.getJSONObject(i);
            String trackId = track.optString("track_id", "");
            if (trackId.isEmpty()) {
                continue;
            }
            Map<Integer, Point> points = readPoints(track.optJSONArray("annotations"));
            if (points.isEmpty()) {
                points = readPoints(track.optJSONArray("frames"));
            }
            if (!points.isEmpty()) {
                snapshot.tracks.put(trackId, points);
            }
            List<Anchor> anchors = readAnchors(track.optJSONArray("anchors"));
            if (!anchors.isEmpty()) {
                snapshot.anchors.put(trackId, anchors);
            }
            JSONObject color = track.optJSONObject("color");
            if (color != null) {
                snapshot.colors.put(trackId, new Color(
                    color.optInt("r", 255),
                    color.optInt("g", 0),
                    color.optInt("b", 0),
                    color.optInt("a", 200)));
            }
            snapshot.optimized.put(trackId, track.optBoolean("optimized", false));
        }
        return snapshot;
    }

    public static void save(
            File file,
            ClipSnapshot snapshot,
            String sourceFilename,
            VideoBatchPlan.ClipRange clip,
            int totalFrames) throws Exception {
        if (file == null || snapshot == null) {
            return;
        }
        JSONObject root = new JSONObject();
        JSONObject metadata = new JSONObject();
        metadata.put("format_type", "rich");
        if (sourceFilename != null) {
            metadata.put("source_filename", sourceFilename);
        }
        metadata.put("total_frames", totalFrames);
        if (clip != null) {
            metadata.put("batch_index", clip.index);
            metadata.put("start_frame", clip.startFrame);
            metadata.put("end_frame", clip.endFrame);
        }
        root.put("metadata", metadata);

        JSONArray tracks = new JSONArray();
        for (Map.Entry<String, Map<Integer, Point>> entry : snapshot.tracks.entrySet()) {
            if (entry.getKey() == null || entry.getValue() == null || entry.getValue().isEmpty()) {
                continue;
            }
            JSONObject track = new JSONObject();
            track.put("track_id", entry.getKey());
            track.put("annotations", writePoints(entry.getValue()));
            List<Anchor> anchors = snapshot.anchors.get(entry.getKey());
            if (anchors != null && !anchors.isEmpty()) {
                track.put("anchors", writeAnchors(anchors));
            }
            Color color = snapshot.colors.get(entry.getKey());
            if (color != null) {
                JSONObject colorObj = new JSONObject();
                colorObj.put("r", color.getRed());
                colorObj.put("g", color.getGreen());
                colorObj.put("b", color.getBlue());
                colorObj.put("a", color.getAlpha());
                track.put("color", colorObj);
            }
            track.put("optimized", snapshot.optimized.getOrDefault(entry.getKey(), false));
            tracks.put(track);
        }
        root.put("tracks", tracks);
        File parent = file.getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }
        Files.write(file.toPath(), root.toString(2).getBytes(StandardCharsets.UTF_8));
    }

    public static void syncOverlapSnapshots(ClipSnapshot source, ClipSnapshot dest, int overlapFrame) {
        if (source == null || dest == null || overlapFrame < 0) {
            return;
        }
        VideoBatchPlan.syncOverlapFrame(source.tracks, dest.tracks, overlapFrame);
        for (Map.Entry<String, Map<Integer, Point>> entry : source.tracks.entrySet()) {
            String trackId = entry.getKey();
            Point point = entry.getValue() == null ? null : entry.getValue().get(overlapFrame);
            if (trackId == null || point == null) {
                continue;
            }
            List<Anchor> destAnchors = dest.anchors.computeIfAbsent(trackId, k -> new ArrayList<>());
            destAnchors.removeIf(anchor -> anchor != null && anchor.frame == overlapFrame);
            destAnchors.add(new Anchor(overlapFrame, point.x, point.y));
            destAnchors.sort((a, b) -> Integer.compare(a.frame, b.frame));
            if (source.colors.containsKey(trackId) && !dest.colors.containsKey(trackId)) {
                dest.colors.put(trackId, source.colors.get(trackId));
            }
        }
    }

    public static Map<Integer, Point> readPoints(JSONArray array) {
        Map<Integer, Point> points = new LinkedHashMap<>();
        if (array == null) {
            return points;
        }
        for (int i = 0; i < array.length(); i++) {
            JSONObject obj = array.optJSONObject(i);
            if (obj == null) {
                continue;
            }
            points.put(obj.getInt("frame"), new Point(obj.getInt("x"), obj.getInt("y")));
        }
        return points;
    }

    private static JSONArray writePoints(Map<Integer, Point> points) {
        JSONArray array = new JSONArray();
        List<Integer> frames = new ArrayList<>(points.keySet());
        frames.sort(Integer::compareTo);
        for (Integer frame : frames) {
            Point point = points.get(frame);
            if (frame == null || point == null) {
                continue;
            }
            JSONObject obj = new JSONObject();
            obj.put("frame", frame);
            obj.put("x", point.x);
            obj.put("y", point.y);
            array.put(obj);
        }
        return array;
    }

    private static List<Anchor> readAnchors(JSONArray array) {
        List<Anchor> anchors = new ArrayList<>();
        if (array == null) {
            return anchors;
        }
        for (int i = 0; i < array.length(); i++) {
            JSONObject obj = array.optJSONObject(i);
            if (obj == null) {
                continue;
            }
            anchors.add(new Anchor(obj.getInt("frame"), obj.getInt("x"), obj.getInt("y")));
        }
        return anchors;
    }

    private static JSONArray writeAnchors(List<Anchor> anchors) {
        JSONArray array = new JSONArray();
        for (Anchor anchor : anchors) {
            if (anchor == null) {
                continue;
            }
            JSONObject obj = new JSONObject();
            obj.put("frame", anchor.frame);
            obj.put("x", anchor.x);
            obj.put("y", anchor.y);
            array.put(obj);
        }
        return array;
    }
}
