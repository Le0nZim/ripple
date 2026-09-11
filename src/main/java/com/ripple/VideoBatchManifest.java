package com.ripple;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * On-disk clip session: ranges, annotation filenames, and bookkeeping.
 */
public final class VideoBatchManifest {

    public String sourceFilename;
    public int totalFrames;
    public int overlapFrames = VideoBatchPlan.OVERLAP_FRAMES;
    public int batchCount;
    public int workingWidth;
    public int workingHeight;
    public int suggestedBatchCount;
    public int activeBatch = -1;
    public int previewBatch = -1;
    public int trackCounter = 1;
    public String flowMethod;
    public final List<ClipEntry> batches = new ArrayList<>();

    public static final class ClipEntry {
        public int index;
        public int startFrame;
        public int endFrame;
        public String annotationFile;
        public boolean flowComputed;
        public long lastSavedMs;
        public boolean hasAnnotations;

        public VideoBatchPlan.ClipRange toRange() {
            return new VideoBatchPlan.ClipRange(index, startFrame, endFrame);
        }
    }

    public static VideoBatchManifest create(
            String sourceFilename,
            List<VideoBatchPlan.ClipRange> ranges,
            int width,
            int height,
            int suggested,
            String videoBaseName,
            String flowMethod) {
        VideoBatchManifest manifest = new VideoBatchManifest();
        manifest.sourceFilename = sourceFilename;
        manifest.workingWidth = width;
        manifest.workingHeight = height;
        manifest.suggestedBatchCount = suggested;
        manifest.flowMethod = flowMethod;
        if (ranges == null || ranges.isEmpty()) {
            manifest.totalFrames = 0;
            manifest.batchCount = 0;
            return manifest;
        }
        manifest.totalFrames = ranges.get(ranges.size() - 1).endFrame + 1;
        manifest.batchCount = ranges.size();
        for (VideoBatchPlan.ClipRange range : ranges) {
            ClipEntry entry = new ClipEntry();
            entry.index = range.index;
            entry.startFrame = range.startFrame;
            entry.endFrame = range.endFrame;
            entry.annotationFile = VideoBatchPlan.clipAnnotationFileName(videoBaseName, range.index);
            manifest.batches.add(entry);
        }
        return manifest;
    }

    public List<VideoBatchPlan.ClipRange> ranges() {
        List<VideoBatchPlan.ClipRange> ranges = new ArrayList<>(batches.size());
        for (ClipEntry entry : batches) {
            ranges.add(entry.toRange());
        }
        return Collections.unmodifiableList(ranges);
    }

    public ClipEntry entryAt(int index) {
        if (index < 0 || index >= batches.size()) {
            return null;
        }
        return batches.get(index);
    }

    public VideoBatchPlan.ClipRange rangeAt(int index) {
        ClipEntry entry = entryAt(index);
        return entry == null ? null : entry.toRange();
    }

    public File annotationFile(File directory, int clipIndex) {
        ClipEntry entry = entryAt(clipIndex);
        if (entry == null || directory == null) {
            return null;
        }
        return new File(directory, entry.annotationFile);
    }

    public void markSaved(int clipIndex, boolean hasAnnotations) {
        ClipEntry entry = entryAt(clipIndex);
        if (entry == null) {
            return;
        }
        entry.lastSavedMs = System.currentTimeMillis();
        entry.hasAnnotations = hasAnnotations;
    }

    public JSONObject toJson() {
        JSONObject root = new JSONObject();
        root.put("source_filename", sourceFilename == null ? "" : sourceFilename);
        root.put("total_frames", totalFrames);
        root.put("overlap_frames", overlapFrames);
        root.put("batch_count", batchCount);
        root.put("working_width", workingWidth);
        root.put("working_height", workingHeight);
        root.put("suggested_batch_count", suggestedBatchCount);
        root.put("active_batch", activeBatch);
        root.put("preview_batch", previewBatch);
        root.put("track_counter", trackCounter);
        if (flowMethod != null) {
            root.put("flow_method", flowMethod);
        }
        JSONArray array = new JSONArray();
        for (ClipEntry entry : batches) {
            JSONObject obj = new JSONObject();
            obj.put("index", entry.index);
            obj.put("start_frame", entry.startFrame);
            obj.put("end_frame", entry.endFrame);
            obj.put("annotation_file", entry.annotationFile);
            obj.put("flow_computed", entry.flowComputed);
            obj.put("last_saved_ms", entry.lastSavedMs);
            obj.put("has_annotations", entry.hasAnnotations);
            array.put(obj);
        }
        root.put("batches", array);
        return root;
    }

    public static VideoBatchManifest fromJson(JSONObject root) {
        VideoBatchManifest manifest = new VideoBatchManifest();
        if (root == null) {
            return manifest;
        }
        manifest.sourceFilename = root.optString("source_filename", "");
        manifest.totalFrames = root.optInt("total_frames", 0);
        manifest.overlapFrames = root.optInt("overlap_frames", VideoBatchPlan.OVERLAP_FRAMES);
        manifest.batchCount = root.optInt("batch_count", 0);
        manifest.workingWidth = root.optInt("working_width", 0);
        manifest.workingHeight = root.optInt("working_height", 0);
        manifest.suggestedBatchCount = root.optInt("suggested_batch_count", manifest.batchCount);
        manifest.activeBatch = root.optInt("active_batch", -1);
        manifest.previewBatch = root.optInt("preview_batch", -1);
        manifest.trackCounter = Math.max(1, root.optInt("track_counter", 1));
        manifest.flowMethod = root.has("flow_method") ? root.optString("flow_method") : null;
        JSONArray array = root.optJSONArray("batches");
        if (array != null) {
            for (int i = 0; i < array.length(); i++) {
                JSONObject obj = array.getJSONObject(i);
                ClipEntry entry = new ClipEntry();
                entry.index = obj.optInt("index", i);
                entry.startFrame = obj.optInt("start_frame", 0);
                entry.endFrame = obj.optInt("end_frame", entry.startFrame);
                entry.annotationFile = obj.optString("annotation_file",
                    VideoBatchPlan.clipAnnotationFileName("video", entry.index));
                entry.flowComputed = obj.optBoolean("flow_computed", false);
                entry.lastSavedMs = obj.optLong("last_saved_ms", 0L);
                entry.hasAnnotations = obj.optBoolean("has_annotations", false);
                manifest.batches.add(entry);
            }
        }
        if (manifest.batchCount <= 0) {
            manifest.batchCount = manifest.batches.size();
        }
        return manifest;
    }

    public void save(File file) throws Exception {
        if (file == null) {
            throw new IllegalArgumentException("manifest file is required");
        }
        File parent = file.getParentFile();
        if (parent != null) {
            parent.mkdirs();
        }
        Files.write(file.toPath(), toJson().toString(2).getBytes(StandardCharsets.UTF_8));
    }

    public static VideoBatchManifest load(File file) throws Exception {
        String content = new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
        return fromJson(new JSONObject(content));
    }

    public static File fileFor(File directory, String videoBaseName) {
        return new File(directory, VideoBatchPlan.manifestFileName(videoBaseName));
    }
}
