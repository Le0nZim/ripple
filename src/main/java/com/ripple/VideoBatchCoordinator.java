package com.ripple;

import java.awt.Color;
import java.awt.Point;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Clip-session state: preview vs work vs entire-video, without touching Swing.
 */
public final class VideoBatchCoordinator {

    private VideoBatchManifest manifest;
    private VideoBatchPlan.ViewMode mode = VideoBatchPlan.ViewMode.DISABLED;
    private File directory;
    private String videoBaseName;

    private final Map<String, Map<Integer, Point>> previewBoundaryAnnotations = new LinkedHashMap<>();
    private final Map<String, Color> previewColors = new LinkedHashMap<>();
    private final Map<String, Map<Integer, Point>> stitchedAnnotations = new LinkedHashMap<>();
    private final Map<String, Color> stitchedColors = new LinkedHashMap<>();

    public boolean isEnabled() {
        return mode != VideoBatchPlan.ViewMode.DISABLED && manifest != null && manifest.batchCount > 1;
    }

    public boolean isSingleClip() {
        return manifest != null && manifest.batchCount <= 1;
    }

    public VideoBatchPlan.ViewMode getMode() {
        return mode;
    }

    public boolean isPreview() {
        return mode == VideoBatchPlan.ViewMode.CLIP_PREVIEW;
    }

    public boolean isWork() {
        return mode == VideoBatchPlan.ViewMode.CLIP_WORK;
    }

    public boolean isEntireVideo() {
        return mode == VideoBatchPlan.ViewMode.ENTIRE_VIDEO;
    }

    public boolean allowsAnnotation() {
        return !isEnabled() || isWork();
    }

    public boolean allowsOpticalFlow() {
        return !isEnabled() || isWork();
    }

    public boolean allowsSat() {
        return !isEnabled() || isWork();
    }

    public VideoBatchManifest getManifest() {
        return manifest;
    }

    public File getDirectory() {
        return directory;
    }

    public String getVideoBaseName() {
        return videoBaseName;
    }

    public List<VideoBatchPlan.ClipRange> ranges() {
        return manifest == null ? Collections.emptyList() : manifest.ranges();
    }

    public int workingClipIndex() {
        return manifest == null ? -1 : manifest.activeBatch;
    }

    public int previewClipIndex() {
        return manifest == null ? -1 : manifest.previewBatch;
    }

    public VideoBatchPlan.ClipRange workingRange() {
        return manifest == null ? null : manifest.rangeAt(manifest.activeBatch);
    }

    public VideoBatchPlan.ClipRange previewRange() {
        return manifest == null ? null : manifest.rangeAt(manifest.previewBatch);
    }

    public VideoBatchPlan.ClipRange navigationRange() {
        if (!isEnabled()) {
            return null;
        }
        if (isPreview()) {
            return previewRange();
        }
        if (isWork()) {
            return workingRange();
        }
        return null;
    }

    public int clampSlice(int slice1Based, int totalSlices) {
        return VideoBatchPlan.clampSlice1Based(slice1Based, navigationRange(), totalSlices);
    }

    public void activate(VideoBatchManifest next, File directory, String videoBaseName) {
        this.manifest = next;
        this.directory = directory;
        this.videoBaseName = videoBaseName;
        previewBoundaryAnnotations.clear();
        previewColors.clear();
        stitchedAnnotations.clear();
        stitchedColors.clear();
        if (next == null || next.batchCount <= 1) {
            mode = VideoBatchPlan.ViewMode.DISABLED;
            return;
        }
        mode = VideoBatchPlan.ViewMode.CLIP_PREVIEW;
        next.previewBatch = 0;
        next.activeBatch = -1;
    }

    public void disable() {
        manifest = null;
        directory = null;
        videoBaseName = null;
        mode = VideoBatchPlan.ViewMode.DISABLED;
        previewBoundaryAnnotations.clear();
        previewColors.clear();
        stitchedAnnotations.clear();
        stitchedColors.clear();
    }

    public void enterEntireVideo() {
        if (!isEnabled() && (manifest == null || manifest.batchCount <= 1)) {
            return;
        }
        mode = VideoBatchPlan.ViewMode.ENTIRE_VIDEO;
        if (manifest != null) {
            manifest.previewBatch = -1;
        }
    }

    public void enterPreview(int clipIndex) {
        if (manifest == null || manifest.entryAt(clipIndex) == null) {
            return;
        }
        mode = VideoBatchPlan.ViewMode.CLIP_PREVIEW;
        manifest.previewBatch = clipIndex;
    }

    public void enterWork(int clipIndex) {
        if (manifest == null || manifest.entryAt(clipIndex) == null) {
            return;
        }
        mode = VideoBatchPlan.ViewMode.CLIP_WORK;
        manifest.activeBatch = clipIndex;
        manifest.previewBatch = clipIndex;
    }

    public File annotationFile(int clipIndex) {
        return manifest == null ? null : manifest.annotationFile(directory, clipIndex);
    }

    public File manifestFile() {
        if (directory == null) {
            return null;
        }
        return VideoBatchManifest.fileFor(directory, videoBaseName);
    }

    public Map<String, Map<Integer, Point>> getPreviewBoundaryAnnotations() {
        return previewBoundaryAnnotations;
    }

    public Map<String, Color> getPreviewColors() {
        return previewColors;
    }

    public Map<String, Map<Integer, Point>> getStitchedAnnotations() {
        return stitchedAnnotations;
    }

    public Map<String, Color> getStitchedColors() {
        return stitchedColors;
    }

    public void setPreviewBoundaries(
            Map<String, Map<Integer, Point>> tracks,
            Map<String, Color> colors,
            VideoBatchPlan.ClipRange clip) {
        previewBoundaryAnnotations.clear();
        previewColors.clear();
        previewBoundaryAnnotations.putAll(VideoBatchPlan.boundaryOnly(tracks, clip));
        if (colors != null) {
            for (String trackId : previewBoundaryAnnotations.keySet()) {
                Color color = colors.get(trackId);
                if (color != null) {
                    previewColors.put(trackId, color);
                }
            }
        }
    }

    public void setStitched(
            Map<String, Map<Integer, Point>> tracks,
            Map<String, Color> colors) {
        stitchedAnnotations.clear();
        stitchedColors.clear();
        if (tracks != null) {
            for (Map.Entry<String, Map<Integer, Point>> entry : tracks.entrySet()) {
                if (entry.getKey() == null || entry.getValue() == null) {
                    continue;
                }
                Map<Integer, Point> copy = new LinkedHashMap<>();
                for (Map.Entry<Integer, Point> frame : entry.getValue().entrySet()) {
                    if (frame.getKey() != null && frame.getValue() != null) {
                        copy.put(frame.getKey(), new Point(frame.getValue()));
                    }
                }
                stitchedAnnotations.put(entry.getKey(), copy);
            }
        }
        if (colors != null) {
            stitchedColors.putAll(colors);
        }
    }

    public List<Integer> neighborIndexes(int clipIndex) {
        List<Integer> neighbors = new ArrayList<>();
        if (manifest == null) {
            return neighbors;
        }
        if (clipIndex > 0) {
            neighbors.add(clipIndex - 1);
        }
        if (clipIndex >= 0 && clipIndex < manifest.batchCount - 1) {
            neighbors.add(clipIndex + 1);
        }
        return neighbors;
    }

    public int overlapFrame(int fromClip, int toClip) {
        VideoBatchPlan.ClipRange from = manifest == null ? null : manifest.rangeAt(fromClip);
        VideoBatchPlan.ClipRange to = manifest == null ? null : manifest.rangeAt(toClip);
        if (from == null || to == null) {
            return -1;
        }
        if (from.index + 1 == to.index) {
            return from.endFrame;
        }
        if (to.index + 1 == from.index) {
            return to.endFrame;
        }
        return -1;
    }

    public void saveManifest() throws Exception {
        File file = manifestFile();
        if (file != null && manifest != null) {
            manifest.save(file);
        }
    }
}
