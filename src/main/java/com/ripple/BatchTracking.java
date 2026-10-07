package com.ripple;

import org.json.JSONArray;
import org.json.JSONObject;

import java.awt.Point;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

/** The batch request/response contract, shared by socket and script transports. */
final class BatchTracking {
    private BatchTracking() {}

    static JSONObject request(String videoPath, String videoName, VideoBatchPlan.ClipRange range,
                              List<String> trackIds, List<List<Anchor>> anchorsList, Properties config) {
        if (trackIds == null || anchorsList == null || trackIds.size() != anchorsList.size()) {
            throw new IllegalArgumentException("Each track must have an anchor list");
        }
        JSONArray tracks = new JSONArray();
        for (int i = 0; i < trackIds.size(); i++) {
            JSONArray anchors = new JSONArray();
            for (Anchor anchor : anchorsList.get(i)) {
                if (range != null && !range.contains(anchor.frame)) {
                    throw new IllegalArgumentException("Track '" + trackIds.get(i)
                        + "' has an anchor outside the working clip: frame " + anchor.frame);
                }
                anchors.put(new JSONObject().put("frame", anchor.frame).put("x", anchor.x).put("y", anchor.y));
            }
            tracks.put(new JSONObject().put("track_id", trackIds.get(i)).put("anchors", anchors));
        }
        JSONObject request = new JSONObject()
            .put("command", "optimize_tracks")
            .put("video_path", videoPath)
            .put("video_name", videoName)
            .put("tracks", tracks)
            .put("correction_method", config.getProperty("correction.method", "full_blend"))
            .put("blob_search_radius", TrackingParameters.validateBlobSearchRadiusPixels(
                config.getProperty("dis.blob.search.radius", "15"), 15))
            .put("blob_radius", Double.parseDouble(config.getProperty("dis.blob.radius", "5.0")))
            .put("corridor_width", config.getProperty("corridor.width", "adaptive"))
            .put("linear_interp_threshold", Integer.parseInt(
                config.getProperty("correction.linear.interp.threshold", "0")));
        if (range != null) {
            request.put("frame_start", range.startFrame).put("frame_end", range.endFrame);
        }
        return request;
    }

    static Map<String, Map<Integer, Point>> parseResults(JSONObject response, List<String> expected,
                                                        VideoBatchPlan.ClipRange range) throws IOException {
        if (response.has("status") && !"ok".equals(response.optString("status"))) {
            throw new IOException(response.optString("message", "Batch tracking failed"));
        }
        Map<String, Map<Integer, Point>> results = new LinkedHashMap<>();
        JSONArray tracks = response.optJSONArray("tracks");
        if (tracks == null) {
            throw new IOException("Tracking server returned no tracks");
        }
        for (int i = 0; i < tracks.length(); i++) {
            JSONObject track = tracks.getJSONObject(i);
            String id = track.getString("track_id");
            if (!expected.contains(id) || results.containsKey(id)) {
                throw new IOException("Tracking server returned an unexpected or duplicate track: " + id);
            }
            JSONArray frames = track.getJSONArray("frames");
            Map<Integer, Point> points = new LinkedHashMap<>();
            for (int j = 0; j < frames.length(); j++) {
                JSONObject point = frames.getJSONObject(j);
                int frame = point.getInt("frame");
                if (frame < 0 || (range != null && !range.contains(frame)) || points.containsKey(frame)) {
                    throw new IOException("Tracking server returned an invalid frame for " + id + ": " + frame);
                }
                points.put(frame, new Point(point.getInt("x"), point.getInt("y")));
            }
            if (points.isEmpty()) {
                throw new IOException("Tracking server returned an empty track: " + id);
            }
            results.put(id, points);
        }
        for (String id : expected) {
            if (!results.containsKey(id)) {
                throw new IOException("Tracking server response missing track: " + id);
            }
        }
        return results;
    }

    static String errorMessage(Throwable error) {
        while ((error instanceof java.util.concurrent.ExecutionException
                || error instanceof java.util.concurrent.CompletionException) && error.getCause() != null) {
            error = error.getCause();
        }
        String message = error.getMessage();
        return message == null || message.isBlank() ? error.toString() : message;
    }
}
