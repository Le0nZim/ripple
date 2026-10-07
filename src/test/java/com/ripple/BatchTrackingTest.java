package com.ripple;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.jupiter.api.Test;

import java.awt.Point;
import java.io.IOException;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ExecutionException;

import static org.junit.jupiter.api.Assertions.*;

class BatchTrackingTest {
    private final VideoBatchPlan.ClipRange clip = new VideoBatchPlan.ClipRange(1, 10, 13);

    private JSONObject result(String id, int frame) {
        return new JSONObject().put("status", "ok").put("tracks", new JSONArray().put(
            new JSONObject().put("track_id", id).put("frames", new JSONArray().put(
                new JSONObject().put("frame", frame).put("x", 8).put("y", 9)))));
    }

    @Test
    void compressedVideoBatchPreservesSourceNameAndGlobalAnchors() {
        JSONObject request = BatchTracking.request("/tmp/neurons_compressed_123.tif", "neurons", clip,
            List.of("neuron α"), List.of(List.of(new Anchor(10, 8, 9), new Anchor(13, 10, 11))), new Properties());
        assertEquals("neurons", request.getString("video_name"));
        assertEquals(10, request.getInt("frame_start"));
        assertEquals(13, request.getInt("frame_end"));
        assertEquals(10, request.getJSONArray("tracks").getJSONObject(0)
            .getJSONArray("anchors").getJSONObject(0).getInt("frame"));
        assertFalse(request.has("output_path"), "Socket responses must not overwrite saved annotations");
    }

    @Test
    void anchorFromAnotherClipIsRejectedBeforeTracking() {
        assertThrows(IllegalArgumentException.class, () -> BatchTracking.request("video.tif", "video", clip,
            List.of("neuron"), List.of(List.of(new Anchor(0, 8, 9))), new Properties()));
    }

    @Test
    void parsesGlobalTrackCoordinates() throws Exception {
        assertEquals(new Point(8, 9), BatchTracking.parseResults(result("neuron α", 11),
            List.of("neuron α"), clip).get("neuron α").get(11));
    }

    @Test
    void rejectsMissingEmptyDuplicateAndUnexpectedTracks() {
        assertThrows(IOException.class, () -> BatchTracking.parseResults(result("neuron", 11),
            List.of("neuron", "missing"), clip));
        assertThrows(IOException.class, () -> BatchTracking.parseResults(result("unexpected", 11),
            List.of("neuron"), clip));
        JSONObject empty = result("neuron", 11);
        empty.getJSONArray("tracks").getJSONObject(0).put("frames", new JSONArray());
        assertThrows(IOException.class, () -> BatchTracking.parseResults(empty, List.of("neuron"), clip));
        JSONObject duplicate = result("neuron", 11);
        duplicate.getJSONArray("tracks").put(duplicate.getJSONArray("tracks").getJSONObject(0));
        assertThrows(IOException.class, () -> BatchTracking.parseResults(duplicate, List.of("neuron"), clip));
    }

    @Test
    void rejectsResultFromWrongClipAndDuplicateFrames() {
        assertThrows(IOException.class, () -> BatchTracking.parseResults(result("neuron", 0),
            List.of("neuron"), clip));
        JSONObject duplicate = result("neuron", 11);
        JSONArray frames = duplicate.getJSONArray("tracks").getJSONObject(0).getJSONArray("frames");
        frames.put(frames.getJSONObject(0));
        assertThrows(IOException.class, () -> BatchTracking.parseResults(duplicate, List.of("neuron"), clip));
    }

    @Test
    void surfacesBackendErrorWithoutSwingWorkerWrapper() {
        IOException error = assertThrows(IOException.class, () -> BatchTracking.parseResults(
            new JSONObject().put("status", "error").put("message", "Clip flow is missing"), List.of("neuron"), clip));
        assertEquals("Clip flow is missing", BatchTracking.errorMessage(new ExecutionException(error)));
    }
}
