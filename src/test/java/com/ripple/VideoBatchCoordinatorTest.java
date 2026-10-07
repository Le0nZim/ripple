package com.ripple;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.Point;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class VideoBatchCoordinatorTest {
    @Test
    void savesWorkingEditsAfterPreviewingAnotherClipOrEntireVideo(@TempDir Path directory) throws Exception {
        VideoBatchManifest manifest = VideoBatchManifest.create("neurons.tif", List.of(
            new VideoBatchPlan.ClipRange(0, 0, 10), new VideoBatchPlan.ClipRange(1, 10, 20)),
            32, 32, 2, "neurons", "dis");
        VideoBatchCoordinator coordinator = new VideoBatchCoordinator();
        coordinator.activate(manifest, directory.toFile(), "neurons");
        coordinator.enterWork(0);
        VideoBatchStore.ClipSnapshot edits = new VideoBatchStore.ClipSnapshot();
        edits.tracks.put("neuron", new LinkedHashMap<>());
        edits.tracks.get("neuron").put(5, new Point(8, 9));
        coordinator.enterPreview(1);
        coordinator.saveWorkingSnapshot(edits, "neurons.tif", 21);
        assertEquals(new Point(8, 9), VideoBatchStore.load(coordinator.annotationFile(0))
            .tracks.get("neuron").get(5));
        assertFalse(coordinator.annotationFile(1).exists(), "Preview must not redirect the save to another clip");
        assertTrue(VideoBatchManifest.load(coordinator.manifestFile()).entryAt(0).hasAnnotations);

        edits.tracks.get("neuron").put(5, new Point(12, 13));
        coordinator.enterEntireVideo();
        coordinator.saveWorkingSnapshot(edits, "neurons.tif", 21);
        assertEquals(new Point(12, 13), VideoBatchStore.load(coordinator.annotationFile(0))
            .tracks.get("neuron").get(5));
    }
}
