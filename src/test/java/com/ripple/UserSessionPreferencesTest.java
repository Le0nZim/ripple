package com.ripple;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class UserSessionPreferencesTest {

    @Test
    void recentListMovesExistingPathToFrontAndCapsAtEight() {
        List<String> existing = Arrays.asList(
            "/a", "/b", "/c", "/d", "/e", "/f", "/g", "/h");
        List<String> next = UserSessionPreferences.addRecent(existing, "/c");
        assertEquals(Arrays.asList("/c", "/a", "/b", "/d", "/e", "/f", "/g", "/h"), next);
        assertEquals("/new|/a|/b|/c|/d|/e|/f|/g", UserSessionPreferences.encodeRecent(
            UserSessionPreferences.addRecent(existing, "/new")));
    }

    @Test
    void parseRecentIgnoresBlanks() {
        List<String> parsed = UserSessionPreferences.parseRecent(" /one | |/two| /one ");
        assertEquals(Arrays.asList("/one", "/two"), parsed);
    }

    @Test
    void autosaveFileUsesVideoBaseName(@TempDir Path tempDir) {
        File video = tempDir.resolve("cells.tif").toFile();
        File autosave = UserSessionPreferences.autosaveFile(video);
        assertEquals("cells_ripple_autosave.json", autosave.getName());
        assertEquals(tempDir.toFile(), autosave.getParentFile());
    }
}
