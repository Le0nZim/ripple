package com.ripple;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Recent videos, remembered chooser folders, and UI toggles stored in config.properties.
 */
public final class UserSessionPreferences {

    public static final int MAX_RECENT = 8;
    public static final String KEY_RECENT = "ui.recent.videos";
    public static final String KEY_LAST_DIR_VIDEO = "ui.last.dir.video";
    public static final String KEY_LAST_DIR_EXPORT = "ui.last.dir.export";
    public static final String KEY_LAST_DIR_IMPORT = "ui.last.dir.import";
    public static final String KEY_FOLLOW_TRACK = "ui.follow.track";
    public static final String KEY_ONION_SKIN = "ui.onion.skin";
    public static final String AUTOSAVE_SUFFIX = "_ripple_autosave.json";

    private UserSessionPreferences() {
    }

    public static List<String> parseRecent(String raw) {
        List<String> paths = new ArrayList<>();
        if (raw == null || raw.trim().isEmpty()) {
            return paths;
        }
        for (String part : raw.split("\\|")) {
            String path = part.trim();
            if (!path.isEmpty() && !paths.contains(path)) {
                paths.add(path);
            }
        }
        return paths;
    }

    public static String encodeRecent(List<String> paths) {
        if (paths == null || paths.isEmpty()) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        int count = 0;
        for (String path : paths) {
            if (path == null || path.trim().isEmpty()) {
                continue;
            }
            if (count > 0) {
                builder.append('|');
            }
            builder.append(path.trim());
            count++;
            if (count >= MAX_RECENT) {
                break;
            }
        }
        return builder.toString();
    }

    public static List<String> addRecent(List<String> existing, String path) {
        List<String> next = new ArrayList<>();
        if (path != null && !path.trim().isEmpty()) {
            next.add(path.trim());
        }
        if (existing != null) {
            for (String candidate : existing) {
                if (candidate == null || candidate.trim().isEmpty()) {
                    continue;
                }
                if (next.contains(candidate)) {
                    continue;
                }
                next.add(candidate);
                if (next.size() >= MAX_RECENT) {
                    break;
                }
            }
        }
        return next;
    }

    public static File directoryOrNull(String path) {
        if (path == null || path.trim().isEmpty()) {
            return null;
        }
        File dir = new File(path);
        return dir.isDirectory() ? dir : null;
    }

    public static void writeDirectories(Properties config, File videoDir, File exportDir, File importDir) {
        if (config == null) {
            return;
        }
        putDir(config, KEY_LAST_DIR_VIDEO, videoDir);
        putDir(config, KEY_LAST_DIR_EXPORT, exportDir);
        putDir(config, KEY_LAST_DIR_IMPORT, importDir);
    }

    public static File autosaveFile(File videoFile) {
        if (videoFile == null) {
            return null;
        }
        String name = videoFile.getName();
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        File parent = videoFile.getParentFile();
        return parent == null ? new File(base + AUTOSAVE_SUFFIX) : new File(parent, base + AUTOSAVE_SUFFIX);
    }

    public static boolean getBoolean(Properties config, String key, boolean defaultValue) {
        if (config == null) {
            return defaultValue;
        }
        return Boolean.parseBoolean(config.getProperty(key, String.valueOf(defaultValue)));
    }

    private static void putDir(Properties config, String key, File dir) {
        if (dir != null && dir.isDirectory()) {
            config.setProperty(key, dir.getAbsolutePath());
        }
    }
}
