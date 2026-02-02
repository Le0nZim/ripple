package com.ripple;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Utility class to load help content from external resource files.
 * This keeps the UI code clean and makes documentation easily editable.
 */
public class HelpContentLoader {
    
    // Help file resource paths (in src/main/resources/help/)
    private static final String HELP_RESOURCE_PATH = "/help/";
    
    // Tab definitions: display name -> resource filename
    private static final Map<String, String> HELP_TABS = new LinkedHashMap<>();
    
    static {
        HELP_TABS.put("Getting Started", "getting_started.txt");
        HELP_TABS.put("Navigation", "navigation.txt");
        HELP_TABS.put("Tracks", "tracks.txt");
        HELP_TABS.put("Detection", "detection.txt");
        HELP_TABS.put("Optical Flow", "optical_flow.txt");
        HELP_TABS.put("SAT", "sat.txt");
        HELP_TABS.put("GPU Features", "gpu_features.txt");
        HELP_TABS.put("Import/Export", "import_export.txt");
        HELP_TABS.put("Shortcuts", "shortcuts.txt");
    }
    
    /**
     * Get all help tab definitions.
     * @return Map of tab name to content
     */
    public static Map<String, String> loadAllHelpContent() {
        Map<String, String> content = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : HELP_TABS.entrySet()) {
            String tabName = entry.getKey();
            String filename = entry.getValue();
            String text = loadHelpFile(filename);
            content.put(tabName, text);
        }
        return content;
    }
    
    /**
     * Load a single help file from resources.
     * @param filename The filename (e.g., "getting_started.txt")
     * @return The file content, or fallback text if not found
     */
    public static String loadHelpFile(String filename) {
        String resourcePath = HELP_RESOURCE_PATH + filename;
        try (InputStream is = HelpContentLoader.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                System.err.println("[HelpContentLoader] Resource not found: " + resourcePath);
                return getDefaultContent(filename);
            }
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(is, StandardCharsets.UTF_8))) {
                return reader.lines().collect(Collectors.joining("\n"));
            }
        } catch (IOException e) {
            System.err.println("[HelpContentLoader] Error reading " + resourcePath + ": " + e.getMessage());
            return getDefaultContent(filename);
        }
    }
    
    /**
     * Convert markdown-style formatting to plain text for display.
     * Handles headers, bullets, and horizontal rules.
     * @param markdown The markdown-formatted text
     * @return Plain text suitable for display
     */
    public static String formatForDisplay(String markdown) {
        if (markdown == null) return "";
        
        StringBuilder result = new StringBuilder();
        String[] lines = markdown.split("\n");
        
        for (String line : lines) {
            // Convert headers (# Header -> HEADER with underline feel)
            if (line.startsWith("# ")) {
                result.append("\n").append(line.substring(2).toUpperCase()).append("\n");
            } else if (line.startsWith("## ")) {
                result.append("\n").append(line.substring(3)).append("\n");
            } else if (line.startsWith("### ")) {
                result.append("\n").append(line.substring(4)).append("\n");
            } else if (line.equals("---")) {
                result.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            } else if (line.startsWith("| ") && line.contains(" | ")) {
                // Table row - convert to aligned format
                String[] cells = line.split("\\|");
                StringBuilder tableRow = new StringBuilder();
                for (int i = 1; i < cells.length; i++) {
                    String cell = cells[i].trim();
                    if (!cell.isEmpty() && !cell.matches("-+")) {
                        if (i == 1) {
                            tableRow.append(String.format("%-22s", cell));
                        } else {
                            tableRow.append(cell);
                        }
                    }
                }
                if (tableRow.length() > 0) {
                    result.append(tableRow.toString().trim()).append("\n");
                }
            } else if (line.startsWith("```")) {
                // Skip code fence markers
            } else {
                result.append(line).append("\n");
            }
        }
        
        return result.toString();
    }
    
    /**
     * Provide default content when resource file is missing.
     */
    private static String getDefaultContent(String filename) {
        String title = filename.replace(".txt", "").replace("_", " ");
        title = title.substring(0, 1).toUpperCase() + title.substring(1);
        return title + "\n\nContent not available.\n\nPlease check that help resources are properly installed.";
    }
}
