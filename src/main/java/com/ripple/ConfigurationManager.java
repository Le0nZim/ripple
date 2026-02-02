package com.ripple;

import java.io.*;
import java.util.Properties;

/**
 * Manages application configuration settings and provides centralized access
 * to system parameters including file paths, processing options, and tracking modes.
 */
public class ConfigurationManager {
    private final Properties config;
    private final String configFilePath;
    
    public ConfigurationManager(String configFilePath) {
        this.configFilePath = configFilePath;
        this.config = new Properties();
        loadConfig();
    }
    
    /**
     * Loads configuration parameters from the specified file.
     * If the file does not exist, initializes with default values.
     */
    private void loadConfig() {
        File configFile = new File(configFilePath);
        if (configFile.exists()) {
            try (FileInputStream fis = new FileInputStream(configFile)) {
                config.load(fis);
                System.out.println("Configuration loaded from " + configFilePath);
            } catch (IOException e) {
                System.err.println("Failed to load config: " + e.getMessage());
                loadDefaultConfig();
            }
        } else {
            loadDefaultConfig();
        }
    }
    
    /**
     * Initializes configuration with default parameter values and persists them to disk.
     */
    private void loadDefaultConfig() {
        config.setProperty("bash.script.path", Constants.DEFAULT_BASH_SCRIPT_PATH);
        config.setProperty("local.python.script", 
            "./src/main/resources/runtime/tracking_server.py");  // Replace with your tracking script path
        config.setProperty("local.work.dir", Constants.DEFAULT_WORK_DIR);
        config.setProperty("local.conda.env", Constants.DEFAULT_CONDA_ENV);
        config.setProperty("local.socket.path", Constants.DEFAULT_SOCKET_PATH);
        config.setProperty("local.tcp.host", Constants.DEFAULT_TCP_HOST);
        config.setProperty("local.tcp.port", Constants.DEFAULT_TCP_PORT);
        config.setProperty("raft.model.size", Constants.DEFAULT_MODEL_SIZE);
        config.setProperty("local.timeout", Constants.DEFAULT_TIMEOUT);
        config.setProperty("output.directory", Constants.DEFAULT_OUTPUT_DIR);
        config.setProperty("auto.compute.optical.flow", "true");
        config.setProperty("corridor.width", Constants.DEFAULT_CORRIDOR_WIDTH);
        config.setProperty("tracking.mode", Constants.DEFAULT_TRACKING_MODE);
        saveConfig();
    }
    
    /**
     * Persists current configuration state to the designated configuration file.
     */
    public void saveConfig() {
        try (FileOutputStream fos = new FileOutputStream(configFilePath)) {
            config.store(fos, "Native Linux RAFT Configuration (Local Execution)");
        } catch (IOException e) {
            System.err.println("Failed to save config: " + e.getMessage());
        }
    }
    
    /**
     * Retrieves a configuration property value by key.
     * @param key The configuration parameter identifier
     * @param defaultValue The fallback value if the key is not found
     * @return The configuration value or default if unavailable
     */
    public String getProperty(String key, String defaultValue) {
        return config.getProperty(key, defaultValue);
    }
    
    /**
     * Sets a configuration property to a specified value.
     * @param key The configuration parameter identifier
     * @param value The value to assign to the parameter
     */
    public void setProperty(String key, String value) {
        config.setProperty(key, value);
    }
    
    /**
     * Retrieves a configuration property as a boolean value.
     * @param key The configuration parameter identifier
     * @param defaultValue The fallback boolean value if the key is not found
     * @return The configuration value parsed as boolean or default if unavailable
     */
    public boolean getBooleanProperty(String key, boolean defaultValue) {
        return Boolean.parseBoolean(config.getProperty(key, String.valueOf(defaultValue)));
    }
    
    /**
     * Retrieves a configuration property as an integer value.
     * Provides graceful error handling for malformed numeric strings.
     * @param key The configuration parameter identifier
     * @param defaultValue The fallback integer value if parsing fails
     * @return The configuration value parsed as integer or default if parsing fails
     */
    public int getIntProperty(String key, int defaultValue) {
        try {
            return Integer.parseInt(config.getProperty(key, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
    
    /**
     * Determines whether the current tracking mode is configured as multi-seed.
     * @return true if multi-seed tracking mode is active, false otherwise
     */
    public boolean isMultiSeedMode() {
        return Constants.MODE_MULTI_SEED.equalsIgnoreCase(
            config.getProperty("tracking.mode", Constants.DEFAULT_TRACKING_MODE));
    }
    
    /**
     * Resolves the output directory path with template variable substitution.
     * Supports placeholders: {VIDEO_DIR} for video directory, {VIDEO_NAME} for video filename.
     * @param videoPath The absolute path to the video file for template resolution
     * @return The resolved absolute output directory path
     */
    public String getOutputDirectory(String videoPath) {
        String outputDirTemplate = getProperty("output.directory", Constants.DEFAULT_OUTPUT_DIR);
        
        if (videoPath == null) {
            return System.getProperty("user.dir");
        }
        
        File videoFile = new File(videoPath);
        String videoDir = videoFile.getParent() != null ? videoFile.getParent() : System.getProperty("user.dir");
        String videoName = videoFile.getName();
        if (videoName.contains(".")) {
            videoName = videoName.substring(0, videoName.lastIndexOf('.'));
        }
        
        return outputDirTemplate
            .replace("{VIDEO_DIR}", videoDir)
            .replace("{VIDEO_NAME}", videoName);
    }
    
    /**
     * Constructs the expected optical flow cache file path for a given video.
     * @param videoPath The absolute path to the video file
     * @return The absolute path where optical flow data should be stored, or null if videoPath is null
     */
    public String getOpticalFlowPath(String videoPath) {
        if (videoPath == null) return null;
        
        String outputDir = getOutputDirectory(videoPath);
        String videoName = new File(videoPath).getName();
        String baseName = videoName.replaceFirst("\\.[^.]+$", "");
        return new File(outputDir, baseName + Constants.FLOW_SUFFIX).getAbsolutePath();
    }
}
