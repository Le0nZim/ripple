package com.ripple;

import java.awt.Color;

/**
 * Application-wide constant definitions for the TIFF Video Annotation Tool.
 * Centralizes configuration parameters, UI specifications, and system defaults.
 */
public class Constants {
    
    // Configuration parameters and default values
    public static final String CONFIG_FILE = "config.properties";
    public static final String DEFAULT_BASH_SCRIPT_PATH = "scripts/run_persistent_tracking.sh";
    public static final String DEFAULT_WORK_DIR = "/tmp/tracking_temp";
    public static final String DEFAULT_CONDA_ENV = "ripple-env";
    public static final String DEFAULT_SOCKET_PATH = "/tmp/ripple-env.sock";
    public static final String DEFAULT_TCP_HOST = "127.0.0.1";
    public static final String DEFAULT_TCP_PORT = "9876";
    public static final String DEFAULT_MODEL_SIZE = "large";
    public static final String DEFAULT_TIMEOUT = "300";
    public static final String DEFAULT_OUTPUT_DIR = "{VIDEO_DIR}";
    public static final String DEFAULT_CORRIDOR_WIDTH = "0";
    public static final String DEFAULT_TRACKING_MODE = "single-seed";
    
    // Linear interpolation threshold: if two anchors are within N frames, skip optical flow and use linear interpolation
    // This helps avoid jitter/erratic displacement from unreliable optical flow in short segments
    public static final int DEFAULT_LINEAR_INTERP_THRESHOLD = 0;  // 0 = disabled (always use optical flow)
    
    // User interface color scheme definitions
    public static final Color UI_BACKGROUND = new Color(250, 250, 250);
    public static final Color UI_BORDER = new Color(200, 200, 200);
    public static final Color UI_PANEL_BG = new Color(245, 245, 245);
    public static final Color UI_SELECTION = new Color(230, 240, 255);
    public static final Color UI_SELECTION_BORDER = new Color(115, 164, 209);
    public static final Color UI_TEXT = new Color(60, 60, 60);
    public static final Color UI_TEXT_GRAY = new Color(100, 100, 100);
    public static final Color UI_HOVER = new Color(245, 248, 255);
    public static final Color UI_INFO_BG = new Color(250, 250, 240);
    public static final Color UI_INFO_BORDER = new Color(200, 200, 180);
    public static final Color UI_WHITE = Color.WHITE;
    public static final Color UI_BLACK = Color.BLACK;
    
    // User interface dimensional specifications
    public static final int DEFAULT_WINDOW_WIDTH = 1200;
    public static final int DEFAULT_WINDOW_HEIGHT = 750;
    public static final int ANNOTATION_PANEL_WIDTH = 320;
    public static final int ANNOTATION_ROW_PADDING = 10;
    public static final int COLOR_BOX_SIZE = 24;
    
    // Temporal navigation parameters
    public static final int FRAME_JUMP_AMOUNT = 10;
    public static final int MAX_FPS = 50;
    
    // Viewport zoom control parameters
    public static final double MIN_ZOOM = 0.05;
    public static final double MAX_ZOOM = 100.0;
    public static final double ZOOM_IN_FACTOR = 1.25;
    public static final double ZOOM_OUT_FACTOR = 0.8;
    
    // Trajectory visualization color specifications
    public static final int TRACK_COLOR_MIN = 50;
    public static final int TRACK_COLOR_MAX = 255;
    public static final int TRACK_COLOR_ALPHA = 200;
    public static final int TRACK_HOVER_ALPHA = 150;
    
    // Trajectory rendering parameters
    public static final int MIN_POINT_SIZE = 1;
    public static final int SELECTION_BOX_MULTIPLIER = 3;
    
    // Supported file format specifications
    public static final String[] SUPPORTED_VIDEO_EXTENSIONS = {"tif", "tiff", "avi"};
    public static final String JSON_EXTENSION = "json";
    public static final String NPZ_EXTENSION = "npz";
    public static final String FLOW_SUFFIX = "_optical_flow.npz";
    public static final String ANNOTATION_SUFFIX = "_annotations.json";
    public static final String FLOW_VIZ_SUFFIX = "_flow_viz.tif";
    
    // AVI format processing specifications
    public static final int AVI_TARGET_WIDTH = 768;
    public static final int AVI_TARGET_HEIGHT = 480;
    
    // Histogram-based contrast adjustment parameters
    public static final double PERCENTILE_LOW = 0.005;
    public static final double PERCENTILE_HIGH = 0.995;
    public static final int BRIGHTNESS_SAMPLE_STEP = 4;
    public static final int BRIGHTNESS_SLIDER_MAX = 1000;
    
    // Progress indicator specifications
    public static final int PROGRESS_DIALOG_WIDTH = 450;
    public static final int PROGRESS_DIALOG_HEIGHT = 180;
    
    // Typography size specifications
    public static final int FONT_SIZE_TITLE = 15;
    public static final int FONT_SIZE_NORMAL = 13;
    public static final int FONT_SIZE_SMALL = 12;
    public static final int FONT_SIZE_TINY = 11;
    
    // Tracking algorithm mode identifiers
    public static final String MODE_SINGLE_SEED = "single-seed";
    public static final String MODE_MULTI_SEED = "multi-seed";
    
    // Spatial search corridor configuration options
    public static final String CORRIDOR_AUTO = "auto";
    public static final String CORRIDOR_FULL = "full";
    public static final String CORRIDOR_CUSTOM = "custom";
    
    // Inter-process communication command identifiers
    public static final String CMD_COMPUTE_FLOW = "compute_flow";
    public static final String CMD_COMPUTE_LOCOTRACK_FLOW = "compute_locotrack_flow";
    public static final String CMD_PREVIEW_DOG_DETECTION = "preview_dog_detection";
    public static final String CMD_TRACK_SEED = "track_seed";
    public static final String CMD_TRACK_ANCHORS = "track_anchors";
    public static final String CMD_OPTIMIZE_TRACKS = "optimize_tracks";
    public static final String CMD_VISUALIZE_FLOW = "visualize_flow";
    
    // Optical flow method identifiers
    public static final String FLOW_METHOD_RAFT = "raft";
    public static final String FLOW_METHOD_LOCOTRACK = "locotrack";
    public static final String FLOW_METHOD_TRACKPY = "trackpy";
    public static final String FLOW_METHOD_DIS = "dis";
    public static final String DEFAULT_FLOW_METHOD = FLOW_METHOD_RAFT;
    
    // Execution mode identifiers (set by launcher via environment variable)
    public static final String ENV_RIPPLE_MODE = "RIPPLE_MODE";
    public static final String MODE_GPU = "gpu";
    public static final String MODE_CPU = "cpu";
    
    // GPU-only features (RAFT and LocoTrack require CUDA)
    public static final String[] GPU_ONLY_FLOW_METHODS = {FLOW_METHOD_RAFT, FLOW_METHOD_LOCOTRACK};
    
    // CPU-compatible features
    public static final String[] CPU_FLOW_METHODS = {FLOW_METHOD_DIS, FLOW_METHOD_TRACKPY};
    
    // Correction method identifiers
    public static final String CORRECTION_METHOD_FULL_BLEND = "full_blend";
    public static final String CORRECTION_METHOD_CORRIDOR_DP = "corridor_dp";
    public static final String CORRECTION_METHOD_BLOB_ASSISTED = "blob_assisted";
    
    // All correction methods are CPU-based (Corridor-DP uses dynamic programming on CPU with any flow data)
    public static final String[] ALL_CORRECTION_METHODS = {CORRECTION_METHOD_FULL_BLEND, CORRECTION_METHOD_CORRIDOR_DP, CORRECTION_METHOD_BLOB_ASSISTED};
    
    // Interpolation kernel identifiers
    public static final String KERNEL_GAUSSIAN_RBF = "gaussian_rbf";
    public static final String KERNEL_GAUSSIAN = "gaussian";
    public static final String KERNEL_THIN_PLATE_SPLINE = "thin_plate_spline";
    public static final String KERNEL_IDW = "idw";
    public static final String KERNEL_WENDLAND = "wendland";
    public static final String KERNEL_MULTIQUADRIC = "multiquadric";
    
    // GPU-only interpolation kernels (use PyTorch GPU acceleration)
    public static final String[] GPU_ONLY_KERNELS = {KERNEL_GAUSSIAN, KERNEL_IDW, KERNEL_WENDLAND};
    
    // CPU-compatible interpolation kernels (use scipy RBF on CPU)
    public static final String[] CPU_KERNELS = {KERNEL_GAUSSIAN_RBF, KERNEL_THIN_PLATE_SPLINE, KERNEL_MULTIQUADRIC};
    
    // LocoTrack/DoG detector default parameters (TrackMate-style)
    public static final double DEFAULT_DOG_RADIUS = 2.5;  // Estimated object radius in pixels
    public static final double DEFAULT_DOG_THRESHOLD = 0.0;  // Quality threshold (0 = accept all)
    public static final boolean DEFAULT_DOG_MEDIAN_FILTER = false;  // Pre-process with median filter
    public static final boolean DEFAULT_DOG_SUBPIXEL = true;  // Sub-pixel localization
    public static final double DEFAULT_OCCLUSION_THRESHOLD = 0.5;
    public static final double DEFAULT_LOCOTRACK_WEIGHT = 0.5;
    public static final double DEFAULT_FLOW_SMOOTHING = 15.0;
    public static final double DEFAULT_TEMPORAL_SMOOTH_FACTOR = 0.1;
    public static final String DEFAULT_INTERPOLATION_KERNEL = "gaussian";
    
    // User notification message templates
    public static final String MSG_OPTICAL_FLOW_REQUIRED = 
        "Optical flow is required for tracking.\nWould you like to compute it now?";
    public static final String MSG_NAVIGATE_TO_FRAME_1 = 
        "Please create a track first by clicking the '+ New Track' button,\nthen click on any frame to initialize it.";
    public static final String MSG_CREATE_TRACK_FIRST = 
        "Please create or select a track first by clicking the '+ New Track' button.";
    public static final String MSG_NO_TRACK_SELECTED = 
        "No track selected.";
    
    // Visual indicator symbols for user interface elements
    public static final String ANCHOR_ICON = "📍";
    public static final String CHECKMARK_ICON = "✓";
    
    // LocoTrack fine-tuning constants (GPU only)
    public static final String CMD_FINETUNE_LOCOTRACK = "finetune_locotrack";
    public static final int DEFAULT_FINETUNE_EPOCHS = 100;
    public static final int DEFAULT_FINETUNE_BATCH_SIZE = 1;
    public static final double DEFAULT_FINETUNE_LR = 1e-4;
    public static final double DEFAULT_TRAIN_TEST_SPLIT = 0.85;  // 85% training, 15% testing
    public static final int MIN_TRACKS_FOR_FINETUNING = 4;  // Need at least 4 tracks for meaningful train/test split
    
    // Turbo fine-tuning constants (fast LoRA-based adaptation)
    public static final int DEFAULT_TURBO_ITERATIONS = 100;
    public static final double DEFAULT_TURBO_LR = 1e-4;
    public static final int DEFAULT_TURBO_BATCH_SIZE = 4;
    
    // Default fine-tuning weights directory (canonical location in locotrack_pytorch)
    public static final String DEFAULT_WEIGHTS_DIR = "locotrack_pytorch/weights";
    
    private Constants() {
        // Utility class: constructor is private to prevent instantiation
    }
}
