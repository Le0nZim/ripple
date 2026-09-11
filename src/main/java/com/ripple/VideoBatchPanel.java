package com.ripple;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSlider;
import javax.swing.JTable;
import javax.swing.SwingConstants;
import javax.swing.table.DefaultTableModel;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Frame;
import java.awt.event.ActionListener;
import java.util.List;

/**
 * Split dialog, clip strip, and preview banner for clip batching.
 */
public final class VideoBatchPanel {

    public interface Listener {
        void onPreviewClip(int clipIndex);

        void onViewEntireVideo();

        void onSwitchToClip(int clipIndex);

        void onBackToWorkingClip();
    }

    public static final class SplitResult {
        public final boolean keepSingle;
        public final int clipCount;
        public final boolean cancelled;

        private SplitResult(boolean cancelled, boolean keepSingle, int clipCount) {
            this.cancelled = cancelled;
            this.keepSingle = keepSingle;
            this.clipCount = clipCount;
        }

        public static SplitResult cancelled() {
            return new SplitResult(true, true, 1);
        }

        public static SplitResult single() {
            return new SplitResult(false, true, 1);
        }

        public static SplitResult split(int clipCount) {
            return new SplitResult(false, false, clipCount);
        }
    }

    private VideoBatchPanel() {
    }

    public static SplitResult showSplitDialog(
            Frame owner,
            int totalFrames,
            int width,
            int height,
            String method,
            int disDownsample,
            boolean float16,
            long availableBytes,
            int suggestedClips) {
        long estimated = VideoBatchPlan.estimateFlowBytes(
            Math.max(0, totalFrames - 1), width, height, method, disDownsample, float16);
        final int suggested = Math.max(1, VideoBatchPlan.clampClipCount(suggestedClips, totalFrames));

        final int[] chosen = {suggested};
        JDialog dialog = new JDialog(owner, "Split video into clips", true);
        dialog.setLayout(new BorderLayout(12, 12));

        JPanel top = new JPanel();
        top.setLayout(new BoxLayout(top, BoxLayout.Y_AXIS));
        top.setBorder(BorderFactory.createEmptyBorder(16, 18, 8, 18));
        JLabel explain = new JLabel("<html>" + VideoBatchPlan.suggestMessage(
            totalFrames, estimated, availableBytes, suggested) + "</html>");
        explain.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        top.add(explain);
        top.add(Box.createVerticalStrut(10));
        JLabel overBudget = new JLabel("Keeping the video as one clip may run out of RAM.");
        overBudget.setForeground(new Color(160, 80, 20));
        overBudget.setVisible(suggested > 1);
        top.add(overBudget);

        JLabel countLabel = new JLabel("Clips: " + suggested);
        countLabel.setFont(new Font("Segoe UI", Font.BOLD, 13));
        JSlider slider = new JSlider(1, Math.max(1, totalFrames - 1), suggested);
        slider.setMajorTickSpacing(Math.max(1, (totalFrames - 1) / 10));
        slider.setPaintTicks(true);

        String[] columns = {"Clip", "Frames", "Length"};
        DefaultTableModel model = new DefaultTableModel(columns, 0) {
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;
            }
        };
        JTable table = new JTable(model);
        table.setRowHeight(22);
        refreshRangeTable(model, totalFrames, suggested);

        slider.addChangeListener(e -> {
            chosen[0] = slider.getValue();
            countLabel.setText("Clips: " + chosen[0]);
            refreshRangeTable(model, totalFrames, chosen[0]);
            overBudget.setVisible(chosen[0] == 1 && suggested > 1);
        });

        JPanel middle = new JPanel(new BorderLayout(8, 8));
        middle.setBorder(BorderFactory.createEmptyBorder(0, 18, 0, 18));
        JPanel sliderRow = new JPanel(new BorderLayout(8, 0));
        sliderRow.add(countLabel, BorderLayout.WEST);
        sliderRow.add(slider, BorderLayout.CENTER);
        middle.add(sliderRow, BorderLayout.NORTH);
        JScrollPane tableScroll = new JScrollPane(table);
        tableScroll.setPreferredSize(new Dimension(420, 160));
        middle.add(tableScroll, BorderLayout.CENTER);

        JPanel buttons = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 10));
        JButton keep = new JButton("Keep as one clip");
        JButton split = new JButton("Split into clips");
        JButton cancel = new JButton("Cancel");
        final SplitResult[] result = {SplitResult.cancelled()};
        keep.addActionListener(e -> {
            result[0] = SplitResult.single();
            dialog.dispose();
        });
        split.addActionListener(e -> {
            result[0] = SplitResult.split(Math.max(1, chosen[0]));
            dialog.dispose();
        });
        cancel.addActionListener(e -> dialog.dispose());
        buttons.add(keep);
        buttons.add(split);
        buttons.add(cancel);

        dialog.add(top, BorderLayout.NORTH);
        dialog.add(middle, BorderLayout.CENTER);
        dialog.add(buttons, BorderLayout.SOUTH);
        dialog.pack();
        dialog.setLocationRelativeTo(owner);
        dialog.setVisible(true);
        return result[0];
    }

    public static int confirmReSplit(Frame owner) {
        return JOptionPane.showConfirmDialog(
            owner,
            "This video already has clip annotations.\n"
                + "Re-splitting will remap existing points into new ranges.\n"
                + "You will need to Switch into each clip to propagate again.\n\nContinue?",
            "Change clip size",
            JOptionPane.YES_NO_OPTION,
            JOptionPane.WARNING_MESSAGE);
    }

    public static JPanel createStrip(VideoBatchCoordinator coordinator, Listener listener) {
        JPanel strip = new JPanel(new FlowLayout(FlowLayout.LEFT, 6, 4));
        strip.setOpaque(true);
        strip.setBackground(new Color(32, 32, 36));
        strip.setBorder(BorderFactory.createEmptyBorder(2, 8, 2, 8));
        refreshStrip(strip, coordinator, listener);
        return strip;
    }

    public static void refreshStrip(JPanel strip, VideoBatchCoordinator coordinator, Listener listener) {
        if (strip == null) {
            return;
        }
        strip.removeAll();
        if (coordinator == null || !coordinator.isEnabled()) {
            strip.setVisible(false);
            strip.revalidate();
            strip.repaint();
            return;
        }
        strip.setVisible(true);
        strip.add(chip("Entire video", coordinator.isEntireVideo(), false, e -> listener.onViewEntireVideo()));
        List<VideoBatchPlan.ClipRange> ranges = coordinator.ranges();
        for (VideoBatchPlan.ClipRange range : ranges) {
            VideoBatchManifest.ClipEntry entry = coordinator.getManifest().entryAt(range.index);
            boolean working = coordinator.isWork() && coordinator.workingClipIndex() == range.index;
            boolean preview = coordinator.isPreview() && coordinator.previewClipIndex() == range.index;
            boolean annotated = entry != null && entry.hasAnnotations;
            String label = "Clip " + (range.index + 1);
            if (annotated) {
                label += " •";
            }
            boolean selected = working || preview;
            final int clipIndex = range.index;
            JButton button = chip(label, selected, preview && !working, e -> listener.onPreviewClip(clipIndex));
            button.setToolTipText("Frames " + range.displayRange1Based());
            strip.add(button);
        }
        strip.revalidate();
        strip.repaint();
    }

    public static JPanel createBanner(VideoBatchCoordinator coordinator, Listener listener) {
        JPanel banner = new JPanel(new FlowLayout(FlowLayout.CENTER, 16, 8));
        banner.setBackground(new Color(50, 55, 80));
        banner.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createMatteBorder(0, 0, 2, 0, new Color(120, 150, 210)),
            BorderFactory.createEmptyBorder(4, 12, 4, 12)));
        refreshBanner(banner, coordinator, listener);
        return banner;
    }

    public static void refreshBanner(JPanel banner, VideoBatchCoordinator coordinator, Listener listener) {
        if (banner == null) {
            return;
        }
        banner.removeAll();
        if (coordinator == null || !coordinator.isEnabled() || coordinator.isWork()) {
            banner.setVisible(false);
            banner.revalidate();
            banner.repaint();
            return;
        }
        banner.setVisible(true);
        JLabel label = new JLabel();
        label.setFont(new Font("Segoe UI", Font.BOLD, 13));
        label.setForeground(new Color(200, 220, 255));
        if (coordinator.isEntireVideo()) {
            label.setText("Viewing entire video — annotations are read-only. Select a clip to preview, then Switch to annotate.");
            banner.add(label);
        } else if (coordinator.isPreview()) {
            VideoBatchPlan.ClipRange range = coordinator.previewRange();
            String rangeText = range == null ? "" : " (frames " + range.displayRange1Based() + ")";
            label.setText("Previewing clip " + (coordinator.previewClipIndex() + 1) + rangeText
                + ". Optical flow is not loaded for this clip.");
            banner.add(label);
            int previewIndex = coordinator.previewClipIndex();
            banner.add(actionButton("Switch to this clip", e -> listener.onSwitchToClip(previewIndex)));
            if (coordinator.workingClipIndex() >= 0) {
                banner.add(actionButton("Back to clip " + (coordinator.workingClipIndex() + 1),
                    e -> listener.onBackToWorkingClip()));
            } else {
                banner.add(actionButton("Entire video", e -> listener.onViewEntireVideo()));
            }
        }
        banner.revalidate();
        banner.repaint();
    }

    private static JButton chip(String text, boolean selected, boolean preview, ActionListener listener) {
        JButton button = new JButton(text);
        button.setFocusPainted(false);
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        button.setFont(new Font("Segoe UI", Font.PLAIN, 11));
        button.setBorder(BorderFactory.createEmptyBorder(4, 10, 4, 10));
        if (selected && !preview) {
            button.setBackground(new Color(70, 120, 190));
            button.setForeground(Color.WHITE);
        } else if (preview) {
            button.setBackground(new Color(45, 55, 75));
            button.setForeground(new Color(180, 210, 255));
        } else {
            button.setBackground(new Color(48, 48, 52));
            button.setForeground(new Color(210, 210, 210));
        }
        button.addActionListener(listener);
        return button;
    }

    private static JButton actionButton(String text, ActionListener listener) {
        JButton button = new JButton(text);
        button.setFocusPainted(false);
        button.setBackground(new Color(70, 120, 190));
        button.setForeground(Color.WHITE);
        button.setBorder(BorderFactory.createEmptyBorder(6, 12, 6, 12));
        button.addActionListener(listener);
        return button;
    }

    private static void refreshRangeTable(DefaultTableModel model, int totalFrames, int clipCount) {
        model.setRowCount(0);
        List<VideoBatchPlan.ClipRange> ranges = VideoBatchPlan.split(totalFrames, clipCount);
        for (VideoBatchPlan.ClipRange range : ranges) {
            model.addRow(new Object[]{
                "Clip " + (range.index + 1),
                range.displayRange1Based(),
                range.frameCount() + " frames"
            });
        }
    }

    public static String frameStatus(int currentSlice, int totalSlices, VideoBatchCoordinator coordinator) {
        if (coordinator == null || !coordinator.isEnabled()) {
            return String.format("Frame: %d / %d", currentSlice, totalSlices);
        }
        VideoBatchPlan.ClipRange range = coordinator.navigationRange();
        if (range == null) {
            return String.format("Frame: %d / %d  (entire video)", currentSlice, totalSlices);
        }
        int local = currentSlice - range.startFrame;
        return String.format("Frame: %d / %d  (clip %d: %d/%d)",
            currentSlice, totalSlices, range.index + 1, local, range.frameCount());
    }
}
