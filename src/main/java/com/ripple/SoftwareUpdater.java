package com.ripple;

import javax.swing.BorderFactory;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import javax.swing.SwingWorker;
import javax.swing.WindowConstants;
import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Frame;
import java.awt.Window;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * Checks the official GitHub repository for software changes and applies
 * an in-place fast-forward update when the user confirms.
 */
public final class SoftwareUpdater {

    private static final int GIT_TIMEOUT_SECONDS = 90;
    private static final int LOG_LINE_LIMIT = 10;
    private static final String APPLY_SCRIPT_UNIX = "scripts/apply_update.sh";
    private static final String APPLY_SCRIPT_WINDOWS = "scripts/apply_update.bat";
    private static final String UPDATE_LOG_PATH = "tools/ripple-update.log";

    public enum Status {
        UP_TO_DATE,
        UPDATE_AVAILABLE,
        DIRTY,
        NOT_A_CLONE,
        NOT_FAST_FORWARD,
        ERROR
    }

    public static final class CheckResult {
        public final Status status;
        public final String message;
        public final int behindCount;
        public final String currentSha;
        public final String recentLog;

        CheckResult(Status status, String message, int behindCount, String currentSha, String recentLog) {
            this.status = status;
            this.message = message;
            this.behindCount = behindCount;
            this.currentSha = currentSha == null ? "" : currentSha;
            this.recentLog = recentLog == null ? "" : recentLog;
        }

        static CheckResult upToDate(String sha) {
            String suffix = (sha == null || sha.isBlank()) ? "" : " (commit " + sha + ")";
            return new CheckResult(
                Status.UP_TO_DATE,
                "You are already on the latest version from GitHub" + suffix + ".",
                0, sha, "");
        }

        static CheckResult available(int behindCount, String sha, String recentLog) {
            return new CheckResult(
                Status.UPDATE_AVAILABLE,
                formatUpdateMessage(behindCount, parseLogLines(recentLog, LOG_LINE_LIMIT)),
                behindCount, sha, recentLog);
        }

        static CheckResult dirty(String changedFiles) {
            String details = (changedFiles == null || changedFiles.isBlank())
                ? ""
                : "\n\nChanged files:\n" + changedFiles;
            return new CheckResult(
                Status.DIRTY,
                "RIPPLE has local source changes, so the update was cancelled "
                    + "to avoid overwriting them." + details,
                0, "", "");
        }

        static CheckResult notAClone() {
            return new CheckResult(
                Status.NOT_A_CLONE,
                "This RIPPLE install is not a git clone, so it cannot be updated in place.\n\n"
                    + "Install from GitHub instead:\n\n"
                    + "  git clone " + Constants.GITHUB_REPO_URL + "\n"
                    + "  cd ripple\n"
                    + "  bash quickstart.sh",
                0, "", "");
        }

        static CheckResult notFastForward() {
            return new CheckResult(
                Status.NOT_FAST_FORWARD,
                "Your local repository has diverged from GitHub, so a safe "
                    + "fast-forward update is not possible.\n\n"
                    + "Re-clone the repository, or resolve the git history manually.",
                0, "", "");
        }

        static CheckResult error(String message) {
            return new CheckResult(
                Status.ERROR,
                message == null || message.isBlank()
                    ? "Could not check for updates."
                    : message,
                0, "", "");
        }
    }

    private SoftwareUpdater() {
    }

    /**
     * Check GitHub for updates and prompt the user. If they confirm, run
     * {@code prepareShutdown}, spawn the apply script, and exit the JVM.
     *
     * @param parent          dialog parent
     * @param prepareShutdown cleanup to run after the user confirms apply
     * @param onFinished      called if the user does not apply (cancel / up to date / error)
     */
    public static void checkForUpdates(Component parent, Runnable prepareShutdown, Runnable onFinished) {
        Window owner = parent == null ? null : SwingUtilities.getWindowAncestor(parent);
        if (owner == null && parent instanceof Window) {
            owner = (Window) parent;
        }
        JDialog progress = createProgressDialog(owner);

        SwingWorker<CheckResult, Void> worker = new SwingWorker<>() {
            @Override
            protected CheckResult doInBackground() {
                return performCheck(resolveRepoRoot());
            }

            @Override
            protected void done() {
                progress.dispose();
                CheckResult result;
                try {
                    result = get();
                } catch (Exception e) {
                    result = CheckResult.error(rootCauseMessage(e));
                }
                handleResult(parent, result, prepareShutdown, onFinished);
            }
        };
        worker.execute();
        progress.setLocationRelativeTo(parent);
        progress.setVisible(true);
    }

    static CheckResult performCheck(File repoRoot) {
        return performCheck(repoRoot, Constants.GITHUB_REPO_URL, Constants.GITHUB_DEFAULT_BRANCH);
    }

    static CheckResult performCheck(File repoRoot, String repoUrl, String branch) {
        if (repoRoot == null || !repoRoot.isDirectory()) {
            return CheckResult.error("Could not find the RIPPLE install directory.");
        }
        if (!isGitOnPath()) {
            return CheckResult.error(
                "Git was not found on your PATH. Install Git and open RIPPLE from a clone of "
                    + Constants.GITHUB_REPO_URL + ".");
        }
        if (!isGitCheckout(repoRoot)) {
            return CheckResult.notAClone();
        }

        String remote = (repoUrl == null || repoUrl.isBlank())
            ? Constants.GITHUB_REPO_URL : repoUrl;
        String ref = (branch == null || branch.isBlank())
            ? Constants.GITHUB_DEFAULT_BRANCH : branch;

        try {
            CommandResult status = runGit(repoRoot, 15,
                "status", "--porcelain", "--untracked-files=no");
            if (status.exitCode != 0) {
                return CheckResult.error(firstNonBlank(status.output,
                    "Could not read the local git status."));
            }
            if (isDirtyWorkingTree(status.output)) {
                return CheckResult.dirty(summarizeDirtyFiles(status.output));
            }

            CommandResult fetch = runGit(repoRoot, GIT_TIMEOUT_SECONDS,
                "fetch", remote, ref);
            if (fetch.exitCode != 0) {
                return CheckResult.error(
                    "Could not reach GitHub. Check your network connection.\n\n"
                        + firstNonBlank(fetch.output, "git fetch failed."));
            }

            String currentSha = shortSha(runGit(repoRoot, 15, "rev-parse", "--short", "HEAD").output);
            CommandResult behindResult = runGit(repoRoot, 15,
                "rev-list", "--count", "HEAD..FETCH_HEAD");
            CommandResult logResult = runGit(repoRoot, 15,
                "log", "--oneline", "-" + LOG_LINE_LIMIT, "HEAD..FETCH_HEAD");
            boolean ancestor = isAncestor(repoRoot);

            return fromGitState(status.output, behindResult.output, logResult.output,
                ancestor, currentSha);
        } catch (IOException e) {
            return CheckResult.error(e.getMessage());
        }
    }

    /**
     * Build a check result from already-captured git output. Used by unit tests
     * so they do not need a live GitHub connection.
     */
    static CheckResult fromGitState(String porcelain, String behindCountText, String logText,
                                    boolean headIsAncestorOfRemote, String currentSha) {
        if (isDirtyWorkingTree(porcelain)) {
            return CheckResult.dirty(summarizeDirtyFiles(porcelain));
        }
        int behind = parseBehindCount(behindCountText);
        if (behind < 0) {
            return CheckResult.error("Could not determine how far this install is behind GitHub.");
        }
        if (behind == 0) {
            return CheckResult.upToDate(currentSha);
        }
        if (!headIsAncestorOfRemote) {
            return CheckResult.notFastForward();
        }
        return CheckResult.available(behind, currentSha, logText);
    }

    /**
     * True when porcelain status contains tracked-file changes.
     * Untracked ({@code ??}) and ignored ({@code !!}) entries do not count, so
     * user videos and annotation files in the repo folder do not block updates.
     */
    static boolean isDirtyWorkingTree(String porcelain) {
        if (porcelain == null || porcelain.isBlank()) {
            return false;
        }
        for (String line : porcelain.split("\\R")) {
            if (line.isBlank()) {
                continue;
            }
            if (line.startsWith("??") || line.startsWith("!!")) {
                continue;
            }
            return true;
        }
        return false;
    }

    static int parseBehindCount(String text) {
        if (text == null) {
            return -1;
        }
        String first = text.trim();
        if (first.isEmpty()) {
            return -1;
        }
        int newline = indexOfNewline(first);
        if (newline >= 0) {
            first = first.substring(0, newline).trim();
        }
        try {
            int count = Integer.parseInt(first);
            return count < 0 ? -1 : count;
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    static List<String> parseLogLines(String logText, int maxLines) {
        List<String> lines = new ArrayList<>();
        if (logText == null || logText.isBlank() || maxLines <= 0) {
            return lines;
        }
        for (String line : logText.split("\\R")) {
            if (line.isBlank()) {
                continue;
            }
            lines.add(line);
            if (lines.size() >= maxLines) {
                break;
            }
        }
        return lines;
    }

    static String formatUpdateMessage(int behindCount, List<String> logLines) {
        StringBuilder sb = new StringBuilder();
        sb.append("A newer version is available on GitHub.\n\n");
        sb.append(behindCount).append(" new commit");
        if (behindCount != 1) {
            sb.append('s');
        }
        sb.append(":\n");
        if (logLines != null) {
            for (String line : logLines) {
                sb.append("  ").append(line).append('\n');
            }
        }
        return sb.toString().stripTrailing();
    }

    static String summarizeDirtyFiles(String porcelain) {
        List<String> files = new ArrayList<>();
        if (porcelain == null) {
            return "";
        }
        for (String line : porcelain.split("\\R")) {
            if (line.isBlank() || line.startsWith("??") || line.startsWith("!!")) {
                continue;
            }
            String path = line.length() >= 3 ? line.substring(3).trim() : line.trim();
            if (!path.isEmpty()) {
                files.add(path);
            }
            if (files.size() >= 8) {
                files.add("...");
                break;
            }
        }
        return String.join("\n", files);
    }

    static boolean isWindows() {
        String os = System.getProperty("os.name", "");
        return os.toLowerCase(Locale.ROOT).startsWith("win");
    }

    static File resolveRepoRoot() {
        return new File(System.getProperty("user.dir")).getAbsoluteFile();
    }

    static String resolveInstallMode() {
        String env = System.getenv(Constants.ENV_RIPPLE_MODE);
        if (env != null && env.equalsIgnoreCase(Constants.MODE_GPU)) {
            return Constants.MODE_GPU;
        }
        return Constants.MODE_CPU;
    }

    static void spawnApplyScript(File repoRoot) throws IOException {
        boolean windows = isWindows();
        File script = new File(repoRoot, windows ? APPLY_SCRIPT_WINDOWS : APPLY_SCRIPT_UNIX);
        if (!script.isFile()) {
            throw new IOException("Update script not found: " + script.getAbsolutePath());
        }
        long pid = ProcessHandle.current().pid();
        String mode = resolveInstallMode();
        File logFile = new File(repoRoot, UPDATE_LOG_PATH);
        File logDir = logFile.getParentFile();
        if (logDir != null) {
            logDir.mkdirs();
        }

        ProcessBuilder pb;
        if (windows) {
            pb = new ProcessBuilder(
                "cmd.exe", "/c", "start", "RIPPLE Update",
                script.getAbsolutePath(), Long.toString(pid), mode);
        } else {
            pb = new ProcessBuilder(
                "bash", "-c",
                "nohup bash \"" + script.getAbsolutePath() + "\" "
                    + pid + " " + mode
                    + " >> \"" + logFile.getAbsolutePath() + "\" 2>&1 < /dev/null &");
        }
        pb.directory(repoRoot);
        pb.start();
        try {
            Thread.sleep(400);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static void handleResult(Component parent, CheckResult result,
                                     Runnable prepareShutdown, Runnable onFinished) {
        if (result == null) {
            finish(onFinished);
            return;
        }
        switch (result.status) {
            case UP_TO_DATE -> {
                JOptionPane.showMessageDialog(parent, result.message,
                    "RIPPLE Update", JOptionPane.INFORMATION_MESSAGE);
                finish(onFinished);
            }
            case UPDATE_AVAILABLE -> promptApply(parent, result, prepareShutdown, onFinished);
            case DIRTY, NOT_A_CLONE, NOT_FAST_FORWARD -> {
                JOptionPane.showMessageDialog(parent, result.message,
                    "RIPPLE Update", JOptionPane.WARNING_MESSAGE);
                finish(onFinished);
            }
            case ERROR -> {
                JOptionPane.showMessageDialog(parent, result.message,
                    "RIPPLE Update", JOptionPane.ERROR_MESSAGE);
                finish(onFinished);
            }
        }
    }

    private static void promptApply(Component parent, CheckResult result,
                                    Runnable prepareShutdown, Runnable onFinished) {
        String confirmText = result.message
            + "\n\nUpdate now? RIPPLE will close, apply the update, and reopen.";
        int choice = JOptionPane.showConfirmDialog(
            parent, wrapMessage(confirmText), "Update Available",
            JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE);
        if (choice != JOptionPane.YES_OPTION) {
            finish(onFinished);
            return;
        }

        int warn = JOptionPane.showConfirmDialog(
            parent,
            "RIPPLE will close to apply the update.\n\nAny unsaved annotations will be lost.",
            "Confirm Update",
            JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);
        if (warn != JOptionPane.YES_OPTION) {
            finish(onFinished);
            return;
        }

        try {
            if (prepareShutdown != null) {
                prepareShutdown.run();
            }
            spawnApplyScript(resolveRepoRoot());
        } catch (Exception e) {
            JOptionPane.showMessageDialog(parent,
                "Could not start the updater:\n" + rootCauseMessage(e),
                "RIPPLE Update", JOptionPane.ERROR_MESSAGE);
            finish(onFinished);
            return;
        }
        System.exit(0);
    }

    private static void finish(Runnable onFinished) {
        if (onFinished != null) {
            onFinished.run();
        }
    }

    private static JDialog createProgressDialog(Window owner) {
        Frame frameOwner = owner instanceof Frame ? (Frame) owner : null;
        JDialog dialog = new JDialog(frameOwner, "RIPPLE Update", true);
        dialog.setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);

        JPanel panel = new JPanel(new BorderLayout(10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(16, 20, 16, 20));
        panel.add(new JLabel("Checking GitHub for updates..."), BorderLayout.NORTH);
        JProgressBar bar = new JProgressBar();
        bar.setIndeterminate(true);
        panel.add(bar, BorderLayout.CENTER);

        dialog.setContentPane(panel);
        dialog.setSize(360, 120);
        dialog.setResizable(false);
        return dialog;
    }

    private static JScrollPane wrapMessage(String text) {
        JTextArea area = new JTextArea(text);
        area.setEditable(false);
        area.setLineWrap(true);
        area.setWrapStyleWord(true);
        area.setOpaque(false);
        area.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));
        JScrollPane scroll = new JScrollPane(area);
        scroll.setPreferredSize(new Dimension(480, 260));
        scroll.setBorder(null);
        return scroll;
    }

    private static boolean isGitOnPath() {
        try {
            CommandResult result = runCommand(null, 10, "git", "--version");
            return result.exitCode == 0;
        } catch (IOException e) {
            return false;
        }
    }

    private static boolean isGitCheckout(File repoRoot) {
        File gitDir = new File(repoRoot, ".git");
        if (gitDir.exists()) {
            return true;
        }
        try {
            CommandResult result = runGit(repoRoot, 10, "rev-parse", "--is-inside-work-tree");
            return result.exitCode == 0 && result.output.trim().equalsIgnoreCase("true");
        } catch (IOException e) {
            return false;
        }
    }

    private static boolean isAncestor(File repoRoot) {
        try {
            CommandResult result = runGit(repoRoot, 15,
                "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD");
            return result.exitCode == 0;
        } catch (IOException e) {
            return false;
        }
    }

    private static String shortSha(String output) {
        if (output == null) {
            return "";
        }
        String sha = output.trim();
        int newline = indexOfNewline(sha);
        if (newline >= 0) {
            sha = sha.substring(0, newline).trim();
        }
        return sha;
    }

    private static CommandResult runGit(File repoRoot, int timeoutSec, String... args)
            throws IOException {
        String[] command = new String[args.length + 1];
        command[0] = "git";
        System.arraycopy(args, 0, command, 1, args.length);
        return runCommand(repoRoot, timeoutSec, command);
    }

    private static CommandResult runCommand(File workDir, int timeoutSec, String... command)
            throws IOException {
        ProcessBuilder pb = new ProcessBuilder(command);
        if (workDir != null) {
            pb.directory(workDir);
        }
        pb.redirectErrorStream(true);
        Process process;
        try {
            process = pb.start();
        } catch (IOException e) {
            throw new IOException("Could not run " + command[0] + ": " + e.getMessage(), e);
        }

        String output;
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            output = reader.lines().collect(Collectors.joining("\n"));
        }

        boolean finished;
        try {
            finished = process.waitFor(timeoutSec, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            process.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while running " + command[0], e);
        }
        if (!finished) {
            process.destroyForcibly();
            throw new IOException("Timed out running: " + String.join(" ", command));
        }
        return new CommandResult(process.exitValue(), output);
    }

    private static String firstNonBlank(String text, String fallback) {
        if (text == null || text.isBlank()) {
            return fallback;
        }
        return text.trim();
    }

    private static String rootCauseMessage(Throwable e) {
        Throwable cursor = e;
        while (cursor.getCause() != null && cursor.getCause() != cursor) {
            cursor = cursor.getCause();
        }
        String message = cursor.getMessage();
        return message == null || message.isBlank() ? e.toString() : message;
    }

    private static int indexOfNewline(String text) {
        int n = text.indexOf('\n');
        int r = text.indexOf('\r');
        if (n < 0) {
            return r;
        }
        if (r < 0) {
            return n;
        }
        return Math.min(n, r);
    }

    private static final class CommandResult {
        final int exitCode;
        final String output;

        CommandResult(int exitCode, String output) {
            this.exitCode = exitCode;
            this.output = output == null ? "" : output;
        }
    }
}
