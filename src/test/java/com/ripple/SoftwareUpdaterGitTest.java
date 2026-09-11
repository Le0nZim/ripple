package com.ripple;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end git fixtures for the Update check and apply_update.sh.
 * Uses local bare remotes only — never the live GitHub repository.
 */
class SoftwareUpdaterGitTest {

    @TempDir
    Path temp;

    @Test
    void performCheckReportsDirtyTrackedFileAndIgnoresUntrackedVideo() throws Exception {
        RepoPair pair = createPair("dirty");
        Files.writeString(pair.clone.resolve("README.md"), "local edit\n");
        Files.writeString(pair.clone.resolve("experiment.tif"), "video-bytes");

        SoftwareUpdater.CheckResult result = SoftwareUpdater.performCheck(
            pair.clone.toFile(), pair.remote.toString(), "main");

        assertEquals(SoftwareUpdater.Status.DIRTY, result.status, result.message);
        assertTrue(result.message.contains("README.md"), result.message);
        assertFalse(result.message.contains("experiment.tif"), result.message);
    }

    @Test
    void performCheckReportsUpToDateWhenCloneMatchesRemote() throws Exception {
        RepoPair pair = createPair("current");

        SoftwareUpdater.CheckResult result = SoftwareUpdater.performCheck(
            pair.clone.toFile(), pair.remote.toString(), "main");

        assertEquals(SoftwareUpdater.Status.UP_TO_DATE, result.status, result.message);
        assertEquals(0, result.behindCount);
        assertFalse(result.currentSha.isBlank());
    }

    @Test
    void performCheckReportsAvailableWhenRemoteIsAhead() throws Exception {
        RepoPair pair = createPair("behind");
        commitOnRemote(pair, "second.txt", "new file from upstream\n", "Add second file");

        SoftwareUpdater.CheckResult result = SoftwareUpdater.performCheck(
            pair.clone.toFile(), pair.remote.toString(), "main");

        assertEquals(SoftwareUpdater.Status.UPDATE_AVAILABLE, result.status, result.message);
        assertEquals(1, result.behindCount);
        assertTrue(result.message.contains("1 new commit"), result.message);
        assertTrue(result.message.toLowerCase().contains("second")
            || result.recentLog.toLowerCase().contains("second"), result.message);
    }

    @Test
    void performCheckReportsManyRemoteCommits() throws Exception {
        RepoPair pair = createPair("many");
        for (int i = 1; i <= 12; i++) {
            commitOnRemote(pair, "f" + i + ".txt", "content " + i + "\n", "Commit number " + i);
        }

        SoftwareUpdater.CheckResult result = SoftwareUpdater.performCheck(
            pair.clone.toFile(), pair.remote.toString(), "main");

        assertEquals(SoftwareUpdater.Status.UPDATE_AVAILABLE, result.status, result.message);
        assertEquals(12, result.behindCount);
        assertTrue(result.message.contains("12 new commits"), result.message);
        assertTrue(result.message.contains("Commit number 12")
            || result.recentLog.contains("Commit number 12"), result.message);
        assertFalse(result.message.matches("(?s).*Commit number 1([^0-9].*)?"),
            "oldest of 12 commits should be truncated from the 10-line preview: " + result.message);
    }

    @Test
    void performCheckReportsDivergedHistory() throws Exception {
        RepoPair pair = createPair("diverged");
        commitOnRemote(pair, "upstream-only.txt", "from remote\n", "Remote commit");
        Files.writeString(pair.clone.resolve("local-only.txt"), "from clone\n");
        git(pair.clone, "add", "local-only.txt");
        git(pair.clone, "-c", "user.email=test@ripple.local", "-c", "user.name=Test",
            "commit", "-m", "Local commit");

        SoftwareUpdater.CheckResult result = SoftwareUpdater.performCheck(
            pair.clone.toFile(), pair.remote.toString(), "main");

        assertEquals(SoftwareUpdater.Status.NOT_FAST_FORWARD, result.status, result.message);
    }

    @Test
    void performCheckFailsFetchWhenRemoteMissing() throws Exception {
        RepoPair pair = createPair("bad-remote");
        Path missing = temp.resolve("no-such-remote.git");

        SoftwareUpdater.CheckResult result = SoftwareUpdater.performCheck(
            pair.clone.toFile(), missing.toString(), "main");

        assertEquals(SoftwareUpdater.Status.ERROR, result.status, result.message);
        assertTrue(result.message.toLowerCase().contains("github")
            || result.message.toLowerCase().contains("fetch")
            || result.message.toLowerCase().contains("could not"), result.message);
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void applyUpdateFastForwardsAndLeavesUntrackedAlone() throws Exception {
        RepoPair pair = createInstall("apply-ff");
        commitOnRemote(pair, "feature.txt", "ship it\n", "Ship feature");
        Files.writeString(pair.clone.resolve("user-video.tif"), "keep-me");

        CommandResult ran = runApply(pair, "", "cpu");
        assertEquals(0, ran.exitCode, ran.output);
        assertTrue(Files.exists(pair.clone.resolve("feature.txt")), ran.output);
        assertEquals("keep-me", Files.readString(pair.clone.resolve("user-video.tif")));
        assertEquals("ship it\n", Files.readString(pair.clone.resolve("feature.txt")));
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void applyUpdateAbortsWhenTrackedFileIsDirty() throws Exception {
        RepoPair pair = createInstall("apply-dirty");
        commitOnRemote(pair, "feature.txt", "ship it\n", "Ship feature");
        Files.writeString(pair.clone.resolve("README.md"), "do not overwrite\n");

        CommandResult ran = runApply(pair, "", "cpu");
        assertNotEquals(0, ran.exitCode, ran.output);
        assertTrue(ran.output.contains("local source changes"), ran.output);
        assertFalse(Files.exists(pair.clone.resolve("feature.txt")));
        assertEquals("do not overwrite\n", Files.readString(pair.clone.resolve("README.md")));
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void applyUpdateAbortsWhenHistoriesDiverge() throws Exception {
        RepoPair pair = createInstall("apply-diverge");
        commitOnRemote(pair, "upstream-only.txt", "from remote\n", "Remote commit");
        Files.writeString(pair.clone.resolve("local-only.txt"), "from clone\n");
        git(pair.clone, "add", "local-only.txt");
        git(pair.clone, "-c", "user.email=test@ripple.local", "-c", "user.name=Test",
            "commit", "-m", "Local commit");

        CommandResult ran = runApply(pair, "", "cpu");
        assertNotEquals(0, ran.exitCode, ran.output);
        assertTrue(ran.output.toLowerCase().contains("fast-forward")
            || ran.output.toLowerCase().contains("not possible")
            || ran.output.toLowerCase().contains("refusing"), ran.output);
        assertTrue(Files.exists(pair.clone.resolve("local-only.txt")));
        assertFalse(Files.exists(pair.clone.resolve("upstream-only.txt")));
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void applyUpdateIsNoOpWhenAlreadyCurrent() throws Exception {
        RepoPair pair = createInstall("apply-current");
        String before = git(pair.clone, "rev-parse", "HEAD").output.trim();

        CommandResult ran = runApply(pair, "", "gpu");
        assertEquals(0, ran.exitCode, ran.output);
        String after = git(pair.clone, "rev-parse", "HEAD").output.trim();
        assertEquals(before, after);
        assertTrue(ran.output.contains("Mode: gpu") || ran.output.contains("gpu"), ran.output);
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void applyUpdateRejectsNonGitInstall() throws Exception {
        Path install = temp.resolve("zip-install");
        Files.createDirectories(install.resolve("scripts"));
        Path script = Path.of("scripts/apply_update.sh").toAbsolutePath();
        if (!Files.exists(script)) {
            script = Path.of("ripple_fixes/scripts/apply_update.sh").toAbsolutePath();
        }
        Files.copy(script, install.resolve("scripts/apply_update.sh"), StandardCopyOption.REPLACE_EXISTING);

        CommandResult ran = runCommand(install, 20,
            "bash", install.resolve("scripts/apply_update.sh").toString(), "", "cpu");
        assertNotEquals(0, ran.exitCode, ran.output);
        assertTrue(ran.output.toLowerCase().contains("not a git"), ran.output);
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void applyUpdateWaitsForDeadPidThenProceeds() throws Exception {
        RepoPair pair = createInstall("apply-pid");
        Process sleeper = new ProcessBuilder("sleep", "30").start();
        long pid = sleeper.pid();
        sleeper.destroyForcibly();
        sleeper.waitFor(5, TimeUnit.SECONDS);

        CommandResult ran = runApply(pair, Long.toString(pid), "cpu");
        assertEquals(0, ran.exitCode, ran.output);
        assertTrue(ran.output.contains("RIPPLE exited") || ran.output.contains("update"), ran.output);
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void applyUpdateTimesOutIfProcessNeverExits() throws Exception {
        RepoPair pair = createInstall("apply-timeout");
        Process sleeper = new ProcessBuilder("sleep", "30").start();
        try {
            CommandResult ran = runApply(pair, Long.toString(sleeper.pid()), "cpu",
                "RIPPLE_UPDATE_WAIT_SECONDS=1");
            assertNotEquals(0, ran.exitCode, ran.output);
            assertTrue(ran.output.toLowerCase().contains("timed out"), ran.output);
        } finally {
            sleeper.destroyForcibly();
        }
    }

    @Test
    @EnabledOnOs({OS.LINUX, OS.MAC})
    void applyUpdateTreatsUnknownModeAsCpu() throws Exception {
        RepoPair pair = createInstall("apply-mode");
        CommandResult ran = runApply(pair, "", "not-a-mode");
        assertEquals(0, ran.exitCode, ran.output);
        assertTrue(ran.output.contains("Mode: cpu"), ran.output);
    }

    private RepoPair createPair(String name) throws Exception {
        Path remote = temp.resolve(name + "-remote.git");
        Path clone = temp.resolve(name + "-clone");
        git(temp, "init", "--bare", "--initial-branch=main", remote.toString());

        Path seed = temp.resolve(name + "-seed");
        git(temp, "init", "--initial-branch=main", seed.toString());
        Files.writeString(seed.resolve("README.md"), "seed\n");
        git(seed, "add", "README.md");
        git(seed, "-c", "user.email=test@ripple.local", "-c", "user.name=Test",
            "commit", "-m", "Initial commit");
        git(seed, "remote", "add", "origin", remote.toString());
        git(seed, "push", "-u", "origin", "main");

        git(temp, "clone", remote.toString(), clone.toString());
        return new RepoPair(remote, clone);
    }

    private RepoPair createInstall(String name) throws Exception {
        RepoPair pair = createPair(name);
        Path scripts = pair.clone.resolve("scripts");
        Files.createDirectories(scripts);
        Path source = locateApplyScript();
        Files.copy(source, scripts.resolve("apply_update.sh"), StandardCopyOption.REPLACE_EXISTING);
        return pair;
    }

    private Path locateApplyScript() {
        Path fromModule = Path.of("scripts/apply_update.sh").toAbsolutePath();
        if (Files.exists(fromModule)) {
            return fromModule;
        }
        Path fromWorkspace = Path.of("ripple_fixes/scripts/apply_update.sh").toAbsolutePath();
        if (Files.exists(fromWorkspace)) {
            return fromWorkspace;
        }
        throw new IllegalStateException("Cannot find apply_update.sh");
    }

    private void commitOnRemote(RepoPair pair, String filename, String content, String message)
            throws Exception {
        Path work = temp.resolve(pair.clone.getFileName() + "-upstream-work");
        if (Files.exists(work)) {
            deleteRecursive(work);
        }
        git(temp, "clone", pair.remote.toString(), work.toString());
        Files.writeString(work.resolve(filename), content);
        git(work, "add", filename);
        git(work, "-c", "user.email=test@ripple.local", "-c", "user.name=Test",
            "commit", "-m", message);
        git(work, "push", "origin", "HEAD:main");
    }

    private CommandResult runApply(RepoPair pair, String pid, String mode, String... extraEnv)
            throws Exception {
        List<String> command = List.of(
            "bash", pair.clone.resolve("scripts/apply_update.sh").toString(), pid, mode);
        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(pair.clone.toFile());
        pb.redirectErrorStream(true);
        pb.environment().put("RIPPLE_UPDATE_REMOTE", pair.remote.toString());
        pb.environment().put("RIPPLE_UPDATE_BRANCH", "main");
        pb.environment().put("RIPPLE_UPDATE_SKIP_QUICKSTART", "1");
        pb.environment().put("RIPPLE_UPDATE_SKIP_RELAUNCH", "1");
        pb.environment().put("RIPPLE_UPDATE_WAIT_SECONDS", "8");
        for (String entry : extraEnv) {
            int eq = entry.indexOf('=');
            pb.environment().put(entry.substring(0, eq), entry.substring(eq + 1));
        }
        return waitFor(pb, 30);
    }

    private static CommandResult git(Path workDir, String... args) throws Exception {
        List<String> command = new ArrayList<>();
        command.add("git");
        command.addAll(List.of(args));
        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(workDir.toFile());
        pb.redirectErrorStream(true);
        pb.environment().put("GIT_AUTHOR_NAME", "Test");
        pb.environment().put("GIT_AUTHOR_EMAIL", "test@ripple.local");
        pb.environment().put("GIT_COMMITTER_NAME", "Test");
        pb.environment().put("GIT_COMMITTER_EMAIL", "test@ripple.local");
        CommandResult result = waitFor(pb, 20);
        if (result.exitCode != 0) {
            throw new IllegalStateException("git " + String.join(" ", args)
                + " failed (" + result.exitCode + "): " + result.output);
        }
        return result;
    }

    private static CommandResult runCommand(Path workDir, int timeoutSec, String... command)
            throws Exception {
        ProcessBuilder pb = new ProcessBuilder(command);
        pb.directory(workDir.toFile());
        pb.redirectErrorStream(true);
        return waitFor(pb, timeoutSec);
    }

    private static CommandResult waitFor(ProcessBuilder pb, int timeoutSec) throws Exception {
        Process process = pb.start();
        String output;
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            output = reader.lines().collect(Collectors.joining("\n"));
        }
        if (!process.waitFor(timeoutSec, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            throw new IllegalStateException("Timed out: " + pb.command() + "\n" + output);
        }
        return new CommandResult(process.exitValue(), output);
    }

    private static void deleteRecursive(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }
        Files.walk(root)
            .sorted((a, b) -> b.compareTo(a))
            .forEach(path -> {
                try {
                    Files.deleteIfExists(path);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
    }

    private static final class RepoPair {
        final Path remote;
        final Path clone;

        RepoPair(Path remote, Path clone) {
            this.remote = remote;
            this.clone = clone;
        }
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
