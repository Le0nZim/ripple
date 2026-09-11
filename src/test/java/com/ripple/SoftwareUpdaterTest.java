package com.ripple;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SoftwareUpdaterTest {

    @Test
    void dirtyTreeIgnoresUntrackedAndIgnoredFiles() {
        assertFalse(SoftwareUpdater.isDirtyWorkingTree(""));
        assertFalse(SoftwareUpdater.isDirtyWorkingTree("   \n"));
        assertFalse(SoftwareUpdater.isDirtyWorkingTree(null));
        assertFalse(SoftwareUpdater.isDirtyWorkingTree("?? video.tif\n?? notes.json"));
        assertFalse(SoftwareUpdater.isDirtyWorkingTree("!! target/ripple.jar"));
        assertFalse(SoftwareUpdater.isDirtyWorkingTree("?? video.tif\r\n!! tools/jdk"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree(" M src/main/java/com/ripple/VideoAnnotationTool.java"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree("M  pom.xml"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree("?? video.tif\n M pom.xml"));
    }

    @Test
    void dirtyTreeTreatsAllTrackedStatusCodesAsBlocking() {
        assertTrue(SoftwareUpdater.isDirtyWorkingTree("A  newfile.java"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree(" D deleted.java"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree("D  staged-delete.java"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree("R  old.java -> new.java"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree("C  copy.java -> copy2.java"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree("MM conflicted-looking.java"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree("UU unmerged.java"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree("AA both-added.java"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree("DU deleted-unmerged.java"));
        assertTrue(SoftwareUpdater.isDirtyWorkingTree(" T file-type-change"));
    }

    @Test
    void parseBehindCountReadsFirstLineAndRejectsJunk() {
        assertEquals(0, SoftwareUpdater.parseBehindCount("0"));
        assertEquals(5, SoftwareUpdater.parseBehindCount("5\n"));
        assertEquals(12, SoftwareUpdater.parseBehindCount("  12\r\nextra"));
        assertEquals(0, SoftwareUpdater.parseBehindCount("0\r\nwarning: extra"));
        assertEquals(2147483647, SoftwareUpdater.parseBehindCount("2147483647"));
        assertEquals(-1, SoftwareUpdater.parseBehindCount(""));
        assertEquals(-1, SoftwareUpdater.parseBehindCount("   "));
        assertEquals(-1, SoftwareUpdater.parseBehindCount(null));
        assertEquals(-1, SoftwareUpdater.parseBehindCount("not-a-number"));
        assertEquals(-1, SoftwareUpdater.parseBehindCount("-3"));
        assertEquals(-1, SoftwareUpdater.parseBehindCount("3 commits"));
        assertEquals(-1, SoftwareUpdater.parseBehindCount("1.5"));
        assertEquals(-1, SoftwareUpdater.parseBehindCount("9999999999999999999"));
    }

    @Test
    void parseLogLinesCapsSkipsBlanksAndKeepsUnicode() {
        String log = "abc123 Fix toolbar\n\ndef456 Update README\n789xyz More\nlast";
        List<String> lines = SoftwareUpdater.parseLogLines(log, 2);
        assertEquals(List.of("abc123 Fix toolbar", "def456 Update README"), lines);

        String crlf = "one\r\n\r\ntwo\r\nthree";
        assertEquals(List.of("one", "two"), SoftwareUpdater.parseLogLines(crlf, 2));

        assertEquals(List.of("αβγ fix"), SoftwareUpdater.parseLogLines("αβγ fix\n", 5));
        assertTrue(SoftwareUpdater.parseLogLines("", 10).isEmpty());
        assertTrue(SoftwareUpdater.parseLogLines("   \n\n", 10).isEmpty());
        assertTrue(SoftwareUpdater.parseLogLines(null, 10).isEmpty());
        assertTrue(SoftwareUpdater.parseLogLines("abc", 0).isEmpty());
        assertTrue(SoftwareUpdater.parseLogLines("abc", -1).isEmpty());
    }

    @Test
    void parseLogLinesStopsAtTenEvenWhenMoreExist() {
        StringBuilder log = new StringBuilder();
        for (int i = 1; i <= 25; i++) {
            log.append(i).append(" commit ").append(i).append('\n');
        }
        List<String> lines = SoftwareUpdater.parseLogLines(log.toString(), 10);
        assertEquals(10, lines.size());
        assertEquals("1 commit 1", lines.get(0));
        assertEquals("10 commit 10", lines.get(9));
    }

    @Test
    void formatUpdateMessageIncludesCountAndSubjects() {
        String message = SoftwareUpdater.formatUpdateMessage(
            2, List.of("abc Fix button", "def Docs"));
        assertTrue(message.contains("2 new commits"));
        assertTrue(message.contains("abc Fix button"));
        assertTrue(SoftwareUpdater.formatUpdateMessage(1, List.of("only")).contains("1 new commit:"));
        assertTrue(SoftwareUpdater.formatUpdateMessage(0, List.of()).contains("0 new commits"));
        assertDoesNotThrow(() -> SoftwareUpdater.formatUpdateMessage(3, null));
    }

    @Test
    void fromGitStateMapsDirtyBehindAndDiverged() {
        SoftwareUpdater.CheckResult dirty = SoftwareUpdater.fromGitState(
            " M pom.xml", "4", "abc Change", true, "deadbee");
        assertEquals(SoftwareUpdater.Status.DIRTY, dirty.status);
        assertTrue(dirty.message.contains("pom.xml"));

        SoftwareUpdater.CheckResult current = SoftwareUpdater.fromGitState(
            "", "0", "", true, "abc1234");
        assertEquals(SoftwareUpdater.Status.UP_TO_DATE, current.status);
        assertTrue(current.message.contains("abc1234"));

        SoftwareUpdater.CheckResult available = SoftwareUpdater.fromGitState(
            "?? video.tif", "3", "aaa Fix\nbbb Docs", true, "abc1234");
        assertEquals(SoftwareUpdater.Status.UPDATE_AVAILABLE, available.status);
        assertEquals(3, available.behindCount);
        assertTrue(available.message.contains("3 new commits"));

        SoftwareUpdater.CheckResult diverged = SoftwareUpdater.fromGitState(
            "", "2", "aaa Fix", false, "abc1234");
        assertEquals(SoftwareUpdater.Status.NOT_FAST_FORWARD, diverged.status);

        SoftwareUpdater.CheckResult badCount = SoftwareUpdater.fromGitState(
            "", "nope", "", true, "");
        assertEquals(SoftwareUpdater.Status.ERROR, badCount.status);
    }

    @Test
    void fromGitStatePrioritizesDirtyOverDivergenceAndBehind() {
        SoftwareUpdater.CheckResult dirtyWins = SoftwareUpdater.fromGitState(
            "UU conflict.java", "9", "aaa Fix", false, "deadbee");
        assertEquals(SoftwareUpdater.Status.DIRTY, dirtyWins.status);
        assertTrue(dirtyWins.message.contains("conflict.java"));
    }

    @Test
    void fromGitStateTreatsLocalAheadAsUpToDate() {
        // behind == 0 even if HEAD is not an ancestor of FETCH_HEAD (local extra commits)
        SoftwareUpdater.CheckResult ahead = SoftwareUpdater.fromGitState(
            "", "0", "", false, "loc4l01");
        assertEquals(SoftwareUpdater.Status.UP_TO_DATE, ahead.status);
    }

    @Test
    void fromGitStateBlankShaStillUpToDate() {
        SoftwareUpdater.CheckResult current = SoftwareUpdater.fromGitState(
            "", "0", "", true, "");
        assertEquals(SoftwareUpdater.Status.UP_TO_DATE, current.status);
        assertFalse(current.message.contains("commit"));
    }

    @Test
    void summarizeDirtyFilesSkipsUntrackedAndTruncates() {
        String summary = SoftwareUpdater.summarizeDirtyFiles(
            "?? video.tif\n M pom.xml\nM  README.md");
        assertTrue(summary.contains("pom.xml"));
        assertTrue(summary.contains("README.md"));
        assertFalse(summary.contains("video.tif"));

        assertEquals("", SoftwareUpdater.summarizeDirtyFiles(null));
        assertEquals("", SoftwareUpdater.summarizeDirtyFiles("?? only-untracked.tif"));

        StringBuilder many = new StringBuilder();
        for (int i = 1; i <= 20; i++) {
            many.append(" M file").append(i).append(".java\n");
        }
        String truncated = SoftwareUpdater.summarizeDirtyFiles(many.toString());
        assertTrue(truncated.contains("file1.java"));
        assertTrue(truncated.contains("file8.java"));
        assertTrue(truncated.contains("..."));
        assertFalse(truncated.contains("file9.java"));
    }

    @Test
    void summarizeDirtyFilesKeepsRenameArrowText() {
        String summary = SoftwareUpdater.summarizeDirtyFiles("R  old.java -> new.java");
        assertTrue(summary.contains("old.java -> new.java"));
    }

    @Test
    void performCheckRejectsMissingAndNonGitRoots(@TempDir Path temp) throws IOException {
        SoftwareUpdater.CheckResult missing = SoftwareUpdater.performCheck(
            new File(temp.toFile(), "does-not-exist"));
        assertEquals(SoftwareUpdater.Status.ERROR, missing.status);
        assertTrue(missing.message.contains("install directory"));

        SoftwareUpdater.CheckResult notDir = SoftwareUpdater.performCheck(null);
        assertEquals(SoftwareUpdater.Status.ERROR, notDir.status);

        Path file = temp.resolve("just-a-file.txt");
        Files.writeString(file, "nope");
        SoftwareUpdater.CheckResult asFile = SoftwareUpdater.performCheck(file.toFile());
        assertEquals(SoftwareUpdater.Status.ERROR, asFile.status);

        Path empty = Files.createDirectory(temp.resolve("empty-install"));
        SoftwareUpdater.CheckResult zipLike = SoftwareUpdater.performCheck(empty.toFile());
        assertEquals(SoftwareUpdater.Status.NOT_A_CLONE, zipLike.status);
        assertTrue(zipLike.message.contains("git clone"));
    }

    @Test
    void spawnApplyScriptFailsWhenHelperMissing(@TempDir Path temp) {
        IOException thrown = assertThrows(IOException.class,
            () -> SoftwareUpdater.spawnApplyScript(temp.toFile()));
        assertTrue(thrown.getMessage().contains("Update script not found"));
    }

    @Test
    void resolveRepoRootIsAbsolute() {
        File root = SoftwareUpdater.resolveRepoRoot();
        assertTrue(root.isAbsolute());
    }

    @Test
    void resolveInstallModeDefaultsToCpuWhenUnset() {
        String mode = SoftwareUpdater.resolveInstallMode();
        assertTrue(mode.equals(Constants.MODE_CPU) || mode.equals(Constants.MODE_GPU));
    }

    @Test
    void checkResultFactoriesCoverNullMessages() {
        assertEquals("Could not check for updates.",
            SoftwareUpdater.CheckResult.error(null).message);
        assertEquals("Could not check for updates.",
            SoftwareUpdater.CheckResult.error("  ").message);
        assertTrue(SoftwareUpdater.CheckResult.notAClone().message.contains(Constants.GITHUB_REPO_URL));
        assertTrue(SoftwareUpdater.CheckResult.notFastForward().message.contains("fast-forward"));
    }

    @Test
    void dirtyFactoryOmitsFileListWhenEmpty() {
        SoftwareUpdater.CheckResult dirty = SoftwareUpdater.CheckResult.dirty("");
        assertEquals(SoftwareUpdater.Status.DIRTY, dirty.status);
        assertFalse(dirty.message.contains("Changed files:"));
    }

    @Test
    void availableResultCapsLogToTenLines() {
        List<String> log = new ArrayList<>();
        for (int i = 1; i <= 20; i++) {
            log.add(i + " change " + i);
        }
        SoftwareUpdater.CheckResult result = SoftwareUpdater.CheckResult.available(
            20, "abc", String.join("\n", log));
        assertEquals(SoftwareUpdater.Status.UPDATE_AVAILABLE, result.status);
        assertEquals(20, result.behindCount);
        assertTrue(result.message.contains("20 new commits"));
        assertTrue(result.message.contains("1 change 1"));
        assertTrue(result.message.contains("10 change 10"));
        assertFalse(result.message.contains("11 change 11"));
    }
}
