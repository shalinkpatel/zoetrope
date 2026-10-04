//! Where pi keeps its sessions on disk, and how to tell what a file is.
//!
//! `<agent-dir>/sessions/--<encoded-cwd>--/<timestamp>_<session-id>.jsonl`,
//! where the agent dir is `$PI_CODING_AGENT_DIR` or `~/.pi/agent` and the cwd
//! is encoded by dropping its leading separator and turning every `/`, `\`
//! and `:` into `-`. What a file is comes from its first line, the `session`
//! header, which carries the session id and the cwd; the file name carries the
//! id too, which lets an id lookup prune without reading. A forked session
//! (`/fork`, `/clone`) is a root of its own that names its origin.
//!
//! Fabric agents a session spawns run as separate pi processes, each
//! exporting a file of its own into a hidden sibling tree:
//! `sessions/.fabric/--<encoded-cwd>--/<timestamp>_<run-id>.jsonl`, the run id
//! being the one the spawning call's result and Fabric's completion report
//! name. Those exports are a session's agent files. They are ordinary pi files
//! whose header names only the run, never the session that spawned it, so
//! this provider is the one place a file's session is not in the file: a
//! child is joined to its session by finding the session file whose spawn
//! results or completion reports name its run. That search reads more than a
//! head, so it is bounded and remembered:
//!
//! - only session files in the child's project directory, or a project
//!   directory above it (an agent may run in a subdirectory), created no
//!   later than the child and written since;
//! - each candidate is scanned once for the runs it names, and afterwards
//!   only for what was appended (the scans are kept per path for the
//!   process), so a live session's growing file is never read twice.
//!
//! A child whose session is not found is classified as its own orphaned
//! session, which no sweep lists and a rescan remembers as a stranger; one
//! created moments ago is not classified at all, since its spawner may not
//! have written the result yet, and is retried. In the browser, which has no
//! filesystem to search, child exports are not classified.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, SystemTime};

use super::wire::{Header, parse_header};
use crate::provider::{FileRole, Provider, ReadMode, Scope, SessionFile};

/// The pi agent directory: `$PI_CODING_AGENT_DIR`, else `~/.pi/agent`.
pub fn agent_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("PI_CODING_AGENT_DIR").filter(|d| !d.is_empty()) {
        let dir = PathBuf::from(dir);
        // pi expands a leading `~` itself, so the variable may carry one.
        if let Ok(rest) = dir.strip_prefix("~") {
            return home().map(|h| h.join(rest));
        }
        return Some(dir);
    }
    Some(home()?.join(".pi").join("agent"))
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
}

/// The directory pi names after a working directory, under `sessions/`.
pub fn encode_cwd(cwd: &Path) -> String {
    let s = cwd.to_string_lossy();
    let s = s.strip_prefix(['/', '\\']).unwrap_or(&s);
    let body: String = s
        .chars()
        .map(|c| {
            if matches!(c, '/' | '\\' | ':') {
                '-'
            } else {
                c
            }
        })
        .collect();
    format!("--{body}--")
}

/// Whether a path is shaped like a session file. Shape only; the header
/// decides what it is.
pub fn is_session_file(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "jsonl")
}

/// The session id a file name carries: what follows the first `_` of
/// `<timestamp>_<id>.jsonl`. Cheaper than reading the file; the header is
/// the truth.
pub fn id_from_path(path: &Path) -> Option<&str> {
    let stem = path.file_stem()?.to_str()?;
    stem.split_once('_')
        .map(|(_, id)| id)
        .filter(|id| !id.is_empty())
}

/// The header: the file's first line, when it is one.
pub fn read_header(path: &Path) -> Option<Header> {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path).ok()?;
    let mut first = String::new();
    BufReader::new(file).read_line(&mut first).ok()?;
    parse_header(&first)
}

/// The session files in one project directory, in name (so creation) order.
fn files_in(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| is_session_file(p) && p.is_file())
        .collect();
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// The provider primitives (see `provider/mod.rs` and docs/DISCOVERY.md)
// ---------------------------------------------------------------------------

/// Every session file under `<agent-dir>/sessions`. A project scope narrows
/// to that project's directory, `since` filters by mtime, an id prefix by
/// file name.
pub fn all_paths(scope: &Scope) -> Vec<PathBuf> {
    let Some(agent) = agent_dir() else {
        return Vec::new();
    };
    all_paths_under(&agent.join("sessions"), scope)
}

/// [`all_paths`] under a given sessions directory.
pub(crate) fn all_paths_under(sessions: &Path, scope: &Scope) -> Vec<PathBuf> {
    let dirs: Vec<PathBuf> = match &scope.project {
        Some(cwd) => vec![sessions.join(encode_cwd(cwd))],
        None => {
            let Ok(entries) = std::fs::read_dir(sessions) else {
                return Vec::new();
            };
            let mut dirs: Vec<PathBuf> = entries
                .flatten()
                .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect();
            dirs.sort();
            dirs
        }
    };
    let mut out: Vec<PathBuf> = dirs.iter().flat_map(|d| files_in(d)).collect();
    if let Some(prefix) = &scope.id_prefix {
        out.retain(|p| id_from_path(p).is_some_and(|id| id.starts_with(prefix.as_str())));
    }
    if let Some(since) = scope.since {
        out.retain(|p| crate::provider::modified(p) >= since);
    }
    out
}

/// What a file is: a session's own file by its header, a child export by
/// the session file that names its run.
pub fn session_file(path: &Path) -> Option<SessionFile> {
    if !is_session_file(path) {
        return None;
    }
    let header = read_header(path)?;
    let modified = crate::provider::modified(path);
    if is_child_path(path) {
        return classify_child(path, &header, modified);
    }
    from_header(path, &header, modified)
}

/// [`session_file`] given the first line already read: the browser's way.
/// A child export's session cannot be found without a filesystem.
pub fn classify_head(path: &Path, head: &str, modified: SystemTime) -> Option<SessionFile> {
    if is_child_path(path) {
        return None;
    }
    let first = head.lines().find(|l| !l.trim().is_empty())?;
    from_header(path, &parse_header(first)?, modified)
}

fn from_header(path: &Path, header: &Header, modified: SystemTime) -> Option<SessionFile> {
    Some(SessionFile {
        provider: Provider::Pi,
        path: path.to_path_buf(),
        session: header.id.clone()?,
        role: FileRole::Root,
        read: ReadMode::Tail,
        project_key: header.cwd.clone().unwrap_or_default(),
        modified,
    })
}

/// Where the rest of a file's session can be. From a session's own file: the
/// exports in its project's `.fabric` directories (and those below it) dated
/// between its creation and its last write. From a child: its session's
/// file, then that file's candidates.
pub fn related_paths(file: &SessionFile) -> Vec<PathBuf> {
    let mut out = match file.role {
        FileRole::Root => children_near(&file.path, file.modified),
        _ => {
            let Some(run) = read_header(&file.path).and_then(|h| h.id) else {
                return Vec::new();
            };
            let Some(root) = find_parent(&file.path, &run) else {
                return Vec::new();
            };
            let mut out = children_near(&root, crate::provider::modified(&root));
            out.insert(0, root);
            out
        }
    };
    out.retain(|p| *p != file.path);
    out
}

// ---------------------------------------------------------------------------
// Fabric child exports
// ---------------------------------------------------------------------------

/// How long after its creation a child whose session is not found is still
/// retried rather than called an orphan: the spawner writes the result that
/// names the run once its program returns, seconds after the child starts.
const CHILD_GRACE: Duration = Duration::from_secs(120);

/// Slack on a session's last write when looking for its children, for a
/// child created in the same moment as the write that spawned it.
const WRITE_SLACK: Duration = Duration::from_secs(60);

/// Whether a path is a Fabric agent's export: `<sessions>/.fabric/<dir>/<file>`.
pub fn is_child_path(path: &Path) -> bool {
    path.parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .is_some_and(|n| n == FABRIC_DIR)
}

const FABRIC_DIR: &str = ".fabric";

/// The `<timestamp>` a file name starts with, `2026-10-02T19-43-54-206Z`.
fn stamp(path: &Path) -> Option<&str> {
    let stem = path.file_stem()?.to_str()?;
    stem.split_once('_').map(|(t, _)| t)
}

fn stamp_time(stamp: &str) -> Option<SystemTime> {
    let t = chrono::NaiveDateTime::parse_from_str(stamp, "%Y-%m-%dT%H-%M-%S-%3fZ").ok()?;
    Some(SystemTime::from(t.and_utc()))
}

/// Whether project directory `inner` is `outer` or a directory below it, by
/// their encoded names. The encoding is lossy, so this may over-include; the
/// run id decides.
fn nests(outer: &str, inner: &str) -> bool {
    inner == outer
        || outer
            .strip_suffix("--")
            .is_some_and(|o| inner.strip_prefix(o).is_some_and(|r| r.starts_with('-')))
}

/// The non-hidden directories directly under `dir` whose names satisfy `keep`.
fn dirs_where(dir: &Path, keep: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            !name.starts_with('.') && keep(&name)
        })
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out
}

/// Child exports that could be a session's: in its project's `.fabric`
/// directory or one below it, created between the session's own creation
/// and its last write.
fn children_near(root: &Path, last_write: SystemTime) -> Vec<PathBuf> {
    let (Some(sessions), Some(dir)) = (
        root.parent().and_then(Path::parent),
        root.parent()
            .and_then(Path::file_name)
            .and_then(|n| n.to_str()),
    ) else {
        return Vec::new();
    };
    let born = stamp(root);
    dirs_where(&sessions.join(FABRIC_DIR), |d| nests(dir, d))
        .iter()
        .flat_map(|d| files_in(d))
        .filter(|p| match (stamp(p), born) {
            (Some(s), Some(b)) => {
                s >= b && stamp_time(s).is_none_or(|t| t <= last_write + WRITE_SLACK)
            }
            _ => true,
        })
        .collect()
}

/// A child export: in the session whose file names its run.
fn classify_child(path: &Path, header: &Header, modified: SystemTime) -> Option<SessionFile> {
    let run = header.id.clone()?;
    let (session, project_key) = match find_parent(path, &run) {
        Some(root) => {
            let h = read_header(&root)?;
            (h.id?, h.cwd.unwrap_or_default())
        }
        None => {
            let fresh = stamp(path)
                .and_then(stamp_time)
                .is_some_and(|t| SystemTime::now() < t + CHILD_GRACE);
            if fresh {
                return None;
            }
            // Its own session, with no root: never listed, never adopted.
            (run, header.cwd.clone().unwrap_or_default())
        }
    };
    Some(SessionFile {
        provider: Provider::Pi,
        path: path.to_path_buf(),
        role: FileRole::Agent {
            parent: session.clone(),
        },
        session,
        read: ReadMode::Tail,
        project_key,
        modified,
    })
}

/// The session file that names `run`, among those that could have spawned
/// the child at `child`: newest first.
fn find_parent(child: &Path, run: &str) -> Option<PathBuf> {
    let sessions = child.parent()?.parent()?.parent()?;
    let dir = child.parent()?.file_name()?.to_str()?;
    let born = stamp(child);
    let born_at = born.and_then(stamp_time);
    let mut candidates: Vec<PathBuf> = dirs_where(sessions, |d| nests(d, dir))
        .iter()
        .flat_map(|d| files_in(d))
        .filter(|p| match (stamp(p), born) {
            (Some(s), Some(b)) => s <= b,
            _ => true,
        })
        .filter(|p| born_at.is_none_or(|t| crate::provider::modified(p) >= t))
        .collect();
    candidates.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    candidates.into_iter().find(|p| names_run(p, run))
}

/// What has been read of one session file, and the runs it named.
#[derive(Default)]
struct Scan {
    len: u64,
    runs: HashSet<String>,
}

static SCANS: LazyLock<Mutex<HashMap<PathBuf, Scan>>> = LazyLock::new(Default::default);

/// Whether the session file at `path` names `run` as one it spawned or was
/// told ended. Reads only what was appended since the last call; a file that
/// shrank (rewritten) is read again from the start.
fn names_run(path: &Path, run: &str) -> bool {
    use std::io::{BufRead, BufReader, Seek, SeekFrom};
    let mut scans = SCANS.lock().unwrap_or_else(|e| e.into_inner());
    let scan = scans.entry(path.to_path_buf()).or_default();
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if len < scan.len {
        *scan = Scan::default();
    }
    if len > scan.len
        && let Ok(mut file) = std::fs::File::open(path)
        && file.seek(SeekFrom::Start(scan.len)).is_ok()
    {
        let mut reader = BufReader::new(file);
        let mut line = Vec::new();
        // Whole lines only: a line still being written is read next time.
        while reader.read_until(b'\n', &mut line).is_ok_and(|n| n > 0) && line.ends_with(b"\n") {
            scan.len += line.len() as u64;
            if let Ok(text) = std::str::from_utf8(&line) {
                scan.runs.extend(super::runs_named(text));
            }
            line.clear();
        }
    }
    scan.runs.contains(run)
}

/// pi records the working directory as it is in the header.
pub fn project_key(cwd: &Path) -> String {
    cwd.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cwd_encoding_matches_pi() {
        assert_eq!(
            encode_cwd(Path::new("/Users/shalin.patel/dev/repos/dotfiles")),
            "--Users-shalin.patel-dev-repos-dotfiles--"
        );
        assert_eq!(encode_cwd(Path::new("/private/tmp")), "--private-tmp--");
        assert_eq!(encode_cwd(Path::new("C:\\work\\x")), "--C--work-x--");
    }

    #[test]
    fn id_is_after_the_first_underscore() {
        let p = Path::new(
            "/s/--p--/2026-09-21T18-46-48-895Z_01a0c54a-cd3f-706b-ae92-c880ee68b69e.jsonl",
        );
        assert!(is_session_file(p));
        assert_eq!(
            id_from_path(p),
            Some("01a0c54a-cd3f-706b-ae92-c880ee68b69e")
        );
        assert_eq!(id_from_path(Path::new("/s/plain.jsonl")), None);
        assert!(!is_session_file(Path::new("/s/x.json")));
    }

    #[test]
    fn head_classifies_a_root() {
        let f = classify_head(
            Path::new("/s/--p--/t_abc.jsonl"),
            "\n{\"type\":\"session\",\"version\":3,\"id\":\"abc\",\"cwd\":\"/p\"}\n",
            SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        assert_eq!(f.session, "abc");
        assert_eq!(f.role, FileRole::Root);
        assert_eq!(f.project_key, project_key(Path::new("/p")));
        assert!(
            classify_head(
                Path::new("/s/x.jsonl"),
                r#"{"type":"message","id":"a","parentId":null}"#,
                SystemTime::UNIX_EPOCH
            )
            .is_none()
        );
    }

    /// The fixtures are laid out like `<agent-dir>/sessions`: a sweep finds
    /// them by project, by id prefix, and skips hidden directories.
    #[test]
    fn sweep_layout_over_the_fixtures() {
        let Some(dir) = crate::provider::harness::fixture_dir("pi") else {
            return;
        };
        let sessions = dir.join("hunk-review");
        let all = all_paths_under(&sessions, &Scope::ALL);
        assert_eq!(all.len(), 1);
        let f = session_file(&all[0]).unwrap();
        assert_eq!(f.session, "01a0d052-a114-73f1-bc57-0a00775efaff");
        assert_eq!(f.project_key, "/Users/me/dev/repos/pi-hunk-island");
        let here = Scope::project(Path::new("/Users/me/dev/repos/pi-hunk-island"));
        assert_eq!(all_paths_under(&sessions, &here), all);
        let elsewhere = Scope::project(Path::new("/Users/me/elsewhere"));
        assert!(all_paths_under(&sessions, &elsewhere).is_empty());
        assert_eq!(all_paths_under(&sessions, &Scope::id("01a0d0")), all);
        assert!(all_paths_under(&sessions, &Scope::id("ffff")).is_empty());
        assert!(related_paths(&f).is_empty());
    }
}
