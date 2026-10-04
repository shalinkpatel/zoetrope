//! The pi provider: session-file lines in, [`Fact`]s out.
//!
//! A pi session is its own file, by `main`, plus one file per Fabric agent
//! it spawned (see [`discovery`]), each by that agent. A [`Stream`] knows
//! which kind it reads: the session's own file states `main` from its
//! header; a child export learns its run id from its header, the way a Codex
//! stream learns its thread, and states nothing before it.
//!
//! What this module knows that the model must not: that a `toolResult`
//! carries its outcome in `isError`; that `bashExecution` is a command the
//! user ran with `!`, a whole tool run in one record; that a code-mode call
//! whose program calls `agents.spawn` or `agents.run` spawns, and that its
//! result's audit trail names each run it started (the join from call to
//! child); that an audited `agents.wait` or `agents.run` that returned a
//! finished run, and Fabric's `pi-fabric-agent-complete` report, observe that
//! run's end; that other `custom` and `custom_message` entries are extensions
//! talking, and a `system` message a prompt patch; that `compaction` and
//! `branch_summary` are bookkeeping, counted but not activity; and how to
//! render each pi tool's arguments as a one-line summary.
//!
//! What it deliberately does not say: an `agents.status` result is a peek
//! at some run, possibly another session's, and states nothing; a run that
//! no record reports finished is left to time-derived liveness.

use chrono::{DateTime, Utc};
use serde_json::Value;

pub mod discovery;
pub mod wire;

use crate::fact::{AgentKind, AgentStatus, Fact, FactKind, Outcome, Statement};
use crate::provider::summary::{short_path, truncate_summary};
use crate::state::session::MAIN_ID;
use wire::{Audit, Block, Body, Entry, Header, Message, Run, parse_line};

/// The `customType` of Fabric's report that background runs ended.
const AGENT_COMPLETE: &str = "pi-fabric-agent-complete";

/// The prefix Fabric gives a child export's session name.
const CHILD_NAME_PREFIX: &str = "fabricagent-";

/// One pi file being read: the session's own, or a Fabric agent's export.
#[derive(Debug, Clone)]
pub struct Stream {
    /// Whether this is a child export rather than the session's own file.
    child: bool,
    /// Whose file this is: `main`, or for a child the run id its header
    /// names. A child's lines before its header state nothing.
    owner: Option<String>,
    /// Whether the owner has been stated.
    announced: bool,
    /// From the header, for summaries.
    cwd: Option<String>,
}

impl Default for Stream {
    fn default() -> Self {
        Stream::new()
    }
}

impl Stream {
    /// A stream for a session's own file.
    pub fn new() -> Self {
        Stream {
            child: false,
            owner: Some(MAIN_ID.to_string()),
            announced: false,
            cwd: None,
        }
    }

    /// A stream for a Fabric agent's export, spawned by the session's `main`.
    pub fn child() -> Self {
        Stream {
            child: true,
            owner: None,
            announced: false,
            cwd: None,
        }
    }

    /// Parse one line and state what it says. `None` for a blank or
    /// unparsable line, or one that states nothing.
    pub fn push(&mut self, line: &str) -> Option<Statement> {
        let entry = parse_line(line)?;
        self.push_entry(&entry)
    }

    /// State what an already-parsed entry says.
    pub fn push_entry(&mut self, entry: &Entry) -> Option<Statement> {
        let at = entry.timestamp;
        if let Body::Session(h) = &entry.body {
            return self.header(h, at);
        }
        let owner = self.owner.clone()?;
        let mut facts = self.facts(&owner, entry);
        if facts.is_empty() {
            return None;
        }
        for f in &mut facts {
            if f.ts.is_none() && !f.is_session_meta() {
                f.ts = at;
            }
        }
        // A session file read without its header (a fragment, a truncated
        // copy) still has a root: stated on the first line that is activity,
        // since a metadata-only statement stays off the timeline.
        if !self.announced && !self.child && !facts.iter().all(Fact::is_session_meta) {
            self.announced = true;
            facts.insert(0, main_agent(at));
        }
        Some(Statement { at, facts })
    }

    /// The header: whose file this is, and for the session's own file the
    /// rows it carries.
    fn header(&mut self, h: &Header, at: Option<DateTime<Utc>>) -> Option<Statement> {
        if self.announced {
            // A second header states nothing; the first decided.
            return None;
        }
        self.cwd = h.cwd.clone();
        let mut facts = Vec::new();
        if self.child {
            let id = h.id.clone()?;
            self.owner = Some(id.clone());
            facts.push(Fact {
                agent: Some(id),
                ts: at,
                kind: FactKind::Agent {
                    kind: AgentKind::Subagent,
                    parent: Some(MAIN_ID.to_string()),
                    agent_type: None,
                    description: None,
                    // The spawning call is the parent's record to state.
                    spawned_by: None,
                    interactive: false,
                },
            });
        } else {
            facts.push(main_agent(at));
            push_session(&mut facts, "cwd", h.cwd.clone());
            // A fork names the file it came from; its id is the file's tail.
            push_session(
                &mut facts,
                "forked from",
                h.parent_session.as_deref().map(|p| {
                    discovery::id_from_path(std::path::Path::new(p))
                        .map(str::to_owned)
                        .unwrap_or_else(|| p.to_string())
                }),
            );
        }
        self.announced = true;
        Some(Statement { at, facts })
    }

    /// Facts stated by one entry after the header, by `owner`.
    fn facts(&self, owner: &str, entry: &Entry) -> Vec<Fact> {
        let mut out = Vec::new();
        let by = |kind| Fact {
            agent: Some(owner.to_string()),
            ts: None,
            kind,
        };
        match &entry.body {
            Body::Message(Message::User { content }) => {
                // A child's user message is its spawner's task, not a person.
                let text: Vec<&str> = content
                    .iter()
                    .filter_map(|b| match b {
                        Block::Text(t) if !self.child => Some(t.as_str()),
                        _ => None,
                    })
                    .collect();
                let text = text.join("\n");
                if !text.trim().is_empty() {
                    out.push(by(FactKind::Prompt(text.trim_end().to_string())));
                }
                ensure_activity(&mut out, owner);
            }
            Body::Message(Message::Assistant(a)) => {
                if let Some(model) = &a.model {
                    out.push(by(FactKind::Model(model.clone())));
                }
                if let Some(output) = a.output_tokens.filter(|&n| n > 0) {
                    // One message is one response: its usage is its own delta.
                    out.push(by(FactKind::Tokens {
                        output,
                        dedup: None,
                    }));
                }
                // Blocks in order, so a call follows the text that led to it
                // and a spawn carries that text as its stated reason.
                for block in &a.content {
                    match block {
                        Block::Text(t) | Block::Thinking(t) if !t.trim().is_empty() => {
                            out.push(by(FactKind::Reasoning(t.clone())));
                        }
                        Block::ToolCall {
                            id: Some(call),
                            name,
                            arguments,
                        } => {
                            let name = name.clone().unwrap_or_default();
                            let summary = summarize_tool(&name, arguments, self.cwd.as_deref());
                            out.push(by(FactKind::ToolStart {
                                call: call.clone(),
                                name,
                                summary,
                            }));
                            if spawns(arguments) {
                                out.push(by(FactKind::Spawn { call: call.clone() }));
                            }
                        }
                        _ => {}
                    }
                }
                ensure_activity(&mut out, owner);
            }
            Body::Message(Message::ToolResult {
                tool_call_id,
                is_error,
                audits,
            }) => {
                if let Some(call) = tool_call_id {
                    let outcome = if *is_error == Some(true) {
                        Outcome::Err
                    } else {
                        Outcome::Ok
                    };
                    out.push(by(FactKind::ToolEnd {
                        call: call.clone(),
                        outcome,
                    }));
                    out.extend(audited_runs(owner, call, audits));
                }
                ensure_activity(&mut out, owner);
            }
            // Written once the command has finished: start and outcome in one
            // record, keyed by the entry, since the format gives it no call id.
            Body::Message(Message::BashExecution {
                command,
                exit_code,
                cancelled,
            }) => {
                if let Some(call) = &entry.id {
                    out.push(by(FactKind::ToolStart {
                        call: call.clone(),
                        name: "bash".into(),
                        summary: command.as_deref().map(truncate_summary),
                    }));
                    let outcome = match exit_code {
                        Some(0) if !cancelled => Outcome::Ok,
                        _ => Outcome::Err,
                    };
                    out.push(by(FactKind::ToolEnd {
                        call: call.clone(),
                        outcome,
                    }));
                }
                ensure_activity(&mut out, owner);
            }
            // Fabric's report that background runs ended: each run's own
            // ending, about that run, whoever's file carried it.
            Body::CustomMessage {
                custom_type,
                content,
                details,
            } if custom_type == AGENT_COMPLETE => {
                for run in completed_runs(content, details) {
                    if let Some(name) = &run.name {
                        out.push(about(&run.id, label(name)));
                    }
                    if let Some(status) = run.status.as_deref().and_then(terminal_status) {
                        out.push(about(&run.id, FactKind::Ended(status)));
                    }
                }
            }
            Body::ModelChange { model_id: Some(m) } => out.push(by(FactKind::Model(m.clone()))),
            // A child's name: Fabric names its export after the run.
            Body::SessionInfo { name: Some(name) } if self.child => {
                let name = name.trim();
                let name = name.strip_prefix(CHILD_NAME_PREFIX).unwrap_or(name);
                if !name.is_empty() {
                    out.push(about(owner, label(name)));
                }
            }
            // Session-level rows come from the session's own file only.
            _ if self.child => {}
            Body::ThinkingLevelChange { level } => {
                push_session(&mut out, "thinking", level.clone())
            }
            Body::SessionInfo { name: Some(name) } if !name.trim().is_empty() => {
                out.push(meta(FactKind::Title(name.trim().to_string())));
            }
            Body::Compaction => out.push(meta(FactKind::Tally("compactions".into()))),
            Body::BranchSummary => out.push(meta(FactKind::Tally("branch summaries".into()))),
            // `system` prompt patches and extension roles; other extension
            // entries, usage records, context edits, labels: nobody's activity.
            Body::Message(Message::Other(_))
            | Body::CustomMessage { .. }
            | Body::ModelChange { .. }
            | Body::SessionInfo { .. }
            | Body::Session(_)
            | Body::Other(_) => {}
        }
        out
    }
}

fn main_agent(ts: Option<DateTime<Utc>>) -> Fact {
    Fact {
        agent: Some(MAIN_ID.to_string()),
        ts,
        kind: FactKind::Agent {
            kind: AgentKind::Main,
            parent: None,
            agent_type: Some("pi".into()),
            description: None,
            spawned_by: None,
            interactive: true,
        },
    }
}

/// Whether a code-mode call's program starts Fabric agents. Its result's
/// audit trail names the runs.
fn spawns(arguments: &Value) -> bool {
    arguments
        .get("code")
        .and_then(Value::as_str)
        .is_some_and(|code| code.contains("agents.spawn(") || code.contains("agents.run("))
}

/// What a code-mode call's audit trail says about agent runs: each run a
/// spawning call started is an agent the call spawned, under the caller; a
/// run a waiting call saw finish has ended.
fn audited_runs(owner: &str, call: &str, audits: &[Audit]) -> Vec<Fact> {
    let mut out = Vec::new();
    for audit in audits.iter().filter(|a| a.success != Some(false)) {
        let spawning = matches!(audit.reference.as_str(), "agents.spawn" | "agents.run");
        let waiting = matches!(audit.reference.as_str(), "agents.wait" | "agents.run");
        for run in &audit.runs {
            if spawning {
                out.push(about(
                    &run.id,
                    FactKind::Agent {
                        kind: AgentKind::Subagent,
                        parent: Some(owner.to_string()),
                        agent_type: run.name.clone(),
                        description: None,
                        spawned_by: Some(call.to_string()),
                        interactive: false,
                    },
                ));
            }
            if waiting && let Some(status) = run.status.as_deref().and_then(terminal_status) {
                out.push(about(&run.id, FactKind::Ended(status)));
            }
        }
    }
    out
}

/// The runs a `pi-fabric-agent-complete` report says ended. Two shapes: one
/// run, its whole record in `details`; or a batch, `details.ids` with one
/// `Agent <name> (<id>) <status> after <duration>:` header per run in the
/// text. Only the listed ids are read off the text, so a report quoting
/// another run's header is not mistaken for one.
fn completed_runs(content: &str, details: &Value) -> Vec<Run> {
    if let Some(run) = wire::run(details) {
        return vec![run];
    }
    let Some(ids) = details.get("ids").and_then(Value::as_array) else {
        return Vec::new();
    };
    ids.iter()
        .filter_map(Value::as_str)
        .map(|id| {
            let marker = format!("({id}) ");
            let header = content.lines().find(|l| l.contains(&marker));
            let status = header
                .and_then(|l| l.split(&marker).nth(1))
                .and_then(|rest| rest.split_whitespace().next())
                .map(|s| s.trim_end_matches(':').to_string());
            let name = header
                .and_then(|l| l.split(&marker).next())
                .and_then(|before| before.trim_start().strip_prefix("Agent "))
                .map(|n| n.trim().to_string())
                .filter(|n| !n.is_empty());
            Run {
                id: id.to_string(),
                name,
                status,
            }
        })
        .collect()
}

/// The runs one line of a session file names as its own: started by an
/// audited spawning call, or reported ended by Fabric. What discovery joins a
/// child export to its session by. Cheap for every other line.
pub(crate) fn runs_named(line: &str) -> Vec<String> {
    if !(line.contains("agents.spawn")
        || line.contains("agents.run")
        || line.contains(AGENT_COMPLETE))
    {
        return Vec::new();
    }
    match parse_line(line).map(|e| e.body) {
        Some(Body::Message(Message::ToolResult { audits, .. })) => audits
            .iter()
            .filter(|a| matches!(a.reference.as_str(), "agents.spawn" | "agents.run"))
            .flat_map(|a| a.runs.iter().map(|r| r.id.clone()))
            .collect(),
        Some(Body::CustomMessage {
            custom_type,
            content,
            details,
        }) if custom_type == AGENT_COMPLETE => completed_runs(&content, &details)
            .into_iter()
            .map(|r| r.id)
            .collect(),
        _ => Vec::new(),
    }
}

/// A Fabric run status the model can act on. An unrecognised one states
/// nothing rather than overriding derived liveness.
fn terminal_status(status: &str) -> Option<AgentStatus> {
    match status {
        "completed" | "succeeded" | "done" => Some(AgentStatus::Done),
        // A run Fabric killed at its deadline did not finish: not a success,
        // and not the user's doing either.
        "failed" | "error" | "timed_out" => Some(AgentStatus::Failed),
        "stopped" | "cancelled" | "canceled" | "aborted" => Some(AgentStatus::Stopped),
        _ => None,
    }
}

fn about(agent: &str, kind: FactKind) -> Fact {
    Fact {
        agent: Some(agent.to_string()),
        ts: None,
        kind,
    }
}

fn label(name: &str) -> FactKind {
    FactKind::Label {
        agent_type: Some(name.to_string()),
        description: None,
    }
}

/// A message is its owner's activity even when it states nothing else.
fn ensure_activity(out: &mut Vec<Fact>, owner: &str) {
    if !out.iter().any(|f| f.agent.as_deref() == Some(owner)) {
        out.push(about(owner, FactKind::Activity));
    }
}

fn meta(kind: FactKind) -> Fact {
    Fact {
        agent: None,
        ts: None,
        kind,
    }
}

fn push_session(out: &mut Vec<Fact>, label: &str, value: Option<String>) {
    if let Some(value) = value.filter(|v| !v.is_empty()) {
        out.push(meta(FactKind::Session {
            label: label.into(),
            value,
        }));
    }
}

// ---------------------------------------------------------------------------
// Tool summaries: the pi tool vocabulary and what makes a good one-liner
// ---------------------------------------------------------------------------

/// One line for a tool call's arguments, if a natural field exists for the
/// tool. pi's built-ins first, then the common extension tools, then the
/// fields any tool tends to name itself by.
pub(crate) fn summarize_tool(name: &str, args: &Value, cwd: Option<&str>) -> Option<String> {
    let pick = |key: &str| {
        args.get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(truncate_summary)
    };
    // The first string of an array field (`queries`, `urls`).
    let first = |key: &str| {
        args.get(key)
            .and_then(Value::as_array)
            .and_then(|a| a.iter().find_map(Value::as_str))
            .map(truncate_summary)
    };
    match name {
        "bash" => pick("command"),
        "read" | "write" | "edit" | "ls" => args
            .get("path")
            .and_then(Value::as_str)
            .map(|p| short_path(p, cwd)),
        "grep" | "find" => pick("pattern"),
        // Fabric's code tool names each program in `display` (an object with
        // a `name`, or a bare string); otherwise its first line of code.
        "fabric_exec" => match args.get("display") {
            Some(Value::String(s)) if !s.trim().is_empty() => Some(truncate_summary(s)),
            Some(d) => d
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(truncate_summary),
            None => None,
        }
        .or_else(|| {
            args.get("code")
                .and_then(Value::as_str)
                .and_then(|c| c.lines().map(str::trim).find(|l| !l.is_empty()))
                .map(truncate_summary)
        }),
        "web_search" => pick("query").or_else(|| first("queries")),
        "fetch_content" => pick("url").or_else(|| first("urls")),
        "mcp" => pick("tool")
            .or_else(|| pick("server"))
            .or_else(|| pick("connect")),
        _ => ["description", "command", "path", "query", "url", "pattern"]
            .into_iter()
            .find_map(pick),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Every capture under `assets/pi/`, each a real session as pi wrote it
    /// (text trimmed and outputs redacted), laid out like `<agent-dir>/sessions`
    /// with its Fabric exports under `.fabric/`, discovered the way a live one
    /// is, and run through the whole conformance check.
    #[test]
    fn captures_conform() {
        let Some(dir) = crate::provider::harness::fixture_dir("pi") else {
            return;
        };
        let mut fixtures: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        fixtures.sort();
        assert!(!fixtures.is_empty(), "no fixtures under {}", dir.display());
        for fixture in fixtures {
            let name = fixture.file_name().unwrap().to_str().unwrap().to_string();
            let roots: Vec<PathBuf> =
                discovery::all_paths_under(&fixture, &crate::provider::Scope::ALL);
            assert_eq!(roots.len(), 1, "{name}: one session per fixture");
            let root = discovery::session_file(&roots[0]).unwrap();
            let children: Vec<PathBuf> = discovery::related_paths(&root)
                .into_iter()
                .filter(|p| {
                    discovery::session_file(p).is_some_and(|f| {
                        f.session == root.session && f.role != crate::provider::FileRole::Root
                    })
                })
                .collect();
            let root_path = root.path.clone();
            let streams = move || -> Vec<Vec<Statement>> {
                let read = |path: &PathBuf, mut stream: Stream| -> Vec<Statement> {
                    let text = std::fs::read_to_string(path).unwrap();
                    text.lines().filter_map(|l| stream.push(l)).collect()
                };
                let mut out = vec![read(&root_path, Stream::new())];
                out.extend(children.iter().map(|c| read(c, Stream::child())));
                out
            };
            crate::provider::harness::conform("pi", &name, streams);
        }
    }

    /// The Fabric fixture: five agents spawned by one call, each with its
    /// export, each ended (one by Fabric's report, the rest by a wait).
    #[test]
    fn fabric_children_are_found_and_end() {
        use crate::provider::{Target, open};
        let Some(dir) = crate::provider::harness::fixture_dir("pi") else {
            return;
        };
        let fixture = dir.join("fabric-spawn");
        let root = discovery::all_paths_under(&fixture, &crate::provider::Scope::ALL).remove(0);
        let s = open(
            &Target::Path(root.clone()),
            Some(crate::provider::Provider::Pi),
        )
        .unwrap();
        assert_eq!(s.id, "01a0fe25-055d-727a-ace8-0e15c862982f");
        assert_eq!(s.files.len(), 5);
        assert!(s.files.iter().all(
            |f| matches!(&f.role, crate::provider::FileRole::Agent { parent } if *parent == s.id)
        ));
        // A child opens its session, provider read off the content.
        let child = s.files[0].path.clone();
        let via_child = open(&Target::Path(child), None).unwrap();
        assert_eq!(via_child.id, s.id);
        assert_eq!(via_child.root.path, root);
        assert_eq!(via_child.files.len(), 5);
    }

    #[test]
    fn spawn_results_and_reports_state_children() {
        let mut s = Stream::new();
        let call = s.push(r#"{"type":"message","id":"a","timestamp":"2026-10-02T19:47:00.000Z","message":{"role":"assistant","content":[{"type":"text","text":"fan out"},{"type":"toolCall","id":"c1","name":"fabric_exec","arguments":{"code":"await agents.spawn({name:'scout'})","display":{"name":"spawn scouts"}}}]}}"#).unwrap();
        assert!(
            call.facts
                .iter()
                .any(|f| matches!(&f.kind, FactKind::Spawn { call } if call == "c1"))
        );
        let res = s.push(r#"{"type":"message","id":"r","timestamp":"2026-10-02T19:47:05.000Z","message":{"role":"toolResult","toolCallId":"c1","toolName":"fabric_exec","isError":false,"details":{"audits":[{"ref":"agents.spawn","success":true,"result":{"id":"r1","name":"scout","status":"running"}},{"ref":"agents.spawn","success":false,"result":null},{"ref":"agents.status","success":true,"result":{"id":"zz","status":"completed"}}]}}}"#).unwrap();
        let spawned: Vec<&Fact> = res
            .facts
            .iter()
            .filter(|f| matches!(f.kind, FactKind::Agent { .. }))
            .collect();
        assert_eq!(spawned.len(), 1);
        assert_eq!(spawned[0].agent.as_deref(), Some("r1"));
        assert!(
            matches!(&spawned[0].kind, FactKind::Agent { kind: AgentKind::Subagent, parent, spawned_by, agent_type, interactive: false, .. } if parent.as_deref() == Some(MAIN_ID) && spawned_by.as_deref() == Some("c1") && agent_type.as_deref() == Some("scout"))
        );
        assert!(
            !res.facts
                .iter()
                .any(|f| matches!(f.kind, FactKind::Ended(_))),
            "a status peek is not an ending"
        );
        // A batched report: only the listed ids, each with its own status.
        let rep = s.push(r#"{"type":"custom_message","customType":"pi-fabric-agent-complete","content":"Unread background agent results (batched).\n\nAgent scout (r1) timed_out after 200.0m:\nquoting Agent other (r9) completed after 1s:\n\nAgent b (r2) failed after 0s:\nno result","display":false,"details":{"ids":["r1","r2"]},"id":"x","timestamp":"2026-10-02T20:00:00.000Z"}"#).unwrap();
        let ended: Vec<(String, AgentStatus)> = rep
            .facts
            .iter()
            .filter_map(|f| match f.kind {
                FactKind::Ended(st) => Some((f.agent.clone().unwrap(), st)),
                _ => None,
            })
            .collect();
        assert_eq!(
            ended,
            [
                ("r1".to_string(), AgentStatus::Failed),
                ("r2".to_string(), AgentStatus::Failed)
            ]
        );
        assert!(
            rep.facts
                .iter()
                .all(|f| f.agent.as_deref() != Some(MAIN_ID)),
            "a report is not main speaking"
        );
        // A single report carries its run in details; a wait observes an end.
        let one = s.push(r#"{"type":"custom_message","customType":"pi-fabric-agent-complete","content":"Fabric agent 3d8ca762 completed after 3.4m: ok","display":true,"details":{"id":"r3","name":"lit","status":"completed"},"timestamp":"2026-10-02T20:01:00.000Z"}"#).unwrap();
        assert!(one.facts.iter().any(|f| f.agent.as_deref() == Some("r3")
            && matches!(f.kind, FactKind::Ended(AgentStatus::Done))));
        let wait = s.push(r#"{"type":"message","id":"w","timestamp":"2026-10-02T20:02:00.000Z","message":{"role":"toolResult","toolCallId":"c2","isError":false,"details":{"audits":[{"ref":"agents.wait","success":true,"result":[{"id":"r4","status":"stopped"},{"id":"r5","status":"running"}]}]}}}"#).unwrap();
        let ended: Vec<&str> = wait
            .facts
            .iter()
            .filter(|f| matches!(f.kind, FactKind::Ended(AgentStatus::Stopped)))
            .filter_map(|f| f.agent.as_deref())
            .collect();
        assert_eq!(ended, ["r4"]);
        assert!(!wait.facts.iter().any(|f| f.agent.as_deref() == Some("r5")));
    }

    #[test]
    fn child_export_is_by_its_run() {
        let st = {
            let mut s = Stream::child();
            [
                r#"{"type":"message","id":"early","timestamp":"2026-10-02T19:47:01.000Z","message":{"role":"assistant","usage":{"output":9}}}"#,
                r#"{"type":"session","version":3,"id":"r1","timestamp":"2026-10-02T19:47:05.033Z","cwd":"/p"}"#,
                r#"{"type":"session_info","id":"info_r1","parentId":null,"timestamp":"2026-10-02T19:47:05.033Z","name":"fabricagent-scout"}"#,
                r#"{"type":"message","id":"m","parentId":"r1","timestamp":"2026-10-02T19:47:09.000Z","message":{"role":"assistant","model":"m","provider":"p","usage":{"output":235}}}"#,
                r#"{"type":"message","id":"u","timestamp":"2026-10-02T19:47:10.000Z","message":{"role":"user","content":"the task"}}"#,
                r#"{"type":"thinking_level_change","id":"t","timestamp":"2026-10-02T19:47:11.000Z","thinkingLevel":"high"}"#,
            ]
            .iter()
            .filter_map(|l| s.push(l))
            .collect::<Vec<_>>()
        };
        assert_eq!(
            st.len(),
            4,
            "nothing before the header; no session rows from a child"
        );
        assert!(
            matches!(&st[0].facts[0].kind, FactKind::Agent { kind: AgentKind::Subagent, parent, .. } if parent.as_deref() == Some(MAIN_ID))
        );
        assert!(
            matches!(&st[1].facts[0].kind, FactKind::Label { agent_type, .. } if agent_type.as_deref() == Some("scout"))
        );
        assert!(
            st.iter()
                .flat_map(|s| &s.facts)
                .all(|f| f.agent.as_deref() == Some("r1"))
        );
        assert!(
            !st.iter()
                .flat_map(|s| &s.facts)
                .any(|f| matches!(f.kind, FactKind::Prompt(_) | FactKind::Title(_)))
        );
        assert!(
            st[2]
                .facts
                .iter()
                .any(|f| matches!(f.kind, FactKind::Tokens { output: 235, .. }))
        );
    }

    #[test]
    fn runs_named_reads_spawns_and_reports_only() {
        let spawn = r#"{"type":"message","message":{"role":"toolResult","toolCallId":"c","details":{"audits":[{"ref":"agents.spawn","success":true,"result":{"id":"r1"}},{"ref":"agents.status","result":{"id":"zz"}}]}}}"#;
        assert_eq!(runs_named(spawn), ["r1"]);
        let report = r#"{"type":"custom_message","customType":"pi-fabric-agent-complete","content":"","details":{"ids":["r2","r3"]}}"#;
        assert_eq!(runs_named(report), ["r2", "r3"]);
        assert!(
            runs_named(r#"{"type":"message","message":{"role":"user","content":"r1"}}"#).is_empty()
        );
    }

    fn push_all(lines: &[&str]) -> Vec<Statement> {
        let mut s = Stream::new();
        lines.iter().filter_map(|l| s.push(l)).collect()
    }

    #[test]
    fn header_states_main_and_session_rows() {
        let st = push_all(&[
            r#"{"type":"session","version":3,"id":"s","timestamp":"2026-09-23T22:11:11.253Z","cwd":"/p","parentSession":"/x/--p--/2026-09-01T00-00-00-000Z_older.jsonl"}"#,
        ]);
        let facts = &st[0].facts;
        assert!(matches!(
            &facts[0].kind,
            FactKind::Agent { kind: AgentKind::Main, interactive: true, agent_type, .. } if agent_type.as_deref() == Some("pi")
        ));
        assert!(facts.iter().any(|f| matches!(&f.kind, FactKind::Session { label, value } if label == "forked from" && value == "older")));
        assert_eq!(facts.iter().filter(|f| f.is_session_meta()).count(), 2);
    }

    #[test]
    fn turn_states_prompt_reasoning_calls_and_outcomes() {
        let st = push_all(&[
            r#"{"type":"session","version":3,"id":"s","timestamp":"2026-09-23T22:11:11.253Z","cwd":"/p"}"#,
            r#"{"type":"message","id":"u","parentId":null,"timestamp":"2026-09-23T22:11:12.000Z","message":{"role":"user","content":[{"type":"text","text":"fix it\n"}]}}"#,
            r#"{"type":"message","id":"a","parentId":"u","timestamp":"2026-09-23T22:11:13.000Z","message":{"role":"assistant","model":"m","usage":{"output":12},"content":[{"type":"thinking","thinking":""},{"type":"thinking","thinking":"look first"},{"type":"toolCall","id":"c1","name":"read","arguments":{"path":"/p/src/a.rs"}},{"type":"toolCall","id":"c2","name":"bash","arguments":{"command":"cargo  test"}}]}}"#,
            r#"{"type":"message","id":"r1","parentId":"a","timestamp":"2026-09-23T22:11:14.000Z","message":{"role":"toolResult","toolCallId":"c1","isError":false}}"#,
            r#"{"type":"message","id":"r2","parentId":"r1","timestamp":"2026-09-23T22:11:15.000Z","message":{"role":"toolResult","toolCallId":"c2","isError":true}}"#,
        ]);
        assert!(matches!(&st[1].facts[0].kind, FactKind::Prompt(p) if p == "fix it"));
        let kinds: Vec<&str> = st[2].facts.iter().map(|f| f.kind.name()).collect();
        assert_eq!(
            kinds,
            ["Model", "Tokens", "Reasoning", "ToolStart", "ToolStart"]
        );
        assert!(
            matches!(&st[2].facts[3].kind, FactKind::ToolStart { summary, .. } if summary.as_deref() == Some("src/a.rs"))
        );
        assert!(
            matches!(&st[2].facts[4].kind, FactKind::ToolStart { summary, .. } if summary.as_deref() == Some("cargo test"))
        );
        assert!(
            matches!(&st[3].facts[0].kind, FactKind::ToolEnd { call, outcome: Outcome::Ok } if call == "c1")
        );
        assert!(
            matches!(&st[4].facts[0].kind, FactKind::ToolEnd { call, outcome: Outcome::Err } if call == "c2")
        );
        assert!(
            st.iter()
                .flat_map(|s| &s.facts)
                .all(|f| f.is_session_meta() || f.agent.as_deref() == Some(MAIN_ID))
        );
        assert_eq!(st[2].at, st[2].facts[0].ts);
    }

    #[test]
    fn user_bash_is_one_whole_tool_run() {
        let st = push_all(&[
            r#"{"type":"message","id":"b1","timestamp":"2026-09-23T22:11:12.000Z","message":{"role":"bashExecution","command":"ls","exitCode":0,"cancelled":false}}"#,
            r#"{"type":"message","id":"b2","timestamp":"2026-09-23T22:11:13.000Z","message":{"role":"bashExecution","command":"sleep 9","cancelled":true}}"#,
        ]);
        // No header: the first activity states main.
        assert!(matches!(st[0].facts[0].kind, FactKind::Agent { .. }));
        assert!(
            matches!(&st[0].facts[2].kind, FactKind::ToolEnd { call, outcome: Outcome::Ok } if call == "b1")
        );
        assert!(matches!(
            &st[1].facts[1].kind,
            FactKind::ToolEnd {
                outcome: Outcome::Err,
                ..
            }
        ));
        assert_eq!(
            st.iter()
                .filter(|s| s
                    .facts
                    .iter()
                    .any(|f| matches!(f.kind, FactKind::Agent { .. })))
                .count(),
            1
        );
    }

    #[test]
    fn bookkeeping_is_metadata_and_extensions_state_nothing() {
        let st = push_all(&[
            r#"{"type":"session","version":3,"id":"s","timestamp":"2026-09-23T22:11:11.253Z","cwd":"/p"}"#,
            r#"{"type":"session_info","id":"i","parentId":null,"timestamp":"2026-09-23T22:11:12.000Z","name":" Refactor auth "}"#,
            r#"{"type":"thinking_level_change","id":"t","parentId":"i","timestamp":"2026-09-23T22:11:12.000Z","thinkingLevel":"high"}"#,
            r#"{"type":"compaction","id":"c","parentId":"t","timestamp":"2026-09-23T22:11:13.000Z","summary":"...","firstKeptEntryId":"t","tokensBefore":5}"#,
            r#"{"type":"custom","customType":"perf_turn","data":{},"id":"x","parentId":"c","timestamp":"2026-09-23T22:11:14.000Z"}"#,
            r#"{"type":"custom_message","customType":"pi-fabric-agent-complete","content":"done","display":true,"id":"y","parentId":"x","timestamp":"2026-09-23T22:11:15.000Z"}"#,
            r#"{"type":"message","id":"z","parentId":"y","timestamp":"2026-09-23T22:11:16.000Z","message":{"role":"system","content":"","sections":{}}}"#,
            r#"{"type":"usage","id":"w","parentId":"z","timestamp":"2026-09-23T22:11:17.000Z","kind":"cache_warm"}"#,
            "garbage",
        ]);
        assert_eq!(st.len(), 4, "header, title, thinking, compaction");
        assert!(st[1..].iter().all(Statement::is_session_meta));
        assert!(matches!(&st[1].facts[0].kind, FactKind::Title(t) if t == "Refactor auth"));
        assert!(matches!(&st[3].facts[0].kind, FactKind::Tally(t) if t == "compactions"));
        assert!(st[1].facts[0].ts.is_none(), "metadata is untimed");
    }

    #[test]
    fn summaries_follow_the_tool_vocabulary() {
        let s = |name: &str, args: Value| summarize_tool(name, &args, Some("/p"));
        use serde_json::json;
        assert_eq!(
            s(
                "fabric_exec",
                json!({"code":"x","display":{"name":"Run tests"}})
            )
            .as_deref(),
            Some("Run tests")
        );
        assert_eq!(
            s("fabric_exec", json!({"code":"x","display":"Shorthand"})).as_deref(),
            Some("Shorthand")
        );
        assert_eq!(
            s("fabric_exec", json!({"code":"\n  const r = 1;\nmore"})).as_deref(),
            Some("const r = 1;")
        );
        assert_eq!(
            s("edit", json!({"path":"/p/src/x.rs","edits":[]})).as_deref(),
            Some("src/x.rs")
        );
        assert_eq!(
            s("web_search", json!({"queries":["pi sessions"]})).as_deref(),
            Some("pi sessions")
        );
        assert_eq!(
            s("fetch_content", json!({"urls":["https://a"]})).as_deref(),
            Some("https://a")
        );
        assert_eq!(
            s("mcp", json!({"connect":"slack"})).as_deref(),
            Some("slack")
        );
        assert_eq!(s("mystery", json!({"query":"q"})).as_deref(), Some("q"));
        assert_eq!(s("mcp", json!({})), None);
        assert_eq!(s("bash", Value::Null), None);
    }
}
