//! pi's wire format: the model for one session-file line, and nothing else.
//! What the records *mean* is the provider's job ([`super`]); where the files
//! *live* is [`super::discovery`].
//!
//! A pi session is one JSONL file. The first line is a header,
//! `{type: "session", version, id, timestamp, cwd, parentSession?}`; every line
//! after it is an entry, `{type, id, parentId, timestamp, ...}`, with a
//! type-specific payload at the top level (not under a `payload` key, unlike
//! Codex). `id`/`parentId` thread the entries into a tree: branching via
//! `/tree` leaves the abandoned branch in the file, and the provider reads it
//! like any other line.
//!
//! The entry that matters is `message`, whose `message` object is an
//! `AgentMessage` keyed by `role`: `user`, `assistant`, `toolResult`,
//! `bashExecution` (a command the user ran with `!`), `system` (a prompt and
//! tool-loadout patch), and whatever extensions add. Assistant content is a
//! list of blocks: `text`, `thinking`, `toolCall`; results carry `isError`.
//! Entry timestamps are ISO-8601; the nested `message.timestamp` is epoch
//! milliseconds and is not read.
//!
//! Two extension shapes are modelled because they carry Fabric's agent runs:
//! a code-mode tool result's `details.audits` (each call the program made,
//! with its result, so `agents.spawn` names the run it started), and the
//! `custom_message` entry Fabric injects when background runs end. A Fabric
//! agent's own export is an ordinary pi file; in the versions seen it holds
//! only a header, a `session_info` name and content-less assistant records
//! (model and usage per turn).
//!
//! Defensive by design: each line is parsed to a JSON value first and every
//! field is read optionally, so an unknown entry type, an unknown role or
//! block, a missing field, a field of the wrong type, or a malformed line
//! parses to something skippable, never a panic and never a lost envelope.

use chrono::{DateTime, Utc};
use serde_json::Value;

/// One parsed line: the envelope every entry shares, plus what it carried.
#[derive(Debug, Clone)]
pub struct Entry {
    /// The entry's own id. The header's is the session id.
    pub id: Option<String>,
    /// The entry this one continues from. `None` for a root entry and for the
    /// header. Not needed to state anything: one file is one thread.
    pub parent_id: Option<String>,
    pub timestamp: Option<DateTime<Utc>>,
    pub body: Body,
}

/// What an entry carried, by its `type`.
#[derive(Debug, Clone)]
pub enum Body {
    /// The first line of the file.
    Session(Header),
    Message(Message),
    /// The user picked another model. Assistant messages name the model that
    /// actually answered, which may differ (a virtual model routes).
    ModelChange {
        model_id: Option<String>,
    },
    ThinkingLevelChange {
        level: Option<String>,
    },
    /// The session's display name, from `/name` or an extension.
    SessionInfo {
        name: Option<String>,
    },
    /// Context was summarised; older entries stay in the file.
    Compaction,
    /// A `/tree` branch switch summarised the abandoned path.
    BranchSummary,
    /// Context an extension injected. Fabric reports its background agents'
    /// ends this way (`pi-fabric-agent-complete`).
    CustomMessage {
        custom_type: String,
        /// The text, whether written as a string or as text blocks.
        content: String,
        /// Extension metadata, not sent to the model. An object; normalised
        /// from a JSON string when written as one.
        details: Value,
    },
    /// `custom` (extension state), `usage`, `context_edit`, `label`, and
    /// anything newer.
    Other(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Header {
    pub id: Option<String>,
    pub version: Option<u64>,
    pub cwd: Option<String>,
    /// The session file this one was forked or cloned from.
    pub parent_session: Option<String>,
}

/// An `AgentMessage`, by `role`.
#[derive(Debug, Clone)]
pub enum Message {
    /// Text a person sent (a typed prompt, or a `/skill` expansion of one).
    User {
        content: Vec<Block>,
    },
    Assistant(Assistant),
    ToolResult {
        tool_call_id: Option<String>,
        /// Absent means success: pi writes `isError: false` explicitly.
        is_error: Option<bool>,
        /// What a code-mode tool (Fabric's `fabric_exec`) called inside the
        /// program, from `details.audits`. Empty for any other tool.
        audits: Vec<Audit>,
    },
    /// A command the user ran from the prompt with `!`.
    BashExecution {
        command: Option<String>,
        /// `None` when the process was killed or cancelled.
        exit_code: Option<i64>,
        cancelled: bool,
    },
    /// `system` and anything newer, by role.
    Other(String),
}

#[derive(Debug, Clone, Default)]
pub struct Assistant {
    pub model: Option<String>,
    /// This response's own output tokens: one message is one API response,
    /// so it is a delta, never a running total.
    pub output_tokens: Option<u64>,
    pub content: Vec<Block>,
}

/// One call a code-mode program made, as Fabric audited it: `agents.spawn`
/// and the run handle it returned, `agents.wait` and the status it saw.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Audit {
    /// `agents.spawn`, `agents.run`, `agents.wait`, `pi.bash`, ...
    pub reference: String,
    /// `false` when the call threw; it returned nothing then.
    pub success: Option<bool>,
    /// The run handles in its result: one for an object, several for an array.
    pub runs: Vec<Run>,
}

/// A Fabric agent run as a result or a report names it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Run {
    pub id: String,
    pub name: Option<String>,
    /// `running`, `completed`, `failed`, `timed_out`, `stopped`, ...
    pub status: Option<String>,
}

/// One content block.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Text(String),
    Thinking(String),
    ToolCall {
        id: Option<String>,
        name: Option<String>,
        /// An object; normalised from a JSON string when a provider sent one.
        arguments: Value,
    },
    /// `image`, and anything newer, by type.
    Other(String),
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Parse one line. `None` for a blank line or one that is not a JSON object
/// with a string `type`.
pub fn parse_line(line: &str) -> Option<Entry> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let v: Value = serde_json::from_str(trimmed).ok()?;
    let kind = v.get("type")?.as_str()?;
    let body = match kind {
        "session" => Body::Session(header(&v)),
        "message" => Body::Message(message(v.get("message").unwrap_or(&Value::Null))),
        "model_change" => Body::ModelChange {
            model_id: str_of(&v, "modelId"),
        },
        "thinking_level_change" => Body::ThinkingLevelChange {
            level: str_of(&v, "thinkingLevel"),
        },
        "session_info" => Body::SessionInfo {
            name: str_of(&v, "name"),
        },
        "compaction" => Body::Compaction,
        "branch_summary" => Body::BranchSummary,
        "custom_message" => Body::CustomMessage {
            custom_type: str_of(&v, "customType").unwrap_or_default(),
            content: text_of(v.get("content")),
            details: object_of(v.get("details")),
        },
        other => Body::Other(other.to_string()),
    };
    Some(Entry {
        id: str_of(&v, "id"),
        parent_id: str_of(&v, "parentId"),
        timestamp: str_of(&v, "timestamp").and_then(|t| parse_ts(&t)),
        body,
    })
}

/// The header, if `line` is one. Discovery's way to classify a file.
pub fn parse_header(line: &str) -> Option<Header> {
    match parse_line(line)?.body {
        Body::Session(h) => Some(h),
        _ => None,
    }
}

fn header(v: &Value) -> Header {
    Header {
        id: str_of(v, "id"),
        version: v.get("version").and_then(Value::as_u64),
        cwd: str_of(v, "cwd"),
        parent_session: str_of(v, "parentSession"),
    }
}

fn message(m: &Value) -> Message {
    let role = m.get("role").and_then(Value::as_str).unwrap_or("");
    match role {
        "user" => Message::User {
            content: blocks(m.get("content")),
        },
        "assistant" => Message::Assistant(Assistant {
            model: str_of(m, "model"),
            output_tokens: m
                .get("usage")
                .and_then(|u| u.get("output"))
                .and_then(Value::as_u64),
            content: blocks(m.get("content")),
        }),
        "toolResult" => Message::ToolResult {
            tool_call_id: str_of(m, "toolCallId"),
            is_error: m.get("isError").and_then(Value::as_bool),
            audits: object_of(m.get("details"))
                .get("audits")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(audit).collect())
                .unwrap_or_default(),
        },
        "bashExecution" => Message::BashExecution {
            command: str_of(m, "command"),
            exit_code: m.get("exitCode").and_then(Value::as_i64),
            cancelled: m.get("cancelled").and_then(Value::as_bool) == Some(true),
        },
        other => Message::Other(other.to_string()),
    }
}

/// A message's content: a string (one text block) or a list of blocks. A
/// usage-only assistant record (written by some extensions) has none.
fn blocks(content: Option<&Value>) -> Vec<Block> {
    match content {
        Some(Value::String(s)) => vec![Block::Text(s.clone())],
        Some(Value::Array(items)) => items.iter().map(block).collect(),
        _ => Vec::new(),
    }
}

fn block(b: &Value) -> Block {
    let kind = b.get("type").and_then(Value::as_str).unwrap_or("");
    match kind {
        "text" => Block::Text(str_of(b, "text").unwrap_or_default()),
        "thinking" => Block::Thinking(str_of(b, "thinking").unwrap_or_default()),
        "toolCall" => Block::ToolCall {
            id: str_of(b, "id"),
            name: str_of(b, "name"),
            arguments: match b.get("arguments") {
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(Value::Null),
                Some(v) => v.clone(),
                None => Value::Null,
            },
        },
        other => Block::Other(other.to_string()),
    }
}

fn audit(a: &Value) -> Option<Audit> {
    Some(Audit {
        reference: str_of(a, "ref")?,
        success: a.get("success").and_then(Value::as_bool),
        runs: match a.get("result") {
            Some(Value::Array(items)) => items.iter().filter_map(run).collect(),
            Some(r) => run(r).into_iter().collect(),
            None => Vec::new(),
        },
    })
}

/// A run handle: an object with a string `id`.
pub fn run(v: &Value) -> Option<Run> {
    Some(Run {
        id: str_of(v, "id").filter(|s| !s.is_empty())?,
        name: str_of(v, "name"),
        status: str_of(v, "status"),
    })
}

/// Text content: a string, or the text of a list of blocks.
fn text_of(content: Option<&Value>) -> String {
    let parts: Vec<String> = blocks(content)
        .into_iter()
        .filter_map(|b| match b {
            Block::Text(t) => Some(t),
            _ => None,
        })
        .collect();
    parts.join("\n")
}

/// An object field, parsed when written as a JSON string. Null otherwise.
fn object_of(v: Option<&Value>) -> Value {
    match v {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(Value::Null),
        Some(v @ Value::Object(_)) => v.clone(),
        _ => Value::Null,
    }
}

fn str_of(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn parse_ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_the_first_line() {
        let h = parse_header(r#"{"type":"session","version":3,"id":"01a0","timestamp":"2026-09-23T22:11:11.253Z","cwd":"/p","parentSession":"/s/x.jsonl"}"#).unwrap();
        assert_eq!(h.id.as_deref(), Some("01a0"));
        assert_eq!(h.version, Some(3));
        assert_eq!(h.cwd.as_deref(), Some("/p"));
        assert_eq!(h.parent_session.as_deref(), Some("/s/x.jsonl"));
        assert!(parse_header(r#"{"type":"message","id":"a"}"#).is_none());
    }

    #[test]
    fn assistant_blocks_in_order() {
        let e = parse_line(r#"{"type":"message","id":"b","parentId":"a","timestamp":"2026-09-23T22:11:13.000Z","message":{"role":"assistant","model":"m","usage":{"output":42},"content":[{"type":"thinking","thinking":"hm","thinkingSignature":"x"},{"type":"text","text":"ok"},{"type":"toolCall","id":"c1","name":"bash","arguments":{"command":"ls"}},{"type":"hologram"}]}}"#).unwrap();
        assert_eq!(e.parent_id.as_deref(), Some("a"));
        assert!(e.timestamp.is_some());
        let Body::Message(Message::Assistant(a)) = e.body else {
            panic!("not assistant");
        };
        assert_eq!(a.model.as_deref(), Some("m"));
        assert_eq!(a.output_tokens, Some(42));
        assert_eq!(a.content.len(), 4);
        assert_eq!(a.content[0], Block::Thinking("hm".into()));
        assert!(
            matches!(&a.content[2], Block::ToolCall { id, arguments, .. } if id.as_deref() == Some("c1") && arguments["command"] == "ls")
        );
        assert_eq!(a.content[3], Block::Other("hologram".into()));
    }

    #[test]
    fn string_arguments_are_normalised() {
        let e = parse_line(r#"{"type":"message","message":{"role":"assistant","content":[{"type":"toolCall","id":"c","name":"read","arguments":"{\"path\":\"/p/a.rs\"}"}]}}"#).unwrap();
        let Body::Message(Message::Assistant(a)) = e.body else {
            panic!("not assistant");
        };
        assert!(
            matches!(&a.content[0], Block::ToolCall { arguments, .. } if arguments["path"] == "/p/a.rs")
        );
    }

    #[test]
    fn results_and_user_bash() {
        let e = parse_line(r#"{"type":"message","message":{"role":"toolResult","toolCallId":"c","toolName":"bash","content":[{"type":"image"}],"isError":true}}"#).unwrap();
        assert!(matches!(
            e.body,
            Body::Message(Message::ToolResult {
                is_error: Some(true),
                ..
            })
        ));
        let e = parse_line(r#"{"type":"message","message":{"role":"bashExecution","command":"ls","output":"","exitCode":null,"cancelled":true}}"#).unwrap();
        assert!(matches!(
            e.body,
            Body::Message(Message::BashExecution {
                exit_code: None,
                cancelled: true,
                ..
            })
        ));
    }

    #[test]
    fn fabric_audits_and_reports() {
        let e = parse_line(r#"{"type":"message","message":{"role":"toolResult","toolCallId":"c","toolName":"fabric_exec","isError":false,"details":{"audits":[{"ref":"agents.spawn","success":true,"result":{"id":"r1","name":"scout","status":"running"}},{"ref":"agents.wait","success":true,"result":[{"id":"r1","status":"completed"},{"no":"id"}]},{"ref":"pi.bash","success":true,"result":"text"},{"noref":1},7]}}}"#).unwrap();
        let Body::Message(Message::ToolResult { audits, .. }) = e.body else {
            panic!("not a result");
        };
        assert_eq!(audits.len(), 3);
        assert_eq!(audits[0].runs[0].name.as_deref(), Some("scout"));
        assert_eq!(audits[1].runs.len(), 1);
        assert!(audits[2].runs.is_empty());
        // Details written as a JSON string are read the same way.
        let e = parse_line(r#"{"type":"custom_message","customType":"pi-fabric-agent-complete","content":[{"type":"text","text":"Agent a (r1) failed after 0s:"}],"details":"{\"ids\":[\"r1\"]}"}"#).unwrap();
        let Body::CustomMessage {
            custom_type,
            content,
            details,
        } = e.body
        else {
            panic!("not a custom message");
        };
        assert_eq!(custom_type, "pi-fabric-agent-complete");
        assert!(content.starts_with("Agent a (r1)"));
        assert_eq!(details["ids"][0], "r1");
    }

    #[test]
    fn unknown_and_malformed_are_skippable_not_fatal() {
        assert!(
            matches!(parse_line(r#"{"type":"custom","customType":"perf_turn","data":{}}"#).unwrap().body, Body::Other(ref k) if k == "custom")
        );
        assert!(
            matches!(parse_line(r#"{"type":"custom_message","content":7,"details":[1]}"#).unwrap().body, Body::CustomMessage { ref content, ref details, .. } if content.is_empty() && details.is_null())
        );
        assert!(
            matches!(parse_line(r#"{"type":"message","message":{"role":"oracle"}}"#).unwrap().body, Body::Message(Message::Other(ref r)) if r == "oracle")
        );
        // A message with no message object, wrong-typed fields, a bad timestamp.
        let e = parse_line(r#"{"type":"message","timestamp":"yesterday","id":7}"#).unwrap();
        assert!(e.timestamp.is_none() && e.id.is_none());
        let e = parse_line(r#"{"type":"message","message":{"role":"assistant","content":7,"usage":{"output":"many"}}}"#).unwrap();
        assert!(
            matches!(e.body, Body::Message(Message::Assistant(ref a)) if a.content.is_empty() && a.output_tokens.is_none())
        );
        assert!(parse_line("not json").is_none());
        assert!(parse_line("[1,2]").is_none());
        assert!(parse_line(r#"{"no":"type"}"#).is_none());
        assert!(parse_line("").is_none());
    }
}
