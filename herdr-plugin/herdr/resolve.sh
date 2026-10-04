#!/usr/bin/env bash
# Print "<agent> <session>" for the pane the plugin was invoked from, where
# <session> is a session id or a transcript path, or exit non-zero with the
# reason on stderr.
#
# The pane id comes from `focused_pane_id` in HERDR_PLUGIN_CONTEXT_JSON, never
# from HERDR_PANE_ID: in a pane command HERDR_PANE_ID is the plugin's own new
# pane, and asking Herdr about that one returns a pane with no agent. The
# context names the pane that was focused when the plugin was invoked, and it
# is the same field for an action and for a pane command.
#
# Herdr's Claude Code and Codex integrations report the native session id from
# a SessionStart hook (`pane.report_agent_session`), so `pane.get` carries
# `agent_session: {source, agent, kind: "id", value}` for both. That pair is
# all zoe needs. Its pi integration reports the session file itself instead,
# `agent_session: {agent: "pi", kind: "path", value: "<absolute .jsonl>"}`,
# which zoe opens the same way. Nothing is guessed from the working directory;
# when Herdr has neither, the caller says so.
set -euo pipefail

herdr="${HERDR_BIN_PATH:-herdr}"
ctx="${HERDR_PLUGIN_CONTEXT_JSON:-{\}}"

command -v jq >/dev/null 2>&1 || { echo "jq is not on PATH, and the plugin reads Herdr's JSON with it" >&2; exit 1; }

pane_id=$(printf '%s' "$ctx" | jq -r '.focused_pane_id // empty')
[ -n "$pane_id" ] || { echo "no focused pane in the invocation context" >&2; exit 1; }

resp=$("$herdr" pane get "$pane_id" 2>&1) || { echo "herdr pane get failed: $resp" >&2; exit 1; }
err=$(printf '%s' "$resp" | jq -r '.error.message // empty' 2>/dev/null || true)
[ -z "$err" ] || { echo "herdr: $err" >&2; exit 1; }

# `pane.get` answers `{"result": {"pane": {...}, "type": "pane_info"}}`: the
# record is under `.result.pane`, not `.result` itself.
pane=$(printf '%s' "$resp" | jq '.result.pane // .result')
agent=$(printf '%s' "$pane" | jq -r '.agent_session.agent // .agent // empty')
kind=$(printf  '%s' "$pane" | jq -r '.agent_session.kind  // empty')
value=$(printf '%s' "$pane" | jq -r '.agent_session.value // empty')

case "$agent" in
  claude | codex | pi) ;;
  "") echo "pane $pane_id has no agent: focus a Claude Code, Codex or pi pane" >&2; exit 1 ;;
  *)  echo "agent '$agent' in pane $pane_id is not one zoe reads (Claude Code, Codex and pi)" >&2; exit 1 ;;
esac

case "$kind" in
  id) ;;
  path)
    case "$value" in
      /*) ;;
      *) echo "Herdr reports a session path for this $agent pane that is not absolute: '$value'" >&2; exit 1 ;;
    esac
    # pi creates the file with the session's first message, not at startup.
    [ -f "$value" ] || { echo "the session file Herdr reports for this $agent pane does not exist yet: $value (send the agent a message first)" >&2; exit 1; } ;;
  "") cat >&2 <<MSG
Herdr has no session for this $agent pane.

  It comes from the agent's session-start hook, which fires only when a
  session begins. So: install the integration if it is missing, then start the
  agent in that pane again. A session that was already running when the
  integration was installed never reports one.

    herdr integration install $agent    (herdr integration status lists them)
MSG
     exit 1 ;;
  *)  echo "Herdr reports a $kind for this $agent pane, and the plugin expects an id or a path" >&2; exit 1 ;;
esac

printf '%s %s\n' "$agent" "$value"
