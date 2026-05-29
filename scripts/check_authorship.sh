#!/usr/bin/env bash
# Reject non-conforming author metadata and reserved markers before they enter
# the repository. Scans content for a set of disallowed tokens and fails if any
# are present. Usage:
#   check_authorship.sh                scan the staged diff (git diff --cached)
#   check_authorship.sh --stdin        scan standard input
#   check_authorship.sh --range A B    scan git diff A B
set -euo pipefail

# Disallowed tokens. Each is written with one bracketed character so this file
# does not itself contain the literal strings it rejects.
pattern='cl[a]ude|anthropi[c]|\bassistan[t]\b|ai-generate[d]|llm-generate[d]|chatgp[t]|\bgp[t]\b|copilo[t]|language mode[l]|machine-authore[d]|generated wit[h]|co-authored-b[y]'

case "${1:-}" in
  --stdin) content="$(cat)" ;;
  --range) content="$(git diff "${2:?range start required}" "${3:?range end required}")" ;;
  "")      content="$(git diff --cached)" ;;
  *)       content="$(git diff "$@")" ;;
esac

if printf '%s\n' "$content" | grep -inE "$pattern"; then
  echo "authorship check failed: a disallowed token is present" >&2
  exit 1
fi
echo "authorship check passed"
