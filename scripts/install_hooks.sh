#!/usr/bin/env bash
# Install local git hooks. Wires a pre-push hook that runs the authorship check
# over the commits being pushed.
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
hook="$root/.git/hooks/pre-push"

cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail
zero="0000000000000000000000000000000000000000"
root="$(git rev-parse --show-toplevel)"
status=0
while read -r local_ref local_sha remote_ref remote_sha; do
  [ "$local_sha" = "$zero" ] && continue
  if [ "$remote_sha" = "$zero" ]; then
    diff="$(git log -p --no-color "$local_sha" --)"
  else
    diff="$(git diff --no-color "$remote_sha" "$local_sha")"
  fi
  if ! printf '%s\n' "$diff" | "$root/scripts/check_authorship.sh" --stdin; then
    status=1
  fi
done
exit "$status"
HOOK

chmod +x "$hook"
echo "installed pre-push hook at $hook"
