#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/notebook_workflow.sh verify <notebook_path>
  scripts/notebook_workflow.sh commit <notebook_path> "<commit_message>"
  scripts/notebook_workflow.sh status <notebook_path>"
USAGE
}

syntax_check() {
  local nb_path="$1"

  python - "$nb_path" <<'PY'
import json
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])

if not path.exists():
    print(f"ERROR: notebook not found: {path}")
    raise SystemExit(1)

nb = json.loads(path.read_text())
errors = []

for i, cell in enumerate(nb.get("cells", []), start=1):
    if cell.get("cell_type") != "code":
        continue
    lines = cell.get("source", [])
    filtered = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        filtered.append(line)
    src = "".join(filtered).strip()
    if not src:
        continue
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append((i, e.lineno, e.offset, e.msg))

if errors:
    print("SYNTAX_ERRORS")
    for err in errors:
        print(err)
    raise SystemExit(1)
else:
    print("OK")
PY
}

verify_notebook() {
  local nb_path="$1"

  echo
  echo "1. Notebook inventory"
  ls -l "$nb_path"

  echo
  echo "2. Notebook-aware syntax check"
  syntax_check "$nb_path"

  echo
  echo "3. Execution check"
  jupyter nbconvert --to notebook --execute --inplace "$nb_path"

  echo
  echo "4. Git footprint"
  git status --short
  git diff --stat -- "$nb_path" || true
}

commit_notebook() {
  local nb_path="$1"
  local msg="$2"

  git add "$nb_path"
  git commit -m "$msg"
}

show_status() {
  local nb_path="$1"
  git status --short
  git diff --stat -- "$nb_path" || true
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

cmd="$1"
nb_path="$2"

case "$cmd" in
  verify)
    verify_notebook "$nb_path"
    ;;
  commit)
    if [[ $# -ne 3 ]]; then
      usage
      exit 1
    fi
    commit_notebook "$nb_path" "$3"
    ;;
  status)
    show_status "$nb_path"
    ;;
  *)
    usage
    exit 1
    ;;
esac
