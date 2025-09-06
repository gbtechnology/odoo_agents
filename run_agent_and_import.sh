#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  . "$PROJECT_ROOT/.env"
  set +a
fi

# --- Choose Python (prefer local venv): make sure your python venv folder is within project's folder, ---
# --- else adjust this section accordingly. --
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  VENVPY="$PROJECT_ROOT/.venv/bin/python"
elif [ -x "$PROJECT_ROOT/venv/bin/python" ]; then
  VENVPY="$PROJECT_ROOT/venv/bin/python"
else
  VENVPY="$(command -v python3)"
fi

# --- Logging ---
LOGDIR="$PROJECT_ROOT/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/runner.log"

ts() { date +"%Y-%m-%d %H:%M:%S%z"; }

# --- Nice header for each run ---
{
  echo "[$(ts)] ================================================"
  echo "[$(ts)] START cycle in $PROJECT_ROOT"
  echo "[$(ts)] Using Python: $VENVPY"
  "$VENVPY" -V || true

  # 1) Generate drafts with the AI agent
  echo "[$(ts)] Running: odoo_blog_agent.py"
  "$VENVPY" "$PROJECT_ROOT/odoo_blog_agent.py"

  # 2) Import drafts into Odoo
  echo "[$(ts)] Running: blog_import.py"
  "$VENVPY" "$PROJECT_ROOT/blog_import.py"

  echo "[$(ts)] END cycle"
} | tee -a "$LOGFILE"
