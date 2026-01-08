#!/bin/bash

# causes script to exit on any unexpected error
set -e

scheduler_log_file="scheduler.log"

# Define cleanup (for both startup and trap)
cleanup() {
    echo "🧹 Cleaning up any leftover background processes..."
    # Get current script's PID
    current_pid=$$
    # Kill old gunicorn processes (except this one)
    pgrep -f "gunicorn" | grep -v "$current_pid" | xargs kill -9 2>/dev/null || true
    # Kill old scheduler processes (except this one)
    pgrep -f "scheduler.py" | grep -v "$current_pid" | xargs kill -9 2>/dev/null || true
    # Kill old tail processes
    pgrep -f "tail -F $scheduler_log_file" | grep -v "$current_pid" | xargs kill -9 2>/dev/null || true
    # Avoid killing the current startup.sh script itself
    pgrep -f "startup.sh" | grep -v "$current_pid" | xargs kill -9 2>/dev/null || true
}
# Register cleanup for script exit/interruption
trap cleanup EXIT INT TERM


# Start the scheduler
echo "🔁 Starting scheduler in background..."
RUN_APP_SCHEDULER=true nohup python -u scheduler.py > $scheduler_log_file 2>&1 &

SEARCH_STRING="Scheduler is running"

## Wait for search string in logs
if command -v stdbuf >/dev/null 2>&1; then
    { tail -F $scheduler_log_file & tail_pid=$!; stdbuf -oL grep -m 1 "$SEARCH_STRING" <(tail -F $scheduler_log_file); kill $tail_pid; }
elif command -v unbuffer >/dev/null 2>&1; then
    { tail -F $scheduler_log_file & tail_pid=$!; unbuffer grep -m 1 "$SEARCH_STRING" <(tail -F $scheduler_log_file); kill $tail_pid; }
else
    brew install unbuffer
    { tail -F $scheduler_log_file & tail_pid=$!; unbuffer grep -m 1 "$SEARCH_STRING" <(tail -F $scheduler_log_file); kill $tail_pid; }
fi

echo && echo "🦄 Starting gunicorn web server... 🚀"
gunicorn main:app \
  --bind=:3000 \
  --worker-class=gthread \
  --workers=3 \
  --threads=5 \
  --timeout=120 \
  --access-logfile=- \
  --error-logfile=- \
  --log-level=debug
#  --preload  (required when not working with a scheduler in a multi worker setup, to load the config once on master before repeatedly loading the same again on all Gunicorn workers)
