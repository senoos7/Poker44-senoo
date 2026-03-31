#!/bin/bash
# ============================================================
# run_multi_miner.sh — manage multiple Poker44 miners (pm2)
#
# Each miner entry needs a unique:
#   HOTKEY          — bittensor hotkey name (under the same coldkey)
#   AXON_PORT       — unique port per miner (default 8091+)
#   PM2_NAME        — unique pm2 process name
#   MODEL_VERSION   — (optional) which model to load
#                     Must match a folder under poker44/miner_model/models/
#                     Leave blank or "-" to use DEFAULT_MODEL_VERSION
#
# A/B testing example:
#   "poker-miner-26001 8091 poker44_miner_1  v1_rf_synthetic"   ← control
#   "poker-miner-26002 8092 poker44_miner_2  v3_gb_mixed"       ← test
#
# Usage:
#   ./scripts/miner/run/run_multi_miner.sh start
#   ./scripts/miner/run/run_multi_miner.sh stop
#   ./scripts/miner/run/run_multi_miner.sh restart
#   ./scripts/miner/run/run_multi_miner.sh status
#   ./scripts/miner/run/run_multi_miner.sh logs <pm2_name>
#   ./scripts/miner/run/run_multi_miner.sh list
# ============================================================

set -e

# ----------------------------------------------------------------
# Shared settings (same for all miners)
# ----------------------------------------------------------------
WALLET_NAME="${WALLET_NAME:-superbit-darnsin}"   # coldkey (shared)
NETWORK="${NETWORK:-finney}"
NETUID="${NETUID:-126}"
MINER_SCRIPT="./neurons/miner.py"

# Python interpreter — set to venv python so PM2 uses the right env
PYTHON="${PYTHON:-$(which python3)}"

# Default model version — used when a miner entry has no 4th field.
# Set to "" to use the legacy model.pkl default (MODEL_VERSION unset).
DEFAULT_MODEL_VERSION="${DEFAULT_MODEL_VERSION:-v3_gb_mixed}"

# Allowlisted validator hotkeys (space-separated).
_DEFAULT_VALIDATOR_HOTKEYS=(
  5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u
  5FxQcdsCXcNjWowQ63Y2oeMhN3JRQksejV3aHRr4XmtknM2k
  5EP9fmtknrTnDhQmLRY9ciFYoM7YZM8rPWvQ9J7yywEsn126
  5HWe7T96SrY4vRvaLmSoriUJ2CGvhRc559U1vZ1pNPuyz2VA
  5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp
  5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD
  5G9hfkx9wGB1CLMT9WXkpHSAiYzjZb5o1Boyq4KAdDhjwrc5
  5HmkWGB5PVzKCNLB4QxWWHFVEHPAbKKxGyoXW7Evs38gs126
  5D9j5f7RV9hfK2aGVxspruj3e4eL1hc5XepUQqZTXEua62BJ
)
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-${_DEFAULT_VALIDATOR_HOTKEYS[*]}}"

# ----------------------------------------------------------------
# Miner definitions — edit this section for your hotkeys/ports
# Format: "HOTKEY AXON_PORT PM2_NAME [MODEL_VERSION]"
# MODEL_VERSION is optional — omit or use "-" to use DEFAULT_MODEL_VERSION
# ----------------------------------------------------------------
MINERS=(
  "poker-miner-26001 8091 poker44_miner_1  v3_gb_mixed"
  "poker-miner-26002 8092 poker44_miner_2  v3_gb_mixed"
  "poker-miner-26003 8093 poker44_miner_3  v3_gb_mixed"
  "poker-miner-26004 8094 poker44_miner_4  v3_gb_mixed"
  # "poker-miner-26011 8101 poker44_miner_11"
  # "poker-miner-26012 8102 poker44_miner_12"
  # "poker-miner-26013 8103 poker44_miner_13"
  # "poker-miner-26014 8104 poker44_miner_14"
  # "poker-miner-26016 8106 poker44_miner_16"
  # "poker-miner-26017 8107 poker44_miner_17"
  # "poker-miner-26018 8108 poker44_miner_18"
  "poker-miner-26019 8109 poker44_miner_19 v3_gb_mixed"
  # "poker-miner-26020 8110 poker44_miner_20"
  "poker-miner-26021 8111 poker44_miner_21 v3_gb_mixed"
)

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
require_pm2() {
  if ! command -v pm2 &>/dev/null; then
    echo "Error: pm2 is not installed. Run: npm install -g pm2"
    exit 1
  fi
}

# Parse a miner entry into named variables.
# Sets: _HOTKEY, _PORT, _PM2_NAME, _MODEL_VER
parse_miner_entry() {
  _HOTKEY=""
  _PORT=""
  _PM2_NAME=""
  _MODEL_VER=""
  read -r _HOTKEY _PORT _PM2_NAME _MODEL_VER <<< "$1"
  # If MODEL_VER is empty or "-", use the default
  if [ -z "$_MODEL_VER" ] || [ "$_MODEL_VER" = "-" ]; then
    _MODEL_VER="$DEFAULT_MODEL_VERSION"
  fi
}

# Build bittensor CLI args — allowlist mode or force_validator_permit fallback
miner_args() {
  local hotkey="$1"
  local port="$2"
  local args="--netuid $NETUID --wallet.name $WALLET_NAME --wallet.hotkey $hotkey \
        --subtensor.network $NETWORK --axon.port $port --logging.debug"

  if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
    # shellcheck disable=SC2206
    read -r -a HOTKEY_ARRAY <<< "$ALLOWED_VALIDATOR_HOTKEYS"
    args="$args --blacklist.allowed_validator_hotkeys ${HOTKEY_ARRAY[*]}"
  else
    args="$args --blacklist.force_validator_permit"
  fi

  echo "$args"
}

# ----------------------------------------------------------------
# Commands
# ----------------------------------------------------------------
start_all() {
  require_pm2
  export PYTHONPATH="$(pwd)"
  for entry in "${MINERS[@]}"; do
    parse_miner_entry "$entry"
    echo "Starting miner: pm2=$_PM2_NAME  hotkey=$_HOTKEY  port=$_PORT  model=$_MODEL_VER"
    pm2 delete "$_PM2_NAME" 2>/dev/null || true

    # Pass MODEL_VERSION into the PM2 process environment so it persists
    # through restarts (PM2 snapshots the env at process creation time).
    # shellcheck disable=SC2046
    MODEL_VERSION="$_MODEL_VER" \
    pm2 start "$MINER_SCRIPT" \
      --name "$_PM2_NAME" \
      --interpreter "$PYTHON" \
      -- $(miner_args "$_HOTKEY" "$_PORT")
  done
  pm2 save
  echo ""
  echo "All miners started. Check status: pm2 list"
  if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
    echo "Access mode: validator allowlist"
  else
    echo "Access mode: force_validator_permit (on-chain permit check)"
  fi
}

stop_all() {
  require_pm2
  for entry in "${MINERS[@]}"; do
    parse_miner_entry "$entry"
    echo "Stopping: $_PM2_NAME"
    pm2 stop "$_PM2_NAME" 2>/dev/null || echo "  (already stopped)"
  done
}

restart_all() {
  require_pm2
  for entry in "${MINERS[@]}"; do
    parse_miner_entry "$entry"
    echo "Restarting: $_PM2_NAME  (model=$_MODEL_VER)"
    # --update-env refreshes MODEL_VERSION in pm2's stored process env
    MODEL_VERSION="$_MODEL_VER" \
    pm2 restart "$_PM2_NAME" --update-env 2>/dev/null \
      || echo "  ($_PM2_NAME not found — run 'start' first)"
  done
}

status_all() {
  require_pm2
  pm2 list
}

logs_one() {
  local name="$1"
  if [ -z "$name" ]; then
    echo "Usage: $0 logs <pm2_name>"
    echo "Available miners:"
    for entry in "${MINERS[@]}"; do
      parse_miner_entry "$entry"
      echo "  $_PM2_NAME  (hotkey=$_HOTKEY  port=$_PORT  model=$_MODEL_VER)"
    done
    exit 1
  fi
  pm2 logs "$name"
}

list_miners() {
  echo "Configured miners:"
  echo ""
  printf "  %-28s %-6s %-26s %s\n" "PM2_NAME" "PORT" "HOTKEY" "MODEL_VERSION"
  printf "  %-28s %-6s %-26s %s\n" "--------" "----" "------" "-------------"
  for entry in "${MINERS[@]}"; do
    parse_miner_entry "$entry"
    printf "  %-28s %-6s %-26s %s\n" "$_PM2_NAME" "$_PORT" "$_HOTKEY" "$_MODEL_VER"
  done
  echo ""
  echo "Default MODEL_VERSION: $DEFAULT_MODEL_VERSION"
}

# ----------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------
case "${1:-}" in
  start)    start_all ;;
  stop)     stop_all ;;
  restart)  restart_all ;;
  status)   status_all ;;
  logs)     logs_one "${2:-}" ;;
  list)     list_miners ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs <pm2_name>|list}"
    echo ""
    list_miners
    exit 1
    ;;
esac
