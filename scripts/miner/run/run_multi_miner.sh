#!/bin/bash
# ============================================================
# run_multi_miner.sh — manage multiple Poker44 miners (pm2)
#
# Each miner entry:  "WALLET  HOTKEY  PORT  PM2_NAME  [MODEL_VERSION]"
#   WALLET          — coldkey wallet name; use "-" to fall back to DEFAULT_WALLET_NAME
#   HOTKEY          — bittensor hotkey name registered under that wallet
#   PORT            — unique axon port per miner
#   PM2_NAME        — unique pm2 process name
#   MODEL_VERSION   — (optional) model subfolder; "-" → DEFAULT_MODEL_VERSION
#
# Multi-wallet example:
#   "wallet-A  poker-miner-001  8091  miner_1   v4_rf_mixed"
#   "wallet-B  poker-miner-002  8092  miner_2   v4_rf_mixed"
#   "-         poker-miner-003  8093  miner_3   -"          ← uses defaults
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
# Shared defaults — used when a miner entry has "-" in that field
# ----------------------------------------------------------------
DEFAULT_WALLET_NAME="${DEFAULT_WALLET_NAME:-superbit-darnsin}"
NETWORK="${NETWORK:-finney}"
NETUID="${NETUID:-126}"
MINER_SCRIPT="./neurons/miner.py"

# Python interpreter — set to venv python so PM2 uses the right env
PYTHON="${PYTHON:-$(which python3)}"

# Default model version — used when a miner entry has no 5th field or "-"
DEFAULT_MODEL_VERSION="${DEFAULT_MODEL_VERSION:-v4_rf_mixed}"

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
  5C8R8ifnxswxhSsRiRhkriRAThdryCpkP6ScZXUotJhsuNZD
)
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-${_DEFAULT_VALIDATOR_HOTKEYS[*]}}"

# ----------------------------------------------------------------
# Miner definitions
# Format: "WALLET  HOTKEY  PORT  PM2_NAME  [MODEL_VERSION]"
# Use "-" for WALLET or MODEL_VERSION to fall back to the defaults above.
# ----------------------------------------------------------------
MINERS=(
  # wallet-name           hotkey               port  pm2-name            model-version
  "superbit-darnsin  poker-miner-26002  8092  poker44_miner_2   v4_rf_mixed"
  "superbit-darnsin  poker-miner-26003  8093  poker44_miner_3   v4_rf_mixed"
  "superbit-darnsin  poker-miner-26004  8094  poker44_miner_4   v4_rf_mixed"
  "superbit-darnsin  poker-miner-26005  8095  poker44_miner_5   v4_rf_mixed"
  "superbit-darnsin  poker-miner-26006  8096  poker44_miner_6   v4_rf_mixed"
  "superbit-darnsin  poker-miner-26007  8097  poker44_miner_7   v4_rf_mixed"
  # Example: different coldkey
  # "my-other-wallet  poker-miner-new01  8101  poker44_miner_n1  v4_rf_mixed"
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
# Sets: _WALLET, _HOTKEY, _PORT, _PM2_NAME, _MODEL_VER
parse_miner_entry() {
  _WALLET=""
  _HOTKEY=""
  _PORT=""
  _PM2_NAME=""
  _MODEL_VER=""
  read -r _WALLET _HOTKEY _PORT _PM2_NAME _MODEL_VER <<< "$1"
  # "-" or empty → fall back to defaults
  if [ -z "$_WALLET" ] || [ "$_WALLET" = "-" ]; then
    _WALLET="$DEFAULT_WALLET_NAME"
  fi
  if [ -z "$_MODEL_VER" ] || [ "$_MODEL_VER" = "-" ]; then
    _MODEL_VER="$DEFAULT_MODEL_VERSION"
  fi
}

# Build bittensor CLI args — allowlist mode or force_validator_permit fallback
miner_args() {
  local wallet="$1"
  local hotkey="$2"
  local port="$3"
  local args="--netuid $NETUID --wallet.name $wallet --wallet.hotkey $hotkey \
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
    echo "Starting miner: pm2=$_PM2_NAME  wallet=$_WALLET  hotkey=$_HOTKEY  port=$_PORT  model=$_MODEL_VER"
    pm2 delete "$_PM2_NAME" 2>/dev/null || true

    # shellcheck disable=SC2046
    MODEL_VERSION="$_MODEL_VER" \
    POKER44_MODEL_VERSION="$_MODEL_VER" \
    POKER44_MODEL_NAME="poker44-rf-bot-detector" \
    POKER44_MODEL_REPO_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo '')" \
    pm2 start "$MINER_SCRIPT" \
      --name "$_PM2_NAME" \
      --interpreter "$PYTHON" \
      -- $(miner_args "$_WALLET" "$_HOTKEY" "$_PORT")
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
    echo "Restarting: $_PM2_NAME  (wallet=$_WALLET  model=$_MODEL_VER)"
    # --update-env refreshes MODEL_VERSION in pm2's stored process env
    MODEL_VERSION="$_MODEL_VER" \
    POKER44_MODEL_VERSION="$_MODEL_VER" \
    POKER44_MODEL_NAME="poker44-rf-bot-detector" \
    POKER44_MODEL_REPO_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo '')" \
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
      echo "  $_PM2_NAME  (wallet=$_WALLET  hotkey=$_HOTKEY  port=$_PORT  model=$_MODEL_VER)"
    done
    exit 1
  fi
  pm2 logs "$name"
}

list_miners() {
  echo "Configured miners:"
  echo ""
  printf "  %-22s %-28s %-6s %-26s %s\n" "WALLET" "PM2_NAME" "PORT" "HOTKEY" "MODEL_VERSION"
  printf "  %-22s %-28s %-6s %-26s %s\n" "------" "--------" "----" "------" "-------------"
  for entry in "${MINERS[@]}"; do
    parse_miner_entry "$entry"
    printf "  %-22s %-28s %-6s %-26s %s\n" "$_WALLET" "$_PM2_NAME" "$_PORT" "$_HOTKEY" "$_MODEL_VER"
  done
  echo ""
  echo "Default wallet: $DEFAULT_WALLET_NAME   Default model: $DEFAULT_MODEL_VERSION"
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
