#!/bin/bash
# ============================================================
# run_multi_miner.sh — manage multiple Poker44 miners (pm2)
#
# Each miner entry needs a unique:
#   HOTKEY      — bittensor hotkey name (under the same coldkey)
#   AXON_PORT   — unique port per miner (default 8091+)
#   PM2_NAME    — unique pm2 process name
#
# Usage:
#   ./scripts/miner/run/run_multi_miner.sh start    # start all miners
#   ./scripts/miner/run/run_multi_miner.sh stop     # stop all miners
#   ./scripts/miner/run/run_multi_miner.sh restart  # restart all miners
#   ./scripts/miner/run/run_multi_miner.sh status   # show pm2 status
#   ./scripts/miner/run/run_multi_miner.sh logs <name>  # tail logs for one miner
# ============================================================

set -e

# ----------------------------------------------------------------
# Shared settings (same for all miners)
# ----------------------------------------------------------------
WALLET_NAME="${WALLET_NAME:-superbit-darnsin}"   # coldkey (shared)
NETWORK="${NETWORK:-finney}"
NETUID="${NETUID:-126}"
MINER_SCRIPT="./neurons/miner.py"

# Allowlisted validator hotkeys (space-separated).
# If set: miners accept only these hotkeys (allowlist mode).
# If empty: miners fall back to --blacklist.force_validator_permit (on-chain permit check).
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD 5D9j5f7RV9hfK2aGVxspruj3e4eL1hc5XepUQqZTXEua62BJ}"

# ----------------------------------------------------------------
# Miner definitions — edit this section for your hotkeys/ports
# Each row: "HOTKEY AXON_PORT PM2_NAME"
# ----------------------------------------------------------------
MINERS=(
  "poker-miner-26001 8091 poker44_miner_1"
  "poker-miner-26002 8092 poker44_miner_2"
  "poker-miner-26003 8093 poker44_miner_3"
  "poker-miner-26004 8094 poker44_miner_4"
  "poker-miner-26005 8095 poker44_miner_5"
  "poker-miner-26006 8096 poker44_miner_6"
  "poker-miner-26007 8097 poker44_miner_7"
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

# Build miner args — mirrors the access-mode logic from run_miner.sh:
#   ALLOWED_VALIDATOR_HOTKEYS set   → allowlist mode
#   ALLOWED_VALIDATOR_HOTKEYS empty → force_validator_permit mode
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

start_all() {
  require_pm2
  export PYTHONPATH="$(pwd)"
  for entry in "${MINERS[@]}"; do
    read -r hotkey port pm2_name <<< "$entry"
    echo "Starting miner: pm2=$pm2_name hotkey=$hotkey port=$port"
    pm2 delete "$pm2_name" 2>/dev/null || true
    # shellcheck disable=SC2046
    pm2 start "$MINER_SCRIPT" \
      --name "$pm2_name" -- \
      $(miner_args "$hotkey" "$port")
  done
  pm2 save
  echo ""
  echo "All miners started. Check status: pm2 list"
  if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
    echo "Access mode: validator allowlist ($ALLOWED_VALIDATOR_HOTKEYS)"
  else
    echo "Access mode: force_validator_permit (on-chain permit check)"
  fi
}

stop_all() {
  require_pm2
  for entry in "${MINERS[@]}"; do
    read -r hotkey port pm2_name <<< "$entry"
    echo "Stopping: $pm2_name"
    pm2 stop "$pm2_name" 2>/dev/null || echo "  (already stopped)"
  done
}

restart_all() {
  require_pm2
  for entry in "${MINERS[@]}"; do
    read -r hotkey port pm2_name <<< "$entry"
    echo "Restarting: $pm2_name"
    pm2 restart "$pm2_name" 2>/dev/null || echo "  (not running, starting instead)"
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
      read -r hotkey port pm2_name <<< "$entry"
      echo "  $pm2_name (hotkey=$hotkey port=$port)"
    done
    exit 1
  fi
  pm2 logs "$name"
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
  *)
    echo "Usage: $0 {start|stop|restart|status|logs <pm2_name>}"
    echo ""
    echo "Configured miners:"
    for entry in "${MINERS[@]}"; do
      read -r hotkey port pm2_name <<< "$entry"
      echo "  pm2=$pm2_name  hotkey=$hotkey  port=$port"
    done
    exit 1
    ;;
esac
