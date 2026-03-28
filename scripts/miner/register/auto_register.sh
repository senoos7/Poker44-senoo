#!/bin/bash
# ============================================================
# auto_register.sh — keep retrying btcli register until success
#
# Works with any Bittensor subnet.
#
# Usage:
#   ./auto_register.sh --netuid 126 --wallet MY_COLDKEY --hotkey MY_HOTKEY
#   ./auto_register.sh --netuid 18  --wallet MY_COLDKEY --hotkey MY_HOTKEY --network finney
#
# All flags:
#   --netuid    <int>     Subnet UID (required)
#   --wallet    <name>    Coldkey wallet name (required)
#   --hotkey    <name>    Hotkey name (required)
#   --network   <name>    Network: finney | test | local  (default: finney)
#   --password  <pass>    Wallet password (omit to be prompted securely)
#   --block-time <secs>   Seconds per block (default: 12)
# ============================================================

set -euo pipefail

# ── Colours ─────────────────────────────────────────────────
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"
CYAN="\033[36m";  BOLD="\033[1m";    RESET="\033[0m"

# ── Defaults ─────────────────────────────────────────────────
NETUID=""
WALLET_NAME=""
HOTKEY=""
NETWORK="finney"
WALLET_PASSWORD=""
BLOCK_TIME_SECS=12

# ── Argument parsing ─────────────────────────────────────────
usage() {
    echo "Usage: $0 --netuid <uid> --wallet <name> --hotkey <name> [options]"
    echo ""
    echo "  --netuid    <int>   Subnet UID (required)"
    echo "  --wallet    <name>  Coldkey wallet name (required)"
    echo "  --hotkey    <name>  Hotkey name (required)"
    echo "  --network   <name>  finney | test | local  (default: finney)"
    echo "  --password  <pass>  Wallet password (prompted if omitted)"
    echo "  --block-time <sec>  Seconds per block (default: 12)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --netuid)     NETUID="$2";          shift 2 ;;
        --wallet)     WALLET_NAME="$2";     shift 2 ;;
        --hotkey)     HOTKEY="$2";          shift 2 ;;
        --network)    NETWORK="$2";         shift 2 ;;
        --password)   WALLET_PASSWORD="$2"; shift 2 ;;
        --block-time) BLOCK_TIME_SECS="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ── Validate required args ───────────────────────────────────
if [[ -z "$NETUID" || -z "$WALLET_NAME" || -z "$HOTKEY" ]]; then
    echo -e "${RED}Error: --netuid, --wallet, and --hotkey are required.${RESET}"
    echo ""
    usage
fi

echo -e "${BOLD}================================================${RESET}"
echo -e "${BOLD}  Bittensor Auto-Registration${RESET}"
echo -e "${BOLD}================================================${RESET}"
echo -e "  netuid     : ${NETUID}"
echo -e "  coldkey    : ${WALLET_NAME}"
echo -e "  hotkey     : ${HOTKEY}"
echo -e "  network    : ${NETWORK}"
echo -e "  block time : ${BLOCK_TIME_SECS}s"
echo ""

# ── Check expect is available ─────────────────────────────────
if ! command -v expect &>/dev/null; then
    echo -e "${YELLOW}  'expect' not found. Installing...${RESET}"
    sudo apt-get install -y expect 2>/dev/null || \
    sudo yum install -y expect 2>/dev/null || {
        echo -e "${RED}  Could not install expect. Run: sudo apt install expect${RESET}"
        exit 1
    }
fi

# ── Password ─────────────────────────────────────────────────
if [[ -z "$WALLET_PASSWORD" ]]; then
    read -r -s -p "Wallet password: " WALLET_PASSWORD
    echo ""
fi
echo ""

# ── Registration loop ────────────────────────────────────────
attempt=0

while true; do
    attempt=$((attempt + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo -e "${CYAN}[${timestamp}] Attempt #${attempt} — registering ${HOTKEY} on netuid ${NETUID}...${RESET}"

    # Use expect to handle btcli's interactive prompts:
    #   "Do you want to continue? [y/n]" → y
    #   "Enter your password:"           → WALLET_PASSWORD
    output=$(expect -c "
        log_user 1
        set timeout 90
        spawn btcli subnet register \
            --netuid ${NETUID} \
            --wallet.name ${WALLET_NAME} \
            --wallet.hotkey ${HOTKEY} \
            --subtensor.network ${NETWORK}
        expect {
            -re {Do you want to continue\?.*} { send \"y\r\"; exp_continue }
            -re {Enter your password.*}        { send \"${WALLET_PASSWORD}\r\"; exp_continue }
            -re {Decrypting}                   { exp_continue }
            timeout                            { exit 3 }
            eof
        }
    " 2>&1) || true

    echo "${output}"

    # ── Success ──────────────────────────────────────────────
    if echo "${output}" | grep -qiE "registered|successfully|is already registered"; then
        echo ""
        echo -e "${GREEN}${BOLD}✓ Registration successful!${RESET}"
        echo -e "${GREEN}  netuid=${NETUID}  wallet=${WALLET_NAME}  hotkey=${HOTKEY}${RESET}"
        exit 0
    fi

    # ── Interval full — parse block count and wait ───────────
    if echo "${output}" | grep -qiE "full for this interval|Try again in"; then
        blocks=$(echo "${output}" | grep -oP 'Try again in \K[0-9]+' || echo "")
        if [[ -n "${blocks}" ]]; then
            wait_secs=$(( blocks * BLOCK_TIME_SECS ))
            wait_min=$(( wait_secs / 60 ))
            wait_sec=$(( wait_secs % 60 ))
            echo -e "${YELLOW}  Subnet full — next slot in ${blocks} blocks (~${wait_min}m ${wait_sec}s).${RESET}"
            remaining=${wait_secs}
            while (( remaining > 0 )); do
                echo -e "  ⏳ ${YELLOW}$(( remaining / 60 ))m $(( remaining % 60 ))s remaining...${RESET}"
                sleep_chunk=$(( remaining < 30 ? remaining : 30 ))
                sleep "${sleep_chunk}"
                remaining=$(( remaining - sleep_chunk ))
            done
        else
            echo -e "${YELLOW}  Subnet full — waiting 60s...${RESET}"
            sleep 60
        fi
        continue
    fi

    # ── Custom error 6 (slot taken this interval) ────────────
    if echo "${output}" | grep -qiE "Custom error: 6|InvalidTransaction"; then
        echo -e "${YELLOW}  Slot taken (custom error 6) — waiting 60s...${RESET}"
        sleep 60
        continue
    fi

    # ── Insufficient balance ─────────────────────────────────
    if echo "${output}" | grep -qiE "insufficient|balance|not enough"; then
        echo -e "${RED}${BOLD}✗ Insufficient balance. Top up your wallet and re-run.${RESET}"
        exit 1
    fi

    # ── Timeout ───────────────────────────────────────────────
    if echo "${output}" | grep -qiE "timeout|exit 3"; then
        echo -e "${YELLOW}  btcli timed out — retrying in 15s...${RESET}"
        sleep 15
        continue
    fi

    # ── Unknown error — short wait ───────────────────────────
    echo -e "${RED}  Unknown error — waiting 30s before retry...${RESET}"
    sleep 30
done
