#!/bin/bash
# ============================================================
# auto_unstake.sh — sequential unstake with delays + retries
#
# Why: `btcli stake remove --all` across many hotkeys often submits
# extrinsics back-to-back and hits Substrate "Transaction is outdated"
# (stale mortality / nonce). This script unstakes ONE hotkey per
# extrinsic, optional delay between calls, and retries on
# outdated errors — same automation style as auto_register.sh.
#
# Usage:
#   ./auto_unstake.sh --wallet superbit-darnsin --netuid 126 --full
#   ./auto_unstake.sh --wallet superbit-darnsin --netuid 126 --hotkeys-file ./hotkeys.txt
#   ./auto_unstake.sh --wallet superbit-darnsin --netuid 126 --hotkey poker-miner-26001
#   ./auto_unstake.sh --wallet superbit-darnsin --netuid 126 --include-hotkeys "hk1,hk2,hk3"
#
# hotkeys.txt: one hotkey name per line; lines starting with # are ignored.
#
# Optional:
#   --network finney|test|local     (default: finney)
#   --password <pass>               (omit to be prompted)
#   --delay-seconds <n>             sleep after each *successful* unstake (default: 18)
#   --retry-delay-seconds <n>       base sleep on outdated retry (default: 24)
#   --full                          unstake all hotkeys on this coldkey for --netuid (enumerates wallet/hotkeys/ individually;
#                                   deregistered / pruned hotkeys are silently skipped — no HotKeyAccountNotExists errors)
#   --max-retries <n>               per hotkey or per --full run (default: 8)
#   --alpha-only                    pass --all-alpha instead of --all (alpha stake only)
#   --wait-finalization             pass --wait-for-finalization (only if your btcli supports it)
#   --no-wait-finalization          never pass --wait-for-finalization (default; many btcli versions lack this flag on stake remove)
#   --extra-args "..."              appended to every btcli invocation (quote carefully)
#
# Build hotkeys.txt from `btcli stake list` output, or paste names from the unstake summary.
#
# Environment (optional overrides):
#   BTCLI_BIN=btcli
# ============================================================

set -euo pipefail
# expect failures are handled explicitly in unstake_one()

# ── Colours ─────────────────────────────────────────────────
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
CYAN="\033[36m"
BOLD="\033[1m"
RESET="\033[0m"

# ── Defaults ─────────────────────────────────────────────────
WALLET_NAME=""
NETUID=""
NETWORK="finney"
WALLET_PASSWORD=""
HOTKEY_SINGLE=""
HOTKEYS_FILE=""
INCLUDE_HOTKEYS=""
FULL_MODE=0
DELAY_AFTER_OK_SECS=18
RETRY_BASE_DELAY_SECS=24
MAX_RETRIES=8
ALPHA_ONLY=0
WAIT_FINALIZATION=0
EXTRA_ARGS=()

BTCLI_BIN="${BTCLI_BIN:-btcli}"

# ── Argument parsing ─────────────────────────────────────────
usage() {
    echo "Usage: $0 --wallet <coldkey> --netuid <n> (--full | --hotkey <name> | --hotkeys-file <path> | --include-hotkeys hk1,hk2,...)"
    echo "See script header for full options."
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wallet) WALLET_NAME="$2"; shift 2 ;;
        --netuid) NETUID="$2"; shift 2 ;;
        --network) NETWORK="$2"; shift 2 ;;
        --password) WALLET_PASSWORD="$2"; shift 2 ;;
        --hotkey) HOTKEY_SINGLE="$2"; shift 2 ;;
        --hotkeys-file) HOTKEYS_FILE="$2"; shift 2 ;;
        --include-hotkeys) INCLUDE_HOTKEYS="$2"; shift 2 ;;
        --full) FULL_MODE=1; shift 1 ;;
        --delay-seconds) DELAY_AFTER_OK_SECS="$2"; shift 2 ;;
        --retry-delay-seconds) RETRY_BASE_DELAY_SECS="$2"; shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        --alpha-only) ALPHA_ONLY=1; shift 1 ;;
        --wait-finalization) WAIT_FINALIZATION=1; shift 1 ;;
        --no-wait-finalization) WAIT_FINALIZATION=0; shift 1 ;;
        --extra-args)
            # shellcheck disable=SC2206
            EXTRA_ARGS=($2)
            shift 2
            ;;
        -h|--help) usage ;;
        *) echo -e "${RED}Unknown argument: $1${RESET}"; usage ;;
    esac
done

if [[ -z "$WALLET_NAME" || -z "$NETUID" ]]; then
    echo -e "${RED}Error: --wallet and --netuid are required.${RESET}"
    usage
fi

hotkeys_list=()
if [[ -n "$HOTKEY_SINGLE" ]]; then
    hotkeys_list+=("$HOTKEY_SINGLE")
fi
if [[ -n "$INCLUDE_HOTKEYS" ]]; then
    IFS=',' read -r -a _split <<< "$INCLUDE_HOTKEYS"
    for h in "${_split[@]}"; do
        h="${h//[[:space:]]/}"
        [[ -n "$h" ]] && hotkeys_list+=("$h")
    done
fi
if [[ -n "$HOTKEYS_FILE" ]]; then
    if [[ ! -f "$HOTKEYS_FILE" ]]; then
        echo -e "${RED}Error: hotkeys file not found: ${HOTKEYS_FILE}${RESET}"
        exit 1
    fi
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line#"${line%%[![:space:]]*}"}"
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        hotkeys_list+=("$line")
    done <"$HOTKEYS_FILE"
fi

if [[ "$FULL_MODE" -eq 1 ]]; then
    if [[ -n "$HOTKEY_SINGLE" || -n "$HOTKEYS_FILE" || -n "$INCLUDE_HOTKEYS" ]]; then
        echo -e "${RED}Error: --full cannot be combined with --hotkey, --hotkeys-file, or --include-hotkeys.${RESET}"
        exit 1
    fi
elif [[ ${#hotkeys_list[@]} -eq 0 ]]; then
    echo -e "${RED}Error: use --full, or provide at least one hotkey via --hotkey, --include-hotkeys, or --hotkeys-file.${RESET}"
    usage
fi

# ── expect ───────────────────────────────────────────────────
if ! command -v expect &>/dev/null; then
    echo -e "${YELLOW}'expect' not found. Install: sudo apt install expect${RESET}"
    exit 1
fi

if [[ -z "$WALLET_PASSWORD" ]]; then
    read -r -s -p "Wallet password: " WALLET_PASSWORD
    echo ""
fi

# Tcl reads this inside expect (avoids bash quoting issues if password contains ")
export AUTOUNSTAKE_PW="$WALLET_PASSWORD"

ALL_FLAG=(--all)
if [[ "$ALPHA_ONLY" -eq 1 ]]; then
    ALL_FLAG=(--all-alpha)
fi

echo -e "${BOLD}================================================${RESET}"
echo -e "${BOLD}  Bittensor sequential unstake${RESET}"
echo -e "${BOLD}================================================${RESET}"
echo -e "  coldkey     : ${WALLET_NAME}"
echo -e "  netuid      : ${NETUID}"
echo -e "  network     : ${NETWORK}"
if [[ "$FULL_MODE" -eq 1 ]]; then
    echo -e "  mode        : ${BOLD}full${RESET} (per-hotkey from ~/.bittensor/wallets/${WALLET_NAME}/hotkeys/)"
else
    echo -e "  hotkeys     : ${#hotkeys_list[@]}"
fi
echo -e "  delay OK    : ${DELAY_AFTER_OK_SECS}s"
echo -e "  retry base  : ${RETRY_BASE_DELAY_SECS}s  (max retries: ${MAX_RETRIES})"
echo -e "  unstake     : ${ALL_FLAG[*]}"
if [[ "$WAIT_FINALIZATION" -eq 1 ]]; then
    echo -e "  finalize    : --wait-for-finalization (opt-in; needs a btcli that supports it)"
else
    echo -e "  finalize    : off (default; stake remove often has no --wait-for-finalization). Use --delay-seconds between hotkeys."
fi
echo ""

# $1 = log label   $2 = extra args after --netuid (either: --include-hotkeys NAME  or  --all-hotkeys)
run_unstake_attempt() {
    local label="$1"
    local hk_target_args="$2"
    local attempt=0
    local output
    local rc
    local wait_args=()

    if [[ "$WAIT_FINALIZATION" -eq 1 ]]; then
        wait_args+=(--wait-for-finalization)
    fi

    while (( attempt < MAX_RETRIES )); do
        attempt=$((attempt + 1))
        local ts
        ts=$(date '+%Y-%m-%d %H:%M:%S')

        echo -e "${CYAN}[${ts}] ${label} — attempt ${attempt}/${MAX_RETRIES}${RESET}"

        set +e
        output=$(
            expect <<EOF
log_user 1
set timeout 600
spawn ${BTCLI_BIN} stake remove --subtensor.network ${NETWORK} --wallet-name ${WALLET_NAME} --netuid ${NETUID} ${hk_target_args} ${ALL_FLAG[*]} --yes ${wait_args[*]} ${EXTRA_ARGS[*]}
expect {
    -re {Do you want to proceed} { send "y\r"; exp_continue }
    -re {Do you want to continue} { send "y\r"; exp_continue }
    -re {Enter your password} { send "\$env(AUTOUNSTAKE_PW)\r"; exp_continue }
    -re {Decrypting} { exp_continue }
    timeout { exit 3 }
    eof
}
EOF
        )
        rc=$?
        set -e

        echo "${output}"

        if echo "${output}" | grep -qiE "Not enough stake|no stake|nothing to unstake|0\\.0+.*stake|Insufficient stake|HotKeyAccountNotExists|HotkeyAccountNotExists|hotkey.*does not exist|does not exist.*hotkey"; then
            echo -e "${YELLOW}  Skip (no stake / account not on-chain): ${label}${RESET}"
            return 0
        fi

        if echo "${output}" | grep -qiE "successfully|Success|Finalized|unstaked|completed|✓"; then
            echo -e "${GREEN}  OK: ${label}${RESET}"
            return 0
        fi

        if echo "${output}" | grep -qiE "Transaction is outdated|outdated|Invalid Transaction|SubstrateRequestException"; then
            local wait_s=$(( RETRY_BASE_DELAY_SECS + (attempt - 1) * 6 ))
            echo -e "${YELLOW}  Outdated / invalid tx — waiting ${wait_s}s then retry...${RESET}"
            sleep "${wait_s}"
            continue
        fi

        if [[ "$rc" == "3" ]]; then
            echo -e "${YELLOW}  expect timeout — retry in 30s...${RESET}"
            sleep 30
            continue
        fi

        echo -e "${RED}  Failed: ${label} (see output above).${RESET}"
        return 1
    done

    echo -e "${RED}  Giving up on: ${label}${RESET}"
    return 1
}

unstake_one_hotkey() {
    local hk="$1"
    run_unstake_attempt "${hk}" "--include-hotkeys ${hk}"
}

failed=0
if [[ "$FULL_MODE" -eq 1 ]]; then
    # --full: enumerate hotkeys from wallet filesystem and call each one
    # individually. This avoids btcli's --all-hotkeys batch flag which fails
    # the entire transaction if any single hotkey has no on-chain account
    # (HotKeyAccountNotExists for deregistered / pruned hotkeys).
    WALLET_PATH="${BITTENSOR_WALLET_PATH:-${HOME}/.bittensor/wallets}"
    HOTKEYS_DIR="${WALLET_PATH}/${WALLET_NAME}/hotkeys"
    if [[ ! -d "$HOTKEYS_DIR" ]]; then
        echo -e "${RED}Error: hotkeys directory not found: ${HOTKEYS_DIR}${RESET}"
        echo -e "${YELLOW}Hint: set BITTENSOR_WALLET_PATH if your wallets are not in ~/.bittensor/wallets${RESET}"
        exit 1
    fi

    mapfile -t full_hotkeys < <(ls -1 "$HOTKEYS_DIR" 2>/dev/null | sort)
    if [[ ${#full_hotkeys[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No hotkeys found in ${HOTKEYS_DIR}${RESET}"
        exit 0
    fi

    echo -e "${CYAN}Found ${#full_hotkeys[@]} hotkey(s) in wallet '${WALLET_NAME}': ${full_hotkeys[*]}${RESET}"
    echo ""

    idx=0
    for hk in "${full_hotkeys[@]}"; do
        idx=$((idx + 1))
        echo -e "${BOLD}[${idx}/${#full_hotkeys[@]}] Processing: ${hk}${RESET}"
        if ! unstake_one_hotkey "$hk"; then
            failed=$((failed + 1))
        fi
        if [[ $idx -lt ${#full_hotkeys[@]} ]]; then
            sleep "$DELAY_AFTER_OK_SECS"
        fi
    done
else
    for hk in "${hotkeys_list[@]}"; do
        if ! unstake_one_hotkey "$hk"; then
            failed=$((failed + 1))
        fi
        sleep "$DELAY_AFTER_OK_SECS"
    done
fi

echo ""
if [[ "$failed" -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}Done. All hotkeys processed without fatal errors.${RESET}"
    exit 0
else
    echo -e "${RED}${BOLD}Finished with ${failed} failure(s). Re-run after the chain settles (or use a hotkeys-file loop if --full keeps hitting outdated).${RESET}"
    exit 1
fi
