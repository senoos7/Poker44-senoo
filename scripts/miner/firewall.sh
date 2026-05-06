#!/usr/bin/env bash
# ============================================================
# Poker44 Miner Firewall Manager
#
# Restricts axon ports to known validator IPs only, blocking
# DDoS floods (empty-synapse probes, port scanners, etc.).
#
# Usage:
#   ./scripts/miner/firewall.sh setup      # Initial setup
#   ./scripts/miner/firewall.sh add-ip 1.2.3.4   # Add new validator IP
#   ./scripts/miner/firewall.sh status     # Show current rules
#   ./scripts/miner/firewall.sh reset      # Restore open access (debug)
#
# When a new validator comes online you will see a new IP in the
# dashboard. Add it with:  ./scripts/miner/firewall.sh add-ip <IP>
# ============================================================

set -euo pipefail

# ---- Known validator IPs (add new ones with "add-ip" sub-command) ----
# Updated 2026-05-06 from dashboard + ss -ant analysis
VALIDATOR_IPS=(
  # Google LLC — same validator org, two IPs
  "136.119.82.183"
  "35.193.38.111"
  # WildSage Labs (Almaty) — same validator org, two IPs
  "167.150.153.126"
  "167.150.153.79"
  # Contabo GmbH (FR)
  "185.196.20.208"
  # Oso Grande Technologies (US)
  "192.150.253.122"
  # Cherry Servers (Chicago)
  "84.32.70.8"
  # Hetzner (Ashburn) — possibly multiple hosts from same operator
  "5.161.37.94"
  "5.161.39.30"
  "5.161.36.120"
  # Sharktech (Las Vegas)
  "208.98.44.2"
)

# ---- All active miner axon ports ----
MINER_PORTS=(8091 8092 8093 8094 8095 8096 8099 8100 8102 8105 8111 8201)

# ---- Helpers ----
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*"; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

require_root() {
    [[ $EUID -eq 0 ]] || die "Run with sudo: sudo $0 $*"
}

# Remove any existing broad ALLOW rules for miner ports
# (the ones we added earlier with "ufw allow <port>/tcp")
_clear_miner_port_rules() {
    info "Removing broad allow rules for miner ports..."
    for port in "${MINER_PORTS[@]}"; do
        # ufw delete returns non-zero if rule doesn't exist — ignore
        ufw delete allow "${port}/tcp" 2>/dev/null || true
        ufw delete deny  "${port}/tcp" 2>/dev/null || true
        # Also clear any "from any" style rules
        ufw delete allow "${port}" 2>/dev/null || true
        ufw delete deny  "${port}" 2>/dev/null || true
    done
}

# Allow one validator IP to connect to all miner ports
_allow_validator() {
    local ip="$1"
    info "  Allowing validator $ip on all miner ports"
    for port in "${MINER_PORTS[@]}"; do
        ufw allow from "$ip" to any port "$port" proto tcp comment "validator-${ip}" 2>/dev/null || true
    done
}

# ---- Sub-commands ----

cmd_setup() {
    require_root

    echo "=================================================="
    echo " Poker44 Firewall Setup — validator-only axon ports"
    echo "=================================================="

    # Step 1: clear existing broad rules
    _clear_miner_port_rules

    # Step 2: allow each known validator IP
    info "Adding validator IP allowlist..."
    for ip in "${VALIDATOR_IPS[@]}"; do
        _allow_validator "$ip"
    done

    # Step 3: deny everything else on miner ports
    # These deny rules come AFTER the allow-from-IP rules in UFW's chain,
    # so specific allows take precedence.
    info "Adding deny-all for miner ports (blocks DDoS/scanners)..."
    for port in "${MINER_PORTS[@]}"; do
        ufw deny "${port}/tcp" comment "block-non-validator" 2>/dev/null || true
    done

    ufw reload
    echo ""
    info "Done. Summary:"
    cmd_status
}

cmd_add_ip() {
    require_root
    local new_ip="${1:-}"
    [[ -n "$new_ip" ]] || die "Usage: $0 add-ip <IP>"

    # Validate rough IP format
    [[ "$new_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "Invalid IP: $new_ip"

    info "Adding new validator IP: $new_ip"
    for port in "${MINER_PORTS[@]}"; do
        # Insert BEFORE the deny rules (insert at position 1 in the user chain)
        # ufw insert ensures the allow rule precedes existing deny rules.
        ufw insert 1 allow from "$new_ip" to any port "$port" proto tcp \
            comment "validator-${new_ip}" 2>/dev/null || true
    done
    ufw reload
    info "Done. Add '$new_ip' to VALIDATOR_IPS array in this script for persistence."
}

cmd_status() {
    echo ""
    echo "--- UFW status (miner ports) ---"
    ufw status numbered | grep -E "(8091|8092|8093|8094|8095|8096|8099|8100|8102|8105|8111)" || \
        echo "  (no rules found for miner ports)"
    echo ""
    echo "--- Active validator IPs in this script ---"
    for ip in "${VALIDATOR_IPS[@]}"; do
        echo "  $ip"
    done
}

cmd_reset() {
    require_root
    warn "Resetting miner ports to open access (all IPs allowed)..."
    _clear_miner_port_rules
    for port in "${MINER_PORTS[@]}"; do
        ufw allow "${port}/tcp" comment "miner-open" || true
    done
    ufw reload
    info "All miner ports now open. Re-run 'setup' to re-enable protection."
}

# ---- Rate-limit alternative (safer for unknown validator IPs) ----
cmd_ratelimit() {
    require_root
    warn "Rate-limit mode: limits each IP to 6 new connections per 30s on miner ports."
    warn "This is less strict than IP whitelist but handles unknown validator IPs."
    _clear_miner_port_rules
    for port in "${MINER_PORTS[@]}"; do
        ufw limit "${port}/tcp" comment "miner-ratelimit" || true
    done
    ufw reload
    info "Rate limiting active on all miner ports."
}

# ---- Entry point ----
CMD="${1:-help}"
shift || true

case "$CMD" in
    setup)      cmd_setup ;;
    add-ip)     cmd_add_ip "${1:-}" ;;
    status)     cmd_status ;;
    reset)      cmd_reset ;;
    ratelimit)  cmd_ratelimit ;;
    help|*)
        echo "Usage: sudo $0 <command>"
        echo ""
        echo "Commands:"
        echo "  setup        Restrict miner ports to known validator IPs only"
        echo "  add-ip <IP>  Add a new validator IP (run when a new validator appears)"
        echo "  status       Show current rules"
        echo "  reset        Re-open all miner ports (debugging)"
        echo "  ratelimit    Rate-limit mode: allow all IPs but cap at 6 conn/30s"
        ;;
esac
