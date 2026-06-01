#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <public-hostname> [backend-host:port]" >&2
  echo "Example: $0 suno.example.com 127.0.0.1:8765" >&2
  exit 2
fi

PUBLIC_HOST="$1"
BACKEND="${2:-127.0.0.1:8765}"
PROJECT_DIR="/home/seutje/projects/discography"
SITE_ADDR="$PUBLIC_HOST"

if [[ "$PUBLIC_HOST" =~ ^https?:// ]]; then
  SITE_ADDR="$PUBLIC_HOST"
elif [[ "$PUBLIC_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  SITE_ADDR="http://$PUBLIC_HOST"
fi

if ! command -v caddy >/dev/null 2>&1; then
  echo "Caddy is not installed. Install it first with: sudo apt install caddy" >&2
  exit 1
fi

sudo install -m 0644 "$PROJECT_DIR/deploy/suno-dashboard.service" /etc/systemd/system/suno-dashboard.service
sudo systemctl daemon-reload
pkill -f "$PROJECT_DIR/scripts/suno_server.py" 2>/dev/null || true
sudo systemctl enable --now suno-dashboard.service

sudo mkdir -p /etc/caddy
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
# Managed by $PROJECT_DIR/deploy/install_suno_proxy.sh

$SITE_ADDR {
	encode zstd gzip

	@callback path /api/suno/callback/* /api/suno/wav-callback/*
	reverse_proxy @callback $BACKEND

	respond 404
}
EOF

sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl enable --now caddy
sudo systemctl reload caddy

echo "Installed Suno callback proxy:"
echo "  $SITE_ADDR/api/suno/callback/<token>/<job_id>/<iteration>"
echo "  $SITE_ADDR/api/suno/wav-callback/<token>/<job_id>/<iteration>/<candidate>"
echo
if [[ "$SITE_ADDR" == http://* ]]; then
  echo "Forward this router port to 192.168.0.181:"
  echo "  TCP 80 -> 192.168.0.181:80"
else
  echo "Forward these router ports to 192.168.0.181:"
  echo "  TCP 80  -> 192.168.0.181:80"
  echo "  TCP 443 -> 192.168.0.181:443"
fi
