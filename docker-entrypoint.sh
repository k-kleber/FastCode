#!/usr/bin/env bash
set -euo pipefail

if [ "${FASTCODE_PRELOAD_ON_STARTUP:-1}" = "1" ]; then
  echo "[fastcode] Preloading embedding model before MCP startup..."
  python preload_embedding_model.py
fi

exec python mcp_server.py "$@"
