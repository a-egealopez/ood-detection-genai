#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Cleaning results/ and wandb/ under $ROOT ..."
find "$ROOT/results" -mindepth 1 -delete 2>/dev/null || true
echo "Done."
