#!/usr/bin/with-contenv bashio
set -e
bashio::log.info "Starting Greece Sky and Weather Nowcast Add-on..."
python3 /app/main.py
