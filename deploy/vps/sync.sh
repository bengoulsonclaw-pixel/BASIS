#!/bin/sh
# BASIS auto-sync: pull the repo every 15 min; on new commits rebuild the
# signals cache with the fresh data, then restart the app to clear its caches.
# Cron: */15 * * * * /docker/basis/sync.sh >> /var/log/basis_sync.log 2>&1
set -e
cd /docker/basis/app
git fetch -q origin main
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
[ "$LOCAL" = "$REMOTE" ] && exit 0

echo "$(date -u '+%F %T') syncing $LOCAL -> $REMOTE"
git reset --hard -q origin/main
# Rebuild the opportunities cache from the new data BEFORE restarting, so the
# app comes back instantly instead of sitting minutes on load_signals().
docker exec basis-basis-1 python run_daily.py || echo "signal rebuild failed (app will rebuild on first load)"
# --build is a cached no-op unless requirements.txt / Dockerfile changed.
docker compose -f /docker/basis/docker-compose.yml up -d --build
docker restart basis-basis-1
echo "$(date -u '+%F %T') sync done"
