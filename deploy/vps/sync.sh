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
# docker-compose.yml / Dockerfile / this script live outside the repo clone (docker compose's
# build context needs them beside each other, and it keeps `git reset --hard` from ever touching
# them directly) -- copy the repo's tracked copies over on every sync so they can't silently
# drift the way they did until 2026-08-05 (a docker-compose.yml edit sat unused for a full deploy
# cycle because nothing ever copied it from the repo to here).
cp /docker/basis/app/deploy/vps/docker-compose.yml /docker/basis/docker-compose.yml
cp /docker/basis/app/deploy/vps/Dockerfile /docker/basis/Dockerfile
cp /docker/basis/app/deploy/vps/sync.sh /docker/basis/sync.sh
chmod +x /docker/basis/sync.sh
# Rebuild the opportunities cache from the new data BEFORE restarting, so the
# app comes back instantly instead of sitting minutes on load_signals().
docker exec basis-basis-1 python run_daily.py || echo "signal rebuild failed (app will rebuild on first load)"
# --build is a cached no-op unless requirements.txt / Dockerfile changed.
docker compose -f /docker/basis/docker-compose.yml up -d --build
docker restart basis-basis-1
echo "$(date -u '+%F %T') sync done"
