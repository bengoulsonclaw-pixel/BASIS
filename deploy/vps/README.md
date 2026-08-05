# BASIS on the Hostinger VPS

Live at **https://basisterminal.com** (also www.basisterminal.com and
basis.srv1608260.hstgr.cloud). Set up 2026-07-25; per-user login added
2026-08-05 (see below) — the old single shared Traefik password is gone,
`~\.ssh\basis_site_password.txt` / `basis_site_hash.txt` on the dev laptop are
vestigial.

DNS: basisterminal.com is registered at Hostinger (3-yr term from 2026-07-25);
its zone has A `@` -> 2.24.221.3 and CNAME `www` -> basisterminal.com.

**View-only by design.** No Bloomberg exists on the VPS, so the app runs on the
data stores committed to git — the "DEMO MODE" badge in the sidebar is expected.
Data freshness = the laptop's last push (the 22:00 nightly backup task, plus any
session pushes). Bloomberg pulls, scheduled reports and emails all stay on the
laptop.

**Per-user login (`src/auth.py`, `BASIS_REQUIRE_LOGIN=1` in docker-compose.yml).**
The site now has real per-user accounts — an admin role (full access, same as
before) and a colleague role (view + generate + email-to-self only, no config
edits — see `data/users.json`, gitignored/local to the VPS). The old single
shared Traefik password is gone; login is handled inside the app.
**Before adding any real colleague account: confirm with whoever manages the
Bloomberg licence that colleagues viewing Bloomberg-derived data this way is
permitted** — this was originally "Ben's own use only" per that licence, and
that question hasn't been resolved yet as of this rollout.

## Layout on the VPS (srv1608260.hstgr.cloud / 2.24.221.3, root via SSH key)

- `/docker/basis/` — `docker-compose.yml`, `Dockerfile`, `sync.sh` (these three
  live OUTSIDE the git clone on purpose: `docker compose`'s build context needs
  them beside each other, and it keeps `git reset --hard` from ever touching
  them directly) + `app/` (clone of this repo via a read-only GitHub deploy
  key, `/root/.ssh/basis_deploy`).
- **These three files are the same as `deploy/vps/{docker-compose.yml,Dockerfile,sync.sh}`
  in the repo — edit them there, not on the VPS.** Every sync copies the
  repo's tracked versions over the deployed ones before rebuilding, so a
  change lands automatically on the next sync (or immediately via `ssh
  root@2.24.221.3 /docker/basis/sync.sh`). Before 2026-08-05 this copy step
  didn't exist and the two could silently drift — a docker-compose.yml edit
  sat unused for a full deploy cycle before anyone noticed.
- Routing: the pre-existing Hostinger Traefik container (host mode, ports
  80/443, Let's Encrypt) picks the container up from its compose labels.
  OpenClaw runs on the same box — do not touch `/docker/openclaw-*` or
  `/docker/traefik`.
- Sync: root crontab runs `/docker/basis/sync.sh` every 15 min — on new
  commits it pulls, refreshes its own three deploy files from the repo,
  rebuilds the signals cache inside the container (`run_daily.py`), then
  restarts the app so Streamlit's caches clear. Log: `/var/log/basis_sync.log`.

## Rebuild from nothing

```
ssh root@2.24.221.3          # key: ~\.ssh\basis_vps on the laptop
mkdir -p /docker/basis && git clone git@github.com:bengoulsonclaw-pixel/BASIS.git /docker/basis/app
# One-time bootstrap copy -- sync.sh keeps these current automatically from here on:
cp /docker/basis/app/deploy/vps/docker-compose.yml /docker/basis/docker-compose.yml
cp /docker/basis/app/deploy/vps/Dockerfile /docker/basis/Dockerfile
cp /docker/basis/app/deploy/vps/sync.sh /docker/basis/sync.sh && chmod +x /docker/basis/sync.sh
cd /docker/basis && docker compose up -d --build
docker exec basis-basis-1 python run_daily.py && docker restart basis-basis-1
(crontab -l; echo '*/15 * * * * /docker/basis/sync.sh >> /var/log/basis_sync.log 2>&1') | crontab -
```

Accounts (admin + colleague) are managed from the local Terminal's System →
Colleague Accounts panel and pushed to the VPS over SSH — see `src/auth.py`.
There is no site-wide password anymore to change.
