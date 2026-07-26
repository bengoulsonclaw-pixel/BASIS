# BASIS on the Hostinger VPS

Live at **https://basis.srv1608260.hstgr.cloud** (HTTP basic auth, user `ben`;
the password sits on the dev laptop in `~\.ssh\basis_site_password.txt` — it is
NOT in git; the compose file carries only its apr1 hash). Set up 2026-07-25.

**View-only by design.** No Bloomberg exists on the VPS, so the app runs on the
data stores committed to git — the "DEMO MODE" badge in the sidebar is expected.
Data freshness = the laptop's last push (the 22:00 nightly backup task, plus any
session pushes). Bloomberg pulls, scheduled reports and emails all stay on the
laptop. Per the Bloomberg licence, the site is for Ben's own use — do not hand
out logins.

## Layout on the VPS (srv1608260.hstgr.cloud / 2.24.221.3, root via SSH key)

- `/docker/basis/` — these three files + `app/` (clone of this repo via a
  read-only GitHub deploy key, `/root/.ssh/basis_deploy`)
- Routing: the pre-existing Hostinger Traefik container (host mode, ports
  80/443, Let's Encrypt) picks the container up from its compose labels.
  OpenClaw runs on the same box — do not touch `/docker/openclaw-*` or
  `/docker/traefik`.
- Sync: root crontab runs `/docker/basis/sync.sh` every 15 min — on new
  commits it pulls, rebuilds the signals cache inside the container
  (`run_daily.py`), then restarts the app so Streamlit's caches clear.
  Log: `/var/log/basis_sync.log`.

## Rebuild from nothing

```
ssh root@2.24.221.3          # key: ~\.ssh\basis_vps on the laptop
mkdir -p /docker/basis && git clone git@github.com:bengoulsonclaw-pixel/BASIS.git /docker/basis/app
# copy these three files to /docker/basis/, then:
cd /docker/basis && docker compose up -d --build
docker exec basis-basis-1 python run_daily.py && docker restart basis-basis-1
(crontab -l; echo '*/15 * * * * /docker/basis/sync.sh >> /var/log/basis_sync.log 2>&1') | crontab -
```

To change the site password: `openssl passwd -apr1 '<new-pw>'`, put the hash in
`docker-compose.yml` (every `$` doubled to `$$`), `docker compose up -d`.
