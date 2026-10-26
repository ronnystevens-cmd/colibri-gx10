#!/usr/bin/env bash
# klaar-voor-run.sh — start de definitieve f7g 64-token GPU-run in een owned shell.
# VOORWAARDE: default heeft vLLM-pool :8000 gestopt (sequentieregel, afspraak Ronny).
# Dit script aanroezen via SSH is ONVOLDOENDE als je de run zelf af schiet: bij het
# afsluiten van de SSH-sessie sterft de run (SIGHUP). Daarom: setsid + los log + vlag.
set -u
cd /home/flitsken/colibri
exec >> /home/flitsken/colibri/klaar-voor-run.log 2>&1
echo "[$(date '+%d/%m %H:%M:%S')] klaar-voor-run.sh aangeroepen"

# 1) POOL MOET WEG ZIJN — anders niet starten (crash-van-vannacht-regel)
if curl -s --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 'model'; then
  echo "[$(date '+%d/%m %H:%M:%S')] Weiger: pool :8000 draait nog — eerst stoppen (default)."
  exit 3
fi
echo "[$(date '+%d/%m %H:%M:%S')] pool :8000 is stil — OK"

# 2) maintenance-vlag (watchdog grijpt niet in) + stale lock opschonen
touch /home/flitsken/POOL-MAINTENANCE
rm -f /home/flitsken/.colibri-gpu.lock

# 3) de run: identiteit-voorbewijs (elke run, verftsbewust) + 64 tokens, anker=989
setsid nohup python3 f7g_run16.py 64 > /home/flitsken/colibri/f7gpu64.log 2>&1 < /dev/null &
echo "[$(date '+%d/%m %H:%M:%S')] run afgeschoten PID $! — log: /home/flitsken/colibri/f7gpu64.log"

# 4) na afloop: lock + vlag netjes weg zodat de watchdog de pool terug kan halen
( while pgrep -f "f7g_run16.py 64" >/dev/null; do sleep 30; done
  rm -f /home/flitsken/.colibri-gpu.lock
  rm -f /home/flitsken/POOL-MAINTENANCE
  echo "[$(date '+%d/%m %H:%M:%S')] run beëindigd — POOL-MAINTENANCE weg, watchdog kan pool herstellen"
) >> /home/flitsken/colibri/klaar-voor-run.log 2>&1 &
