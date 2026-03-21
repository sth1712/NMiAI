"""Quick test — kjør én oppgave mot deployed agent."""
import requests, json

AGENT_URL = "https://tripletex-agent-421519138388.europe-north1.run.app"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM3MDIwLCJ0b2tlbiI6ImI1ODIyYzhlLTNiNzktNDhiMS1hMDc3LWRkZWVlMjdkMWNkNyJ9"

prompt = "Vi har mottatt faktura INV-2026-100 fra leverandør TestLev AS (org.nr 987654321) på 25000 NOK inkl MVA. Beløpet gjelder kontortjenester (konto 7100). Registrer leverandørfakturaen med korrekt inngående MVA (25%)."

r = requests.post(f"{AGENT_URL}/solve", json={
    "prompt": prompt,
    "files": [],
    "tripletex_credentials": {"base_url": SANDBOX_URL, "session_token": SANDBOX_TOKEN}
}, timeout=120)

print(f"Status: {r.status_code}")
print(f"Response: {r.text}")
