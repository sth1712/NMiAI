"""Test employment-oppretting (uten PDF — simulerer kontraktdata i prompt)."""
import requests, json

AGENT_URL = "https://tripletex-agent-421519138388.europe-north1.run.app"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM3MDIwLCJ0b2tlbiI6ImI1ODIyYzhlLTNiNzktNDhiMS1hMDc3LWRkZWVlMjdkMWNkNyJ9"

# Simulerer PDF-kontraktdata direkte i prompten
prompt = """Du har mottatt en arbeidskontrakt. Her er detaljene fra kontrakten:

Navn: Maria Gonzalez Torres
Personnummer: 12345678901
Fodselsdato: 1990-05-15
E-post: maria.gonzalez@firma.no
Avdeling: Innkjop
Yrkeskode: Account Manager
Startdato: 2026-04-01
Stillingsprosent: 80%
Aarslonn: 520000 NOK
Ansettelsesform: Fast
Avlonningstype: Maanedslonn

Opprett ansatt med alle detaljer fra kontrakten, inkludert ansettelsesforhold med lonn og stillingsprosent."""

r = requests.post(f"{AGENT_URL}/solve", json={
    "prompt": prompt,
    "files": [],
    "tripletex_credentials": {"base_url": SANDBOX_URL, "session_token": SANDBOX_TOKEN}
}, timeout=120)

print(f"Status: {r.status_code}")
print(f"Response: {r.text}")
print(f"\nSjekk logger: gcloud run services logs read tripletex-agent --region europe-north1 --limit 30")
