"""Test purring-oppgave isolert."""
import requests

r = requests.post("https://tripletex-agent-421519138388.europe-north1.run.app/solve", json={
    "prompt": "Kunden har en forfalt faktura. Registrer purregebyr paa 55 NOK. Debet kundefordring (1500), kredit purregebyr-inntekt (3400).",
    "files": [],
    "tripletex_credentials": {
        "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
        "session_token": "eyJ0b2tlbklkIjoyMTQ3NjM3MDIwLCJ0b2tlbiI6ImI1ODIyYzhlLTNiNzktNDhiMS1hMDc3LWRkZWVlMjdkMWNkNyJ9"
    }
}, timeout=120)
print(r.status_code, r.text)
