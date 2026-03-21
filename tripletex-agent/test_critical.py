"""Test de mest kritiske oppgavetypene — fokus på de som har feilet."""
import requests, json, time

AGENT_URL = "https://tripletex-agent-421519138388.europe-north1.run.app"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM3MDIwLCJ0b2tlbiI6ImI1ODIyYzhlLTNiNzktNDhiMS1hMDc3LWRkZWVlMjdkMWNkNyJ9"

TS = str(int(time.time()))[-5:]

TASKS = [
    {
        "name": "Faktura med eksisterende produkter + betaling",
        "prompt": f"Opprett en ordre til kunde Test AS med produktet TestProdukt (produktnummer 9999) til 1500 NOK. Fakturér ordren og registrer full betaling."
    },
    {
        "name": "Leverandørfaktura med orgnr",
        "prompt": f"Vi har mottatt faktura FV-{TS} fra leverandøren NattTest Leverandor AS (org.nr som finnes i systemet) på 18750 NOK inkl MVA. Beløpet gjelder kontorrekvisita (konto 6800). Registrer leverandørfakturaen med korrekt inngående MVA (25%)."
    },
    {
        "name": "Purring/reminder",
        "prompt": f"Kunden har en forfalt faktura. Registrer purregebyr på 55 NOK. Debet kundefordring (1500), kredit purregebyr-inntekt (3400)."
    },
    {
        "name": "Feilretting hovedbok",
        "prompt": f"Vi har oppdaget feil i hovedboka. Konto 6860 ble brukt i stedet for 6590 (beløp 3200 kr). Rett feilen med et korreksjonsbilag."
    },
    {
        "name": "Reiseregning med kostnader",
        "prompt": f"Opprett reiseregning for den innloggede brukeren. Tittel: Kundemøte {TS}. Dato: 2026-03-18. Legg til fly 3500 kr og hotell 1800 kr."
    },
]

def run(task):
    start = time.time()
    try:
        r = requests.post(f"{AGENT_URL}/solve", json={
            "prompt": task["prompt"], "files": [],
            "tripletex_credentials": {"base_url": SANDBOX_URL, "session_token": SANDBOX_TOKEN}
        }, timeout=120)
        return r.status_code, time.time() - start
    except Exception as e:
        return f"ERR: {e}", time.time() - start

print(f"=== KRITISKE TESTER ({len(TASKS)}) ===\n")
for i, t in enumerate(TASKS):
    print(f"[{i+1}/{len(TASKS)}] {t['name']}...")
    status, elapsed = run(t)
    print(f"  → {'OK' if status == 200 else f'FEIL ({status})'} ({elapsed:.1f}s)")
    time.sleep(0.5)

print(f"\nSjekk logger: gcloud run services logs read tripletex-agent --region europe-north1 --limit 100")
