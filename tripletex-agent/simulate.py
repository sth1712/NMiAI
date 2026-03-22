"""
Simulering av kritiske oppgavetyper mot deployed agent + sandbox.
Kjør: python3 simulate.py
"""
import requests
import json
import time

AGENT_URL = "https://tripletex-agent-421519138388.europe-north1.run.app"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM3MDIwLCJ0b2tlbiI6ImI1ODIyYzhlLTNiNzktNDhiMS1hMDc3LWRkZWVlMjdkMWNkNyJ9"

TS = str(int(time.time()))[-6:]

TASKS = [
    {
        "name": "Feilretting hovedbok (Tier 3)",
        "prompt": f"Vi har oppdaget feil i hovedboka. Konto 6860 ble brukt i stedet for 6590 (beløp 5100 kr). Rett feilen med et korreksjonsbilag."
    },
    {
        "name": "Leverandørfaktura via voucher (Tier 2)",
        "prompt": f"Registrer leverandørfaktura FV-{TS} fra Kontorservice AS på 12500 NOK inkl MVA for kontorrekvisita"
    },
    {
        "name": "Lønnskjøring via voucher (Tier 2/3)",
        "prompt": f"Kjør lønn for ansatt Marie Hansen. Grunnlønn 42000 kr + overtid 5500 kr."
    },
    {
        "name": "Reiseregning med kostnader (Tier 2)",
        "prompt": f"Opprett reiseregning for den innloggede brukeren. Tittel: Kundemøte Oslo {TS}. Dato: 2026-03-15. Legg til fly 3500 kr, hotell 1800 kr og taxi 450 kr."
    },
    {
        "name": "Avskrivning (Tier 3)",
        "prompt": f"Avskriv kontorutstyr til verdi 60000 kr med 20% lineær avskrivning. Registrer avskrivningsbilag."
    },
    {
        "name": "Periodisering (Tier 3)",
        "prompt": f"Forskuddsbetal 24000 kr for årlig forsikring. Registrer betalingen og periodiser første måneds kostnad (2000 kr)."
    },
    {
        "name": "Bankgebyr (Tier 3)",
        "prompt": f"Registrer bankgebyr på 350 kr for mars 2026."
    },
    {
        "name": "Utvidet lønn m/skatt+AGA (Tier 3)",
        "prompt": f"Kjør lønn for ansatt med grunnlønn 48000 kr, skattetrekk 33%, og arbeidsgiveravgift 14.1%. Registrer som lønnsbilag."
    },
    {
        "name": "Betaling eksisterende faktura (Tier 2)",
        "prompt": f"Kunden har en utestående faktura. Registrer full betaling på fakturaen."
    },
    {
        "name": "Kredittnotat (Tier 2)",
        "prompt": f"Lag kreditnota for den siste fakturaen i systemet."
    },
]

def run_task(task):
    """Send task to agent and return result."""
    start = time.time()
    try:
        r = requests.post(
            f"{AGENT_URL}/solve",
            json={
                "prompt": task["prompt"],
                "files": [],
                "tripletex_credentials": {
                    "base_url": SANDBOX_URL,
                    "session_token": SANDBOX_TOKEN
                }
            },
            timeout=120
        )
        elapsed = time.time() - start
        return r.status_code, elapsed
    except Exception as e:
        return f"ERROR: {e}", time.time() - start

print(f"=== SIMULERING — {len(TASKS)} oppgaver ===\n")

results = []
for i, task in enumerate(TASKS):
    print(f"[{i+1}/{len(TASKS)}] {task['name']}...")
    status, elapsed = run_task(task)
    result = "OK" if status == 200 else f"FEIL ({status})"
    results.append((task["name"], result, elapsed))
    print(f"  → {result} ({elapsed:.1f}s)")
    time.sleep(1)

print(f"\n=== RESULTATER ===\n")
for name, result, elapsed in results:
    icon = "✅" if result == "OK" else "❌"
    print(f"  {icon} {name}: {result} ({elapsed:.1f}s)")

ok_count = sum(1 for _, r, _ in results if r == "OK")
print(f"\n  {ok_count}/{len(results)} returnerte 200")
print(f"\n⚠️  200 betyr bare at agenten SVARTE — sjekk loggene for 201 vs 422 på API-kallene!")
print(f"  Kjør: gcloud run services logs read tripletex-agent --region europe-north1 --limit 200")
