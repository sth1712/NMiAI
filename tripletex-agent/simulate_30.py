"""
Simulering av ALLE 30 oppgavetyper mot deployed agent + sandbox.
Kjør: python3 simulate_30.py
"""
import requests
import json
import time

AGENT_URL = "https://tripletex-agent-421519138388.europe-north1.run.app"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM3MDIwLCJ0b2tlbiI6ImI1ODIyYzhlLTNiNzktNDhiMS1hMDc3LWRkZWVlMjdkMWNkNyJ9"

TS = str(int(time.time()))[-6:]

TASKS = [
    # === TIER 1 (×1) ===
    {"name": "T1-01 Opprett ansatt (standard)", "tier": 1,
     "prompt": f"Opprett en ansatt med navn Lars Olsen, e-post lars.olsen.{TS}@firma.no"},

    {"name": "T1-02 Opprett ansatt (admin)", "tier": 1,
     "prompt": f"Opprett ansatt Kari Berg, e-post kari.berg.{TS}@firma.no. Hun skal være kontoadministrator."},

    {"name": "T1-03 Opprett ansatt (prosjektleder)", "tier": 1,
     "prompt": f"Opprett ansatt Per Holm, e-post per.holm.{TS}@firma.no. Han skal være prosjektleder."},

    {"name": "T1-04 Opprett kunde", "tier": 1,
     "prompt": f"Opprett en kunde med navn Nordvik AS {TS}, e-post kontakt@nordvik.no, organisasjonsnummer 912345678"},

    {"name": "T1-05 Opprett leverandør", "tier": 1,
     "prompt": f"Registrer leverandøren Bygg og Anlegg {TS} AS med e-post faktura@bygg.no og organisasjonsnummer 987654321"},

    {"name": "T1-06 Opprett produkt", "tier": 1,
     "prompt": f"Opprett produktet 'Konsulenttjeneste {TS}' med produktnummer {TS}. Pris 1500 NOK ekskl. MVA, standard 25% MVA-sats."},

    {"name": "T1-07 Opprett avdelinger", "tier": 1,
     "prompt": f"Opprett tre avdelinger: Økonomi (avdelingsnr 100), Salg (avdelingsnr 200) og IT (avdelingsnr 300)"},

    {"name": "T1-08 Opprett kunde+leverandør combo", "tier": 1,
     "prompt": f"Opprett Samarbeid {TS} AS som både kunde og leverandør, med e-post post@samarbeid.no"},

    {"name": "T1-09 Opprett kontaktperson", "tier": 1,
     "prompt": f"Legg til kontaktperson Erik Svendsen (erik.{TS}@firma.no, mobil 99887766) hos en eksisterende kunde"},

    {"name": "T1-10 Oppdater ansatt", "tier": 1,
     "prompt": f"Oppdater den første ansatte i systemet med mobilnummer 41122334"},

    # === TIER 2 (×2) ===
    {"name": "T2-11 Opprett faktura (ny)", "tier": 2,
     "prompt": f"Lag en faktura til kunde Acme {TS} AS for 3 timer konsulentarbeid à 1200 NOK ekskl. MVA (25%)"},

    {"name": "T2-12 Prosjektfakturering", "tier": 2,
     "prompt": f"Opprett prosjekt Omega {TS} med en prosjektleder, og fakturér kunden Digital {TS} AS for 'Webutvikling' (25000 NOK ekskl. MVA)"},

    {"name": "T2-13 Betaling eksisterende faktura", "tier": 2,
     "prompt": f"Kunden har en utestående faktura. Finn fakturaen og registrer full betaling via bank."},

    {"name": "T2-14 Betalingsreversering", "tier": 2,
     "prompt": f"Betalingen på den siste fakturaen ble returnert av banken. Reverser betalingen slik at fakturaen viser utestående beløp igjen."},

    {"name": "T2-15 Kredittnotat", "tier": 2,
     "prompt": f"Kunden har reklamert på den siste fakturaen. Opprett en kreditnota som reverserer hele fakturaen."},

    {"name": "T2-16 Leverandørfaktura (voucher)", "tier": 2,
     "prompt": f"Registrer leverandørfaktura FV-{TS} fra Kontorpartner AS på 18750 NOK inkl MVA (25%) for kontorrekvisita"},

    {"name": "T2-17 Lønnskjøring (voucher)", "tier": 2,
     "prompt": f"Kjør lønn for ansatt Mona Larsen. Grunnlønn 38000 kr + bonus 4500 kr. Registrer som lønnsbilag."},

    {"name": "T2-18 Reiseregning m/kostnader", "tier": 2,
     "prompt": f"Opprett reiseregning for den innloggede brukeren. Tittel: Kundebesøk Bergen {TS}. Dato: 2026-03-18. Legg til fly 4200 kr, hotell 2100 kr, taxi 380 kr og mat 250 kr."},

    {"name": "T2-19 Timeregistrering", "tier": 2,
     "prompt": f"Registrer 7.5 timer for den innloggede brukeren på prosjekt i dag."},

    {"name": "T2-20 Oppdater kunde", "tier": 2,
     "prompt": f"Endre e-posten til den første kunden i systemet til ny.epost.{TS}@oppdatert.no"},

    {"name": "T2-21 Oppdater leverandør", "tier": 2,
     "prompt": f"Oppdater den første leverandøren i systemet med ny e-post innkjop.{TS}@leverandor.no"},

    {"name": "T2-22 Oppdater produkt", "tier": 2,
     "prompt": f"Oppdater det første produktet i systemet med ny pris 1850 NOK ekskl. MVA"},

    {"name": "T2-23 Slett reiseregning", "tier": 2,
     "prompt": f"Slett den siste reiseregningen i systemet"},

    {"name": "T2-24 Slett avdeling", "tier": 2,
     "prompt": f"Slett avdelingen med navn 'IT'"},

    # === TIER 3 (×3) ===
    {"name": "T3-25 Feilretting hovedbok", "tier": 3,
     "prompt": f"Vi har oppdaget feil i hovedboka. Konto 6860 ble brukt i stedet for 6590 (beløp 5100 kr). Rett feilen med et korreksjonsbilag."},

    {"name": "T3-26 Avskrivning", "tier": 3,
     "prompt": f"Avskriv kontorutstyr til verdi 80000 kr med 20% lineær avskrivning. Registrer avskrivningsbilag."},

    {"name": "T3-27 Periodisering", "tier": 3,
     "prompt": f"Forskuddsbetal 36000 kr for årlig leie. Registrer betalingen og periodiser første måneds kostnad (3000 kr)."},

    {"name": "T3-28 Bankgebyr", "tier": 3,
     "prompt": f"Registrer bankgebyr på 475 kr for mars 2026."},

    {"name": "T3-29 Utvidet lønn (skatt+AGA)", "tier": 3,
     "prompt": f"Kjør lønn for ansatt med grunnlønn 52000 kr, skattetrekk 38%, og arbeidsgiveravgift 14.1%. Registrer som lønnsbilag med separate bilag for lønn og arbeidsgiveravgift."},

    {"name": "T3-30 Regnskapsdimensjoner", "tier": 3,
     "prompt": f"Opprett en fri regnskapsdimensjon kalt 'Region' med verdiene 'Nord', 'Sør' og 'Vest'. Bokfør deretter 15000 kr på konto 6800 fordelt på region Nord."},

    # === NYE OPPGAVETYPER (lagt til etter gap-analyse) ===
    {"name": "T2-31 Leverandørbetaling", "tier": 2,
     "prompt": f"Betal leverandørfaktura til NattTest Leverandor AS på 12500 kr fra bankkonto."},

    {"name": "T2-32 Ansattutlegg", "tier": 2,
     "prompt": f"Ansatt har lagt ut 4200 kr for kontorrekvisita. Registrer ansattutlegget."},

    {"name": "T3-33 MVA-oppgjør", "tier": 3,
     "prompt": f"Registrer MVA-oppgjør for Q1 2026. Utgående MVA 95000 kr, inngående MVA 38000 kr. Betal differansen til skatteetaten."},

    {"name": "T2-34 Delbetaling", "tier": 2,
     "prompt": f"Kunden har betalt 10000 kr av en utestående faktura. Registrer delbetalingen via bank."},

    {"name": "T2-35 Privatperson som kunde", "tier": 2,
     "prompt": f"Opprett privatkunden Per Hansen {TS} med e-post per.hansen@privat.no. Han er privatperson, ikke bedrift."},

    {"name": "T3-36 Ansatt fra kontrakt", "tier": 3,
     "prompt": f"Opprett ansatt Maria Garcia, e-post maria.garcia.{TS}@firma.no, foedselsdato 1992-03-22. Hun starter 1. april 2026, 100% stilling, aarslonn 480000 kr, fast ansettelse med maanedslonn. Opprett baade ansatt og ansettelsesforhold."},
]

def run_task(task):
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
            timeout=180
        )
        elapsed = time.time() - start
        return r.status_code, elapsed
    except Exception as e:
        return f"ERROR: {e}", time.time() - start

print(f"=== FULL SIMULERING — {len(TASKS)} oppgaver (Tier 1/2/3) ===\n")

results = []
tier_results = {1: [], 2: [], 3: []}

for i, task in enumerate(TASKS):
    tier = task["tier"]
    print(f"[{i+1}/{len(TASKS)}] (T{tier}) {task['name']}...")
    status, elapsed = run_task(task)
    result = "OK" if status == 200 else f"FEIL ({status})"
    results.append((task["name"], tier, result, elapsed))
    tier_results[tier].append(result == "OK")
    print(f"  → {result} ({elapsed:.1f}s)")
    time.sleep(0.5)

print(f"\n{'='*60}")
print(f"RESULTATER")
print(f"{'='*60}\n")

for tier in [1, 2, 3]:
    ok = sum(tier_results[tier])
    total = len(tier_results[tier])
    mult = tier
    print(f"Tier {tier} (×{mult}): {ok}/{total} OK")

print()
for name, tier, result, elapsed in results:
    icon = "✅" if result == "OK" else "❌"
    print(f"  {icon} T{tier} {name}: {result} ({elapsed:.1f}s)")

ok_count = sum(1 for _, _, r, _ in results if r == "OK")
print(f"\n  TOTALT: {ok_count}/{len(results)} returnerte 200")
print(f"\n⚠️  200 = agenten svarte. Sjekk logger for 201 vs 422:")
print(f"  gcloud run services logs read tripletex-agent --region europe-north1 --limit 500")
