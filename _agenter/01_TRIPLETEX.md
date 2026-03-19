# Agent 01: Tripletex — AI Accounting Agent

## Rolle
Tolker norske regnskapsprompts og utfører Tripletex API-kall. Hosted som HTTPS-endepunkt.

## Filer
- `tripletex-agent/main.py` — FastAPI + Gemini-agent
- `tripletex-agent/Dockerfile` — Container for Cloud Run
- `tripletex-agent/requirements.txt` — fastapi, uvicorn, requests, google-genai

## Live
- **URL:** https://tripletex-agent-421519138388.europe-north1.run.app
- **Endepunkt:** POST /solve
- **Health:** GET /health

## Arbeidsflyt
1. Motta POST /solve med `prompt`, `files`, `tripletex_credentials`
2. Gemini tolker prompt → returnerer JSON-array med API-kall
3. Utfør kall sekvensielt mot Tripletex proxy (base_url fra request)
4. Ved 4xx-feil: Gemini forsøker å fikse kallet automatisk
5. Returner `{"status": "completed"}`

## Scoring
- **Felt-for-felt:** Hvert felt i forventet output sjekkes
- **Tier-multiplikator:** Tier 1 (x1), Tier 2 (x2), Tier 3 (x3)
- **Effektivitetsbonus:** Færre API-kall + null 4xx = opptil 2x tier-score
- **Maks per oppgave:** 6.0 poeng (perfekt + maks effektivitet)
- 30 oppgavetyper, 56 varianter (7 språk x 8 datasett). Best run per variant beholdes.

## Kjente oppgavetyper og API-mønstre

### Tier 1 (x1) — Tilgjengelig nå
| Oppgave | API-kall |
|---------|----------|
| Opprett ansatt | POST /employee `{firstName, lastName, email}` |
| Opprett kunde | POST /customer `{name, email, isCustomer: true}` |
| Opprett leverandør | POST /supplier `{name, email, isSupplier: true}` |
| Opprett produkt | POST /product `{name, number, priceExcludingVat}` |
| Opprett avdeling | POST /department `{name, departmentNumber}` |
| Opprett prosjekt | POST /project `{name, number, projectManagerId}` |

### Tier 2 (x2) — Åpner fredag
| Oppgave | API-kall |
|---------|----------|
| Faktura | POST /order → POST /invoice `{orderId}` |
| Betaling | POST /payment + koble til faktura |
| Reiseregning | POST /travelExpense med kostnader |

### Tier 3 (x3) — Åpner lørdag
- Bankavstemminger, komplekse bilag, multi-entitet-operasjoner

## Auth
- Basic Auth: username=`0`, password=`session_token` (fra request body)
- base_url kommer i request (Tripletex proxy)

## Deploy
```bash
cd /Users/sanderholm/NMiAI/tripletex-agent
gcloud run deploy tripletex-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --timeout 300
```

## Testing mot sandbox
```bash
# Hent sandbox-credentials fra app.ainm.no
curl -X POST https://tripletex-agent-421519138388.europe-north1.run.app/solve \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Opprett en ansatt med navn Test Testesen, e-post test@test.no",
    "files": [],
    "tripletex_credentials": {
      "base_url": "SANDBOX_BASE_URL",
      "session_token": "SANDBOX_TOKEN"
    }
  }'
```

## Forbedringsstrategi

### 1. Systemprompt-iterasjon
Nåværende systemprompt i main.py er generisk. Forbedre med:
- Eksakte feltnavn per endepunkt (fra API-docs)
- Obligatoriske felt som mangler (f.eks. `dateOfBirth` for ansatte?)
- Vanlige fallgruver per oppgavetype

### 2. Oppskriftsbank
Bygg en mapping i systemprompt: oppgavetype → eksakt API-sekvens.
Reduserer Gemini-hallusinering og gir færre kall (bedre effektivitet).

### 3. Feilmønster-analyse
Etter submission: sjekk Cloud Run-logger for 4xx-feil.
```bash
gcloud run services logs read tripletex-agent --region europe-north1 --limit 50
```
Identifiser gjentakende feilmønstre og legg til fixes i systemprompt.

### 4. Retry-logikk
Nåværende: Gemini forsøker å fikse 400/422-feil. Forbedre med:
- Spesifikke fixes for kjente feilmeldinger
- Fallback-strategier per endepunkt

## Eksperiment-format
```
HYPOTESE: [Endring i systemprompt/kode] vil forbedre [oppgavetype]
ENDRING: [Hva ble gjort]
RESULTAT: [Score før → etter, feilrate]
BEHOLDE: Ja/Nei
```

## Sjekkliste daglig
- [ ] Sjekk scoreboard på app.ainm.no
- [ ] Les logger for feilmønstre
- [ ] Oppdater systemprompt basert på funn
- [ ] Deploy ny versjon
- [ ] Bruk 3 submissions strategisk (test viktigste endring)
