# NM i AI 2026 — Klar til start (OPPDATERT med faktiske oppgaver)

> De gamle oppgavene (grocery bot, race car, etc.) var eksempler. Dette er de faktiske oppgavene.

---

## Oppgave 1: Tripletex — AI Accounting Agent ⭐ HØYEST PRIORITET

**Hvorfor først:** LLM-agent + API = raskest å score. Tier 1-oppgaver er enkle (opprett ansatt, kunde). Kan ha noe som scorer i løpet av timer.

### Hva trengs:
- [ ] Sett opp FastAPI /solve-endepunkt
- [ ] Koble til LLM (Claude API / Gemini) for å tolke norske prompts
- [ ] Implementer Tripletex API-kall (Basic Auth, username=0, password=session_token)
- [ ] Deploy som HTTPS-endepunkt (Cloud Run eller cloudflared tunnel)
- [ ] Hent sandbox-konto fra app.ainm.no for testing
- [ ] Test med enkel oppgave (opprett ansatt)
- [ ] Submit endepunkt-URL

### Arkitektur:
```
POST /solve
  → Motta prompt + credentials
  → Send prompt til LLM: "Tolk denne regnskapsoppgaven og returner API-kall"
  → Utfør API-kall mot Tripletex proxy
  → Returner {"status": "completed"}
```

### Nøkkel-API-endepunkter:
- `/employee` — ansatte
- `/customer` — kunder
- `/product` — produkter
- `/invoice` — fakturaer
- `/order` — ordrer
- `/travelExpense` — reiseregninger
- `/project` — prosjekter
- `/department` — avdelinger

### Scoring:
- Tier 1 (×1): Enkel — opprett ansatt, kunde. Tilgjengelig NÅ.
- Tier 2 (×2): Multi-steg — faktura med betaling. Åpner fredag.
- Tier 3 (×3): Kompleks — bankavstemminger etc. Åpner lørdag.
- Perfekt score + effektivitet kan gi opptil 6.0 per oppgave.
- Best score per oppgave beholdes (dårlige runs teller ikke).

---

## Oppgave 2: NorgesGruppen Data — Object Detection

**Hvorfor andre:** Trenger GPU-trening. Men detection-only (ignorer kategori) gir 70% — realistisk med pretrained YOLOv8.

### Hva trengs:
- [ ] Last ned treningsdata (864 MB COCO-dataset + 60 MB produktbilder)
- [ ] Fine-tune YOLOv8 (n/s/m/l) på datasettet (nc=356 for klassifisering, eller ignorer for detection-only)
- [ ] Skriv run.py som tar --input og --output
- [ ] Pakk som .zip (run.py + modellvekter, maks 420MB)
- [ ] Test lokalt
- [ ] Submit

### Quick Detection-Only (70%):
- Bruk pretrained YOLOv8 COCO-modell → setter category_id: 0 for alt
- Fine-tune kort på datasettet for bedre deteksjon
- Scorer opptil 70% uten å identifisere produkter

### Full Score (100%):
- Fine-tune YOLOv8 med nc=356 på treningsdataene
- Krever mer trening men gir 30% ekstra

### Sandbox-miljø:
- NVIDIA L4 GPU, 24GB VRAM
- Python 3.11, PyTorch 2.6, ultralytics 8.1.0
- INGEN nettverkstilgang
- 300 sek timeout, 8GB RAM
- Pin ultralytics==8.1.0 lokalt!

---

## Oppgave 3: Astar Island — Norse World Prediction

**Hvorfor tredje:** Mest kompleks. Men en enkel baseline (initial_states + uniform) er bedre enn 0.

### Hva trengs:
- [ ] Hent JWT-token fra app.ainm.no
- [ ] Hent aktiv runde og initial_states
- [ ] Submit uniform baseline for alle 5 seeds (scorer 1-5, men bedre enn 0)
- [ ] Bygg smartere prediksjoner basert på simulator-queries
- [ ] Bruk 50 queries strategisk

### Quick Baseline:
```python
import numpy as np

# Fra initial_states: sett kjente terrengtyper til høy sannsynlighet
# Mountain (5) → [0.01, 0.01, 0.01, 0.01, 0.01, 0.95]
# Ocean/Plains → [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
# Forest (4) → [0.01, 0.01, 0.01, 0.01, 0.95, 0.01]
# Settlement → query simulator for å se hva som skjer
# ALDRI sett 0.0 — bruk minimum 0.01 og renormaliser!
```

### Scoring:
- KL-divergens, entropivektet. 0-100 skala.
- Statiske celler (hav, fjell) teller nesten ikke (lav entropi).
- Dynamiske celler (settlements, ports, ruins) er der poengene er.
- Uniform baseline ≈ 1-5 poeng. God modell ≈ 70-90.

---

## Totalstrategi

| Tid | Fokus |
|-----|-------|
| **Kveld 19. mars** | Sett opp Tripletex-agent (FastAPI + LLM). Submit Astar baseline. |
| **Natt/morgen 20. mars** | Tripletex Tier 1 scoring. Start NorgesGruppen data-nedlasting og trening. |
| **Fredag 21. mars** | Tripletex Tier 2 åpner. Forbedre agent. NorgesGruppen submit. Forbedre Astar. |
| **Lørdag 22. mars** | Tripletex Tier 3. Finjustere alt. Siste submissions før kl 15:00. |

---

## Server-info (fyll inn)

| Oppgave | Endepunkt | Token/Auth |
|---------|-----------|------------|
| Tripletex proxy | https://tx-proxy.ainm.no/v2 | Basic Auth (0, session_token) |
| Astar Island API | https://api.ainm.no/astar-island/ | Bearer JWT |
| NorgesGruppen | Kode-upload via app.ainm.no | — |
| Sandbox Tripletex | (fyll inn fra app.ainm.no) | |
