# NM i AI 2026 — Klar til start

> Oppsummering av alt som er klart + hva som gjenstår.
> Oppdatert: 19. mars 2026.

---

## Status per oppgave

### 1. Grocery Bot (PRE-GAME — LIVE NÅ)
- **Kode:** `workspace/grocery_bot/` — fungerende bot med A* pathfinding
- **Status:** Kan kjøres, men ikke testet mot live server enda
- **Kjør:** `cd workspace/grocery_bot && python3 main.py`
- **Trenger:**
  - [ ] JWT-token fra app.ainm.no (logg inn → hent fra nettleseren)
  - [ ] Teste mot live server — verifiser at game_state-formatet stemmer
  - [ ] Vurder å fikse A* (bruker listekopi per node — bytt til came_from-dict)

### 2. Race Car
- **Kode:** `workspace/race_car/` — regelbasert agent med sensorlogikk
- **Status:** Baseline klar
- **Kjør:** `cd workspace/race_car && python3 main.py`
- **Trenger:**
  - [ ] Server-URL og token
  - [ ] Teste mot server
  - [ ] Vurder PID-kontroller og hastighetssensitiv styring

### 3. Healthcare RAG
- **Kode:** `workspace/healthcare_rag/` — NLI + LLM kaskade
- **Status:** Arkitektur klar, men MANGLER kunnskapsbase
- **Kjør:** `cd workspace/healthcare_rag && python3 main.py`
- **Trenger:**
  - [ ] Bygge kunnskapsbase med 115 nødhelse-temaer
  - [ ] Laste ned NLI-modell (cross-encoder/nli-deberta-v3-base) til cache
  - [ ] API-nøkkel (OPENAI_API_KEY eller ANTHROPIC_API_KEY)

### 4. Tumor Segmentation
- **Kode:** `workspace/tumor_segmentation/` — U-Net med MONAI
- **Status:** Minimal — trenger pretrained modell for å gi verdi
- **Kjør:** `cd workspace/tumor_segmentation && python3 main.py`
- **Trenger:**
  - [ ] Pretrained vekter (nnU-Net eller TotalSegmentator)
  - [ ] Testbilder
- **Prioritet:** LAV — bruk maks 30 min under konkurransen

---

## Prioritert rekkefølge under konkurransen

1. **Grocery Bot** — allerede live, gir poeng NÅ
2. **Healthcare RAG** — høy ROI med riktig kunnskapsbase
3. **Race Car** — finjuster regelbasert agent
4. **Tumor Seg** — minimal innsats, lav forventning

---

## Teknisk sjekkliste (gjør FØR konkurransen)

- [ ] Alle pip-avhengigheter installert
- [ ] NLI-modell lastet ned til cache
- [ ] API-nøkler satt som miljøvariabler
- [ ] JWT-token hentet fra app.ainm.no
- [ ] Grocery Bot testet mot live server
- [ ] tmux/screen satt opp for 4 parallelle bots
- [ ] Git commit av nåværende state

---

## Server-info (fyll inn fra kickoff)

| Oppgave | Server-URL | API-endepunkt | Tidsfrist per request |
|---------|-----------|---------------|----------------------|
| Grocery Bot | | | 2 sek |
| Race Car | | | sanntid |
| Healthcare RAG | | | |
| Tumor Seg | | | < 10 sek/bilde |

---

## Nyttige kommandoer

```bash
# Installer alt
cd /Users/sanderholm/NMiAI/workspace
for d in grocery_bot race_car healthcare_rag tumor_segmentation; do
    cd $d && pip3 install -r requirements.txt && cd ..
done

# Last ned NLI-modell
python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-base')"

# Last ned embedding-modell
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Sett API-nøkler
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```
