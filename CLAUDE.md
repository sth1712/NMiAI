# CLAUDE.md — NM i AI 2026

## Hvem er Sander
- Bachelorstudent ved HiØ, bygger KI-konsulenthuset **Digliate** (ansvarlig KI-implementering for norske SMBer)
- Deltaker i **NM i AI 2026** (Norges mesterskap i kunstig intelligens), 19–22. mars 2026
- Premiepott: 1 MNOK. Konkurransen består av 4 oppgaver (se under).
- Sander har et 7-agentsystem for bachelorprosjektet, men det er IKKE relevant for dette repoet — her handler det kun om NM i AI.

## Språk
- Skriv på **norsk bokmål** i kommunikasjon med Sander. Bruk æ, ø, å.
- Kode og kommentarer i kode kan være på engelsk.

## Repostruktur

```
NMiAI/
├── CLAUDE.md              ← Denne filen (les FØRST)
├── MOBIL_INBOX.md         ← Sander skriver instrukser/info fra telefonen her
├── KLAR_TIL_START.md      ← Status per oppgave, sjekklister, prioritering
└── workspace/
    ├── README.md          ← Teknisk oversikt over alle oppgaver
    ├── cheatsheet/        ← To tekniske cheat sheets med strategier og kode
    ├── grocery_bot/       ← Oppgave 1: WebSocket matbutikk-bot
    ├── race_car/          ← Oppgave 2: Regelbasert kjøreagent
    ├── healthcare_rag/    ← Oppgave 3: Medisinsk NLI/RAG-system
    └── tumor_segmentation/← Oppgave 4: U-Net tumorsegmentering
```

## Viktige filer å lese

1. **`MOBIL_INBOX.md`** — Sjekk denne FØRST. Sander legger inn instrukser, server-URLer, tokens og info fra kickoff her.
2. **`KLAR_TIL_START.md`** — Komplett statusoversikt per oppgave med sjekklister og prioritering.
3. **`workspace/README.md`** — Teknisk dokumentasjon og kjørekommandoer for alle 4 oppgaver.
4. **`workspace/cheatsheet/`** — To detaljerte cheat sheets med strategier, kodeeksempler og optimalisering.

## De 4 oppgavene

### 1. Grocery Bot (`workspace/grocery_bot/`)
- **Hva:** WebSocket-bot som handler i en virtuell matbutikk. Navigerer et grid, plukker varer, leverer ordrer.
- **Teknologi:** A* pathfinding, multi-bot koordinering, kollisjonsunngåelse
- **Status:** Fungerende kode, MEN ikke testet mot live server. A*-implementasjonen har en ytelsessvakhet (kopierer hele stien per node — bør bruke came_from-dict).
- **Pre-game:** Denne oppgaven er LIVE — gir poeng allerede nå.
- **Filer:** `bot.py` (WebSocket + hovedloop), `pathfinding.py` (A*), `strategy.py` (ordretildeling)
- **Kjør:** `JWT_TOKEN="..." python3 bot.py`

### 2. Race Car (`workspace/race_car/`)
- **Hva:** KI-agent med 16 sensorer som skal kjøre lengst mulig på 1 minutt.
- **Teknologi:** Regelbasert sensorlogikk med sektorer og EMA-glatting. Cheat sheet har FSM og PID-varianter.
- **Status:** Baseline fungerer, men mangler hastighetssensitiv styring.
- **Filer:** `bot.py` (tilkobling), `agent.py` (kjørelogikk)
- **Kjør:** `JWT_TOKEN="..." python3 bot.py`

### 3. Emergency Healthcare RAG (`workspace/healthcare_rag/`)
- **Hva:** Klassifiserer medisinske påstander som sant/usant + tildeler ett av 115 temaer.
- **Teknologi:** NLI (DeBERTa) → LLM-fallback → regelbasert kaskade. FAISS for retrieval.
- **Status:** Arkitektur klar, MEN kunnskapsbasen har kun 3 eksempeldokumenter. Trenger riktig medisinsk data for å fungere.
- **Filer:** `rag_pipeline.py` (FAISS-indeks), `classifier.py` (NLI/LLM kaskade + server)
- **Kjør:** `OPENAI_API_KEY="..." python3 classifier.py --server`
- **ML-modeller:** `cross-encoder/nli-deberta-v3-base` og `paraphrase-multilingual-MiniLM-L12-v2` er allerede lastet ned til lokal cache.

### 4. Tumor Segmentation (`workspace/tumor_segmentation/`)
- **Hva:** Segmentere svulster i MIP-PET-bilder. Maks 10 sek/bilde.
- **Teknologi:** MONAI U-Net med pretrained fallback-hierarki.
- **Status:** Minimal — trenger pretrained vekter for å gi verdi. LAVEST prioritet.
- **Filer:** `bot.py` (server), `model.py` (U-Net), `inference.py` (TTA + postprocessing)
- **Kjør:** `JWT_TOKEN="..." python3 bot.py`

## Prioriteringsrekkefølge under konkurransen

1. **Grocery Bot** — allerede live, gir poeng NÅ. Høyest ROI.
2. **Healthcare RAG** — høy ROI hvis kunnskapsbase bygges. NLI-tilnærmingen er teknisk riktig.
3. **Race Car** — finjuster regelbasert agent. PID-kontroller og fartsjustering kan gi mye.
4. **Tumor Segmentation** — maks 30 min. Kun verdt det med pretrained modell.

## Teknisk miljø
- **OS:** macOS (Darwin)
- **Python:** 3.13
- **Alle pip-avhengigheter:** Installert for alle 4 oppgaver.
- **ML-modeller cached:** DeBERTa NLI + multilingual MiniLM embedding.
- **API-nøkler trengs:** `JWT_TOKEN` (fra app.ainm.no), `OPENAI_API_KEY` eller `ANTHROPIC_API_KEY`

## Arbeidsflyt

1. Les `MOBIL_INBOX.md` for nye instrukser fra Sander
2. Les `KLAR_TIL_START.md` for status og sjekklister
3. Gjør det Sander ber om — kodeendringer, forberedelser, testing
4. Commit og push endringer til GitHub slik at alt er synkronisert

## Vanlige oppgaver Sander kan be om
- Fiks/forbedre kode i en spesifikk oppgave
- Bygg kunnskapsbase for Healthcare RAG
- Test en bot mot serveren
- Oppdater strategi basert på ny info fra konkurransen
- Legg inn server-URLer, tokens eller nye regler
