# CLAUDE.md — NM i AI 2026

## Hvem er Sander
- Bachelorstudent ved HiØ, bygger KI-konsulenthuset **Digliate**
- Deltaker i **NM i AI 2026**, 19–22. mars 2026 (69 timer)
- Premiepott: 1 MNOK. Totalpoeng = gjennomsnitt av normalisert score på alle 3 oppgaver (33% hver).

## Språk
- **Norsk bokmål** i kommunikasjon. Bruk æ, ø, å.
- Kode og kommentarer kan være på engelsk.

## MCP Docs Server
```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```
Bruk denne for å slå opp detaljer i offisiell dokumentasjon.

## De 3 oppgavene

### 1. NorgesGruppen Data: Object Detection
- **Hva:** Detekter og klassifiser dagligvareprodukter på butikkhyller
- **Type:** Kode-opplasting (.zip med run.py + modellvekter)
- **Metrikk:** mAP@0.5 — 70% deteksjon + 30% klassifikasjon
- **Sandbox:** NVIDIA L4 GPU (24GB VRAM), Python 3.11, PyTorch 2.6, ultralytics 8.1.0, INGEN nettverkstilgang
- **Treningsdata:** 248 bilder, ~22.700 bboxer, 356 produktkategorier (COCO-format). Også produktbilder (327 produkter, multi-vinkel).
- **Tidsfrist per submission:** 300 sekunder
- **Daglig kvote:** 3 submissions/dag, maks 2 in-flight
- **Nøkkelinnsikt:** Detection-only (category_id: 0 for alt) gir opptil 70%. Fine-tuning YOLOv8 på treningsdataene gir resten.
- **Quick win:** Last ned data, fine-tune YOLOv8n/s/m på datasettet, eksporter, submit.

### 2. Tripletex: AI Accounting Agent
- **Hva:** Bygg en AI-agent som utfører regnskapsoppgaver via Tripletex API
- **Type:** Host et HTTPS /solve-endepunkt. De sender oppgaver, du utfører API-kall.
- **Metrikk:** Felt-for-felt verifisering × tier-multiplikator + effektivitetsbonus
- **30 oppgaver**, 56 varianter per oppgave (7 språk × 8 datasett)
- **Tier 1** (×1): Enkle oppgaver (opprett ansatt, kunde). Tilgjengelig fra start.
- **Tier 2** (×2): Multi-steg (faktura med betaling). Åpner fredag tidlig.
- **Tier 3** (×3): Komplekse scenarioer. Åpner lørdag tidlig.
- **Timeout:** 5 minutter per oppgave
- **Nøkkelinnsikt:** LLM-agent som tolker norsk prompt → mapper til API-kall. Effektivitet belønnes (færre kall, null feil).
- **Quick win:** FastAPI + LLM (Claude/Gemini) som tolker prompts og kaller Tripletex API.

### 3. Astar Island: Norse World Prediction
- **Hva:** Observer en norrøn sivilisasjonssimulator gjennom begrenset viewport, prediker sluttilstanden
- **Type:** REST API — query simulator, submit sannsynlighetsfordelinger
- **Metrikk:** KL-divergens (entropivektet), 0-100 skala
- **40×40 kart, 5 seeds, 50 queries per runde** (delt på alle seeds)
- **Viewport:** Maks 15×15 per query
- **Prediksjonsformat:** H×W×6 tensor med sannsynligheter for 6 terrengklasser
- **6 klasser:** Empty (0), Settlement (1), Port (2), Ruin (3), Forest (4), Mountain (5)
- **VIKTIG:** Aldri sett sannsynlighet til 0.0 — bruk minimum 0.01 og renormaliser
- **Nøkkelinnsikt:** Statiske celler (hav, fjell, skog) kan utledes fra initial_states. Dynamiske celler (settlements, ports, ruins) er det som scorer.
- **Quick win:** Hent initial_states, sett kjente celler til ~1.0, ellers uniform. Submit for alle 5 seeds.

## Prioritering

1. **Tripletex** — Raskest å score. LLM-agent + API. Kan starte scoring på Tier 1 umiddelbart.
2. **NorgesGruppen** — Trenger GPU-trening. YOLOv8 fine-tuning. Detection-only gir 70%.
3. **Astar Island** — Submit uniform baseline ASAP (scores 1-5, men bedre enn 0). Forbedre gradvis.

## Google Cloud Platform (GRATIS — ingen kredittkort, ingen grenser)
Sander har fått en dedikert GCP-konto (@gcplab.me) via NM-partnerskap med Google. Ingen kreditgrenser.

**Tilgjengelige tjenester:**
- **Cloud Run** — Deploy Tripletex-agenten som HTTPS-endepunkt (perfekt for /solve)
- **Compute Engine** — GPU VMs for NorgesGruppen-trening (YOLOv8 fine-tuning)
- **Vertex AI / Model Garden** — Pretrained modeller
- **Gemini models & AI Studio** — Gratis Gemini API (bruk som LLM i Tripletex-agent!)
- **Cloud Shell & VS Code IDE** — Kan utvikle direkte i nettleseren

**Strategisk bruk:**
- **Tripletex-agent → Cloud Run** (deploy FastAPI som HTTPS-endepunkt, null config)
- **NorgesGruppen-trening → Compute Engine med GPU** (YOLOv8 fine-tuning)
- **LLM-backbone → Gemini API** (gratis, erstatter betalt Claude/OpenAI-avhengighet)

## Teknisk miljø
- **OS:** macOS (Darwin), Python 3.13
- **GCP:** Dedikert prosjekt med @gcplab.me-konto. Cloud Run, Compute Engine, Vertex AI, Gemini, Cloud Shell.
- **API-nøkler trengs:** JWT fra app.ainm.no, Gemini API-nøkkel (fra AI Studio), evt. ANTHROPIC_API_KEY

## Arbeidsflyt
1. Les `MOBIL_INBOX.md` for instrukser fra Sander
2. Les `KLAR_TIL_START.md` for status og sjekklister
3. Gjør det Sander ber om
4. Commit og push endringer

## VIKTIG: Gammel kode er IRRELEVANT
Mappene `workspace/grocery_bot/`, `workspace/race_car/`, `workspace/healthcare_rag/`, `workspace/tumor_segmentation/` var basert på EKSEMPELOPPGAVER som ikke ble brukt. De faktiske oppgavene er helt andre. Ignorer gammel kode.
