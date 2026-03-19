# CLAUDE.md — NM i AI 2026

## Hvem er Sander
- Bachelorstudent ved HiØ, bygger KI-konsulenthuset **Digliate**
- Deltaker i **NM i AI 2026**, 19–22. mars 2026 (69 timer)
- Premiepott: 1 MNOK. Totalpoeng = gjennomsnitt av normalisert score på alle 3 oppgaver (33% hver).
- NM er en stresstest av Missing Middle-konseptet: 1 person + agentsystem vs. 4-persons lag uten system.

## Språk
- **Norsk bokmål** i kommunikasjon. Bruk æ, ø, å — aldri ae/oe/aa.
- Kode og kommentarer kan være på engelsk.

## Agentsystem (7 agenter)

Systemet er inspirert av Sanders bachelorprosjekt-agentsystem (v9.0 LEAN Orchestra). Les agentfilen FØR du tar en rolle.

| # | Agent | Rolle | Fil |
|---|-------|-------|-----|
| 00 | **Kommandosentral** | COO — Orkestrator, Obeya-forvalter, «hva nå?» | `_agenter/00_KOMMANDOSENTRAL.md` |
| 01 | **Tripletex-spesialist** | CTO — Eier agentens kode, API, deployment | `_agenter/01_TRIPLETEX.md` |
| 02 | **NorgesGruppen-spesialist** | CTO — YOLOv8 pipeline, run.py, modelloptimering | `_agenter/02_NORGESGRUPPEN.md` |
| 03 | **Astar Island-spesialist** | CTO — Query-strategi, prediksjonsmodell | `_agenter/03_ASTAR_ISLAND.md` |
| 04 | **Analytiker** | Metode — Resultatanalyse, hypotesetesting, ROI | `_agenter/04_ANALYTIKER.md` |
| 05 | **Researcher** | Research — Ekstern kunnskap, leaderboard, teknikker | `_agenter/05_RESEARCHER.md` |
| 06 | **Strateg** | CEO — Overordnet retning, djevelens advokat | `_agenter/06_STRATEG.md` |

**Sander er den 7. agenten** — Missing Middle-agenten. Intuisjon, sosial intelligens, beslutninger, kvalitetssikring.

## Obeya-rom (delt bevissthet)

| Fil | Innhold | Oppdateres |
|-----|---------|------------|
| `_obeya/AKTIV_KONTEKST.md` | Sanntidsstatus, scores, kvoter, neste handling | Per syklus (~30 min) |
| `_obeya/KAMPLOGG.md` | Alle submissions med hypotese → resultat → lærdom | Per submission |
| `_obeya/EKSPERIMENTLOGG.md` | Hypoteser: formulert → testet → bekreftet/avkreftet | Per eksperiment |
| `_obeya/LEADERBOARD_TRACKER.md` | Vår posisjon + konkurrentanalyse | Flere ganger daglig |

## Sessionstart
1. Les `_obeya/AKTIV_KONTEKST.md` — den gir nåtilstand
2. Les relevant agentfil i `_agenter/`
3. Sjekk `MOBIL_INBOX.md` for instrukser fra Sander

## Design-prinsipper
1. **Hver submission er et eksperiment** — hypotese FØR, analyse ETTER
2. **Dobbel output** — alt produserer (1) poeng og (2) bachelor/Digliate-materiale
3. **Obeya-tempo** — AKTIV_KONTEKST oppdateres per syklus, ikke per dag
4. **Missing Middle** — Sander bestemmer, Claude rådgir og utfører

## MCP Docs Server
```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```

## De 3 oppgavene

### 1. NorgesGruppen Data: Object Detection
- **Type:** Kode-opplasting (.zip med run.py + modellvekter)
- **Metrikk:** mAP@0.5 — 70% deteksjon + 30% klassifikasjon
- **Sandbox:** NVIDIA L4 GPU, Python 3.11, ultralytics 8.1.0, INGEN nett
- **Kvote:** 3/dag, maks 2 in-flight

### 2. Tripletex: AI Accounting Agent
- **Type:** HTTPS /solve-endepunkt. Mottar prompt → utfør API-kall.
- **Metrikk:** Felt-for-felt × tier-multiplikator + effektivitetsbonus
- **Live:** https://tripletex-agent-421519138388.europe-north1.run.app
- **Tier 1** (×1) nå, **Tier 2** (×2) fredag, **Tier 3** (×3) lørdag

### 3. Astar Island: Norse World Prediction
- **Type:** REST API — query simulator, submit sannsynlighetsfordelinger
- **Metrikk:** KL-divergens (entropivektet), 0-100
- **50 queries** per runde, delt på 5 seeds

## Google Cloud Platform (GRATIS)
- **GCP prosjekt:** ainm26osl-705 (konto: devstar7051@gcplab.me)
- **Cloud Run** — Tripletex-agent deployment
- **Compute Engine** — GPU for YOLOv8-trening
- **Gemini API** — Gratis LLM-backbone
- **Cloud Shell** — Terminal i nettleseren

## Viktig
- Gammel kode i `workspace/` er IRRELEVANT (eksempeloppgaver som ikke ble brukt)
- Daglig kvote resetter **midnatt UTC (01:00 norsk tid)**
- Best score per oppgave beholdes — dårlige runs teller aldri mot deg
- **ALDRI** sett sannsynlighet til 0.0 i Astar Island — bruk minimum 0.01
