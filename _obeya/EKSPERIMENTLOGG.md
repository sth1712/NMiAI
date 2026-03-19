# EKSPERIMENTLOGG — NM i AI 2026

> Hypotesedrevet logg. Hver forbedring starter som en hypotese, testes med submission, og evalueres.
> Status: UTESTET | PÅGÅR | BEKREFTET | AVKREFTET | DELVIS

## Hypoteser

| # | Hypotese | Oppgave | Status | Resultat | Neste steg |
|---|---------|---------|--------|----------|------------|
| H001 | Gemini forstår norske Tripletex-prompts tilstrekkelig for Tier 1 | Tripletex | UTESTET | — | Kjør første Tier 1-submission |
| H002 | Pretrained YOLOv8n gir >30% detection mAP uten fine-tuning | NorgesGruppen | UTESTET | — | Last ned data, kjør inference, submit |
| H003 | initial_states + uniform gir >5 score på Astar Island | Astar Island | UTESTET | — | Test API, submit enkel prediksjon |
| H004 | Fine-tuning YOLOv8m på 248 bilder gir >50% detection mAP | NorgesGruppen | UTESTET | — | Avhenger av H002-resultat |
| H005 | Tripletex-agenten håndterer multi-steg oppgaver (kunde → order → faktura) | Tripletex | UTESTET | — | Test med sammensatt oppgave lokalt |
| H006 | Strategisk querying (rundt settlements) gir bedre Astar-prediksjon enn tilfeldig | Astar Island | UTESTET | — | Avhenger av H003-resultat |
| H007 | Detection-only submission scorer ~50% av maks (0.35 totalt) | NorgesGruppen | UTESTET | — | Submit kun detection uten segmentering |
| H008 | Effektivitetsbonus på Tripletex krever <5 API-kall per oppgave | Tripletex | UTESTET | — | Analyser scoring-kriterier nærmere |

## Avsluttede hypoteser

_Ingen ennå._
