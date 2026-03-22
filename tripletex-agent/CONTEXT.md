# Tripletex Agent — Kontekst for QA og utvikling

## Hva er dette?
AI-agent for NM i AI 2026 som løser regnskapsoppgaver i Tripletex.
Mottar norsk prompt → Gemini planlegger API-kall → utfører mot Tripletex → returnerer "completed".

## Regler (fra app.ainm.no)
- Endpoint: HTTPS, POST /solve (og POST /)
- Timeout: 5 minutter (300 sekunder)
- MUST returnere {"status": "completed"} med HTTP 200
- Alle Tripletex API-kall MÅ gå via base_url fra request (proxy)
- Auth: Basic Auth, username "0", password = session_token fra request
- AI-verktøy (Claude, Gemini, Copilot) er EKSPLISITT tillatt
- Koden MÅ være public på GitHub for premie-eligibilitet (MIT-lisens)

## Scoring
- 30 oppgavetyper, 56 varianter per type (7 språk × 8 datasett)
- Felt-for-felt verifisering × tier-multiplikator + effektivitetsbonus
- Tier 1 (×1): Enkel CRUD. Tier 2 (×2): Multi-steg. Tier 3 (×3): Kompleks.
- Effektivitetsbonus: Færre API-kall + null 4xx-feil = opptil 2× tier-score
- Best score per oppgavetype beholdes — dårlige runs koster ingenting
- Total = sum av best scores per type
- 10 submissions per dag

## Arkitektur
- FastAPI på Cloud Run (europe-north1)
- Gemini 2.5 Flash som LLM (via API-nøkkel, IKKE Vertex AI)
- Pre-flight: Henter department_id, company_id, paymentType IDs, setter bankkontonummer
- Environment-verdier injiseres i Gemini-prompt for å unngå unødvendige GET-kall

## Kritiske regler for kodeendringer
1. SYSTEM_PROMPT MÅ ha eksakte feltnavn fra Tripletex API (LLM hallusinerer uten)
2. Pre-flight auto-setup MÅ kjøre FØR Gemini kalles (setter bankkontonummer etc.)
3. Environment-verdier MÅ injiseres i prompt (department_id, company_id varierer per sandbox)
4. Aldri hardkod IDer — de endres mellom sandbox-instanser
5. POST /ledger/voucher postings MÅ ha "row"-felt (ellers "systemgenererte"-feil)
6. GET /invoice krever invoiceDateFrom + invoiceDateTo (422 uten)
7. PUT /:invoice, /:payment, /:createCreditNote bruker params, ikke body
8. Leverandør = POST /supplier, IKKE POST /customer med isSupplier
9. Employee email MÅ være unik
10. PUT employee krever dateOfBirth

## Kjente svakheter (fra testing)
- Oppdatering av eksisterende entiteter: Søk-og-oppdater feiler med mange like-navnede
- Leverandørfaktura: Krever voucher med postings med row + supplier.id per posting
- Lønnskjøring: salary/transaction gir 403 (token-begrensning)
- Sletting basert på søk: Tittelsøk på travelExpense er ikke robust

## Test
```bash
python3 test_sandbox.py          # 7 grunnleggende tester
python3 test_15_tasks.py         # 15 oppgavetyper (hvis finnes)
```

## Deploy
```bash
bash deploy.sh DIN_GEMINI_API_NOEKKEL
```
