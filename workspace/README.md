# NM i AI 2026 — Arbeidsmappe

Fire oppgaver. Hver oppgave har egen mappe med kjørbar kode og requirements.txt.

---

## Hurtigstart

```bash
# Installer alle avhengigheter
for d in grocery_bot race_car healthcare_rag tumor_segmentation; do
    cd $d && pip3 install -r requirements.txt && cd ..
done

# Sett JWT-token (felles for alle oppgaver)
export JWT_TOKEN="ditt_token_her"
```

---

## 1. Grocery Bot (`grocery_bot/`)

WebSocket-bot som spiller automatisk supermarked. Støtter enkelt- og multi-bot-modus.

**Kjør:**
```bash
export JWT_TOKEN="ditt_token_her"
python3 bot.py
```

**Filer:**
- `bot.py` — WebSocket-tilkobling, hovedløkke, multi-bot-svar
- `pathfinding.py` — A*-algoritme med Manhattan-heuristikk
- `strategy.py` — Multi-bot ordretildeling, kollisjonsunngåelse, forbedret scoring

**Funksjoner:**
- Multi-bot koordinering (tildeler forskjellige ordrer til forskjellige bots)
- Kollisjonsunngåelse mellom bots (dynamiske hindringer)
- Forbedret ordrescoring (full rutelengde, nesten-ferdig-bonus, inventarbonus)

---

## 2. Race Car (`race_car/`)

Regelbasert kjøreagent med 16 sensorer. WebSocket- og HTTP-tilkobling.

**Kjør:**
```bash
export JWT_TOKEN="ditt_token_her"
python3 bot.py                    # WebSocket (standard)
MODE=http python3 bot.py          # HTTP-polling
python3 agent.py                  # Lokal test
```

**Filer:**
- `bot.py` — WebSocket/HTTP-tilkobling med reconnect og tidsmåling
- `agent.py` — Regelbasert sensorlogikk med sektorer og glatting

---

## 3. Emergency Healthcare RAG (`healthcare_rag/`)

Klassifiserer medisinske påstander som sant/usant + ett av 115 temaer.
Tre strategier: NLI (sterkest) → LLM-fallback → Regelbasert.

**Kjør:**
```bash
python3 rag_pipeline.py                   # Bygg FAISS-indeks
python3 classifier.py                     # Lokal test
python3 classifier.py --server            # Koble til server (WebSocket)
python3 classifier.py --server --url=ws://...  # Egendefinert URL
```

**Filer:**
- `rag_pipeline.py` — FAISS-indeks med sentence-transformers (flerspråklig)
- `classifier.py` — NLI/LLM/regelbasert kaskade + server-tilkobling

**Miljøvariabler:**
- `OPENAI_API_KEY` eller `ANTHROPIC_API_KEY` for LLM-fallback
- `JWT_TOKEN` for server-autentisering

---

## 4. Tumor Segmentation (`tumor_segmentation/`)

U-Net segmentering av svulster i MIP-PET-bilder. Mål: < 10 sek/bilde.

**Kjør:**
```bash
export JWT_TOKEN="ditt_token_her"
python3 bot.py                     # Koble til server (WebSocket)
python3 bot.py --http              # HTTP-modus
python3 bot.py --no-tta            # Uten test-time augmentation
python3 inference.py bilde.dcm     # Segmenter lokalt bilde
python3 model.py                   # Verifiser arkitektur
```

**Filer:**
- `bot.py` — Server-integrasjon (WebSocket/HTTP), base64/RLE-output
- `model.py` — MONAI U-Net med pretrained fallback-hierarki
- `inference.py` — TTA, terskeloptimering, morfologisk etterbehandling

**Modell-fallback:** egne vekter → MONAI Model Zoo → HuggingFace → tilfeldig init

---

## Avhengigheter per oppgave

| Oppgave | Nøkkelavhengigheter |
|---------|---------------------|
| Grocery Bot | websockets |
| Race Car | numpy, websockets, aiohttp |
| Healthcare RAG | sentence-transformers, faiss-cpu, openai/anthropic |
| Tumor Segmentation | torch, monai, nibabel, pydicom, scipy |
