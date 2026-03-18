# NM i AI 2026 — Arbeidsmappe

Fire oppgaver. Hver oppgave har egen mappe med kjørbar kode og requirements.txt.

---

## Oppsett

```bash
# Klone / last ned repo, naviger til ønsket oppgave
cd grocery_bot && pip install -r requirements.txt
```

---

## 1. Grocery Bot (`grocery_bot/`)

WebSocket-bot som spiller automatisk supermarked.

**Kobling:** `wss://game.ainm.no/ws?token=<jwt_token>`

**Kjør:**
```bash
export JWT_TOKEN="ditt_token_her"
python bot.py
```

**Filer:**
- `bot.py` — WebSocket-tilkobling og hovedløkke
- `pathfinding.py` — A*-algoritme
- `strategy.py` — Ordre-scoring og planlegging

**TODO:** Sett `JWT_TOKEN` som miljøvariabel.

---

## 2. Race Car (`race_car/`)

Regelbasert kjøreagent med 16 sensorer.

**Kjør (test):**
```bash
python agent.py
```

**TODO:** Koble `kjør_episode()` til faktisk spillserver-API. Bytt til RL når regelbasert er optimalisert.

---

## 3. Emergency Healthcare RAG (`healthcare_rag/`)

Klassifiserer medisinske påstander som sant/usant + ett av 115 temaer.

**Bygg indeks og test:**
```bash
python rag_pipeline.py   # Bygger FAISS-indeks
python classifier.py     # Kjører testklassifisering
```

**TODO:**
1. Fyll `kunnskapsbase.json` med faktisk medisinsk innhold (115 temaer)
2. Sett `OPENAI_API_KEY` for LLM-fallback (valgfritt)

---

## 4. Tumor Segmentation (`tumor_segmentation/`)

U-Net segmentering av svulster i MIP-PET-bilder. Mål: < 10 sek/bilde.

**Test modell:**
```bash
python model.py           # Verifiser arkitektur og forward pass
python inference.py       # Ytelsestest med syntetisk data
python inference.py bilde.dcm   # Segmenter faktisk bilde
```

**TODO:**
1. Tren modell på PET-treningsdata (`model.py:tren_modell()`)
2. Lagre vekter til `modell_vekter.pth`
3. Tilpass `BILDE_STØRRELSE` til faktisk inputformat

---

## Avhengigheter per oppgave

| Oppgave | Nøkkelavhengigheter |
|---------|---------------------|
| Grocery Bot | websockets |
| Race Car | numpy (+ stable-baselines3 for RL) |
| Healthcare RAG | sentence-transformers, faiss-cpu |
| Tumor Segmentation | torch, monai, nibabel, pydicom |
