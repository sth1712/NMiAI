# Agent 03: Astar Island — Norse World Prediction

## Rolle
Observer en norrøn sivilisasjonssimulator, prediker sluttilstand for alle celler i 40x40 kart.

## Filer
- `astar-island/` — arbeidsmappe for klient og prediksjonslogikk

## API
- **Base:** https://api.ainm.no/astar-island/
- **Auth:** Bearer JWT (fra app.ainm.no)
- **Rate limit:** 5 req/sek (simulate), 2 req/sek (submit)

### Endepunkter
| Metode | Sti | Beskrivelse |
|--------|-----|-------------|
| GET | /rounds | List aktive runder |
| GET | /rounds/{id} | Detaljer om runde (initial_states, status) |
| POST | /simulate | Observer viewport: `{round_id, seed, year, x, y, width, height}` |
| POST | /submit | Send prediksjon: `{round_id, seed, predictions}` |

## Scoring
- **Entropivektet KL-divergens**, 0-100 skala
- Statiske celler (hav, fjell, skog) har lav entropi → teller nesten ikke
- **Dynamiske celler (settlements, ports, ruins) er der poengene er**
- ALDRI sett sannsynlighet til 0.0 — bruk minimum 0.01 og renormaliser

## Kartstruktur
- 40x40 rutenett, 5 seeds per runde, 50 queries per runde (delt på alle seeds)
- Viewport: maks 15x15 per query
- Simulering: 50 år (Growth → Conflict → Trade → Winter → Environment)

## Terrengkoder → Klasser
| Terrengkode | Verdi | Klasse-indeks | Beskrivelse |
|-------------|-------|---------------|-------------|
| Ocean | 10 | 0 (Empty) | Statisk — endres aldri |
| Plains | 11 | 0 (Empty) | Kan bli settlement/port/ruin |
| Empty | 0 | 0 | Tom celle |
| Settlement | 1 | 1 | Dynamisk — kan vokse/forsvinne |
| Port | 2 | 2 | Dynamisk — ved kyst |
| Ruin | 3 | 3 | Dynamisk — ødelagt settlement |
| Forest | 4 | 4 | Semi-statisk |
| Mountain | 5 | 5 | Statisk — endres aldri |

## Arbeidsflyt per runde

### 1. Hent rundeinfo og initial_states
```python
headers = {"Authorization": f"Bearer {JWT_TOKEN}"}
round_id = requests.get(f"{BASE}/rounds", headers=headers).json()[0]["id"]
initial_states = requests.get(f"{BASE}/rounds/{round_id}", headers=headers).json()["initial_states"]
```

### 2. Bygg baseline fra initial_states
```python
import numpy as np

def build_baseline(initial_grid):
    """Konverter initial_state til 40x40x6 sannsynlighetsfordeling."""
    pred = np.full((40, 40, 6), 0.01)  # Gulv: aldri 0.0
    for r in range(40):
        for c in range(40):
            cell = initial_grid[r][c]
            if cell == 10:       # Ocean → Empty (0), 100% statisk
                pred[r][c] = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]
            elif cell == 5:      # Mountain → statisk
                pred[r][c] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.95]
            elif cell == 4:      # Forest → semi-statisk
                pred[r][c] = [0.05, 0.01, 0.01, 0.01, 0.91, 0.01]
            elif cell == 11:     # Plains → sannsynligvis empty, men kan utvikle seg
                pred[r][c] = [0.70, 0.10, 0.05, 0.05, 0.05, 0.05]
            elif cell == 1:      # Settlement → dynamisk
                pred[r][c] = [0.10, 0.40, 0.10, 0.20, 0.10, 0.10]
            elif cell == 2:      # Port → dynamisk
                pred[r][c] = [0.10, 0.10, 0.40, 0.20, 0.10, 0.10]
            elif cell == 3:      # Ruin → var ødelagt
                pred[r][c] = [0.30, 0.10, 0.05, 0.40, 0.10, 0.05]
    # Renormaliser
    for r in range(40):
        for c in range(40):
            pred[r][c] = np.maximum(pred[r][c], 0.01)
            pred[r][c] /= pred[r][c].sum()
    return pred
```

### 3. Observer med simulate (50 queries totalt!)
```python
def observe(round_id, seed, year, x, y, w=15, h=15):
    return requests.post(f"{BASE}/simulate", headers=headers, json={
        "round_id": round_id, "seed": seed, "year": year,
        "x": x, "y": y, "width": w, "height": h
    }).json()
```

### 4. Submit prediksjon
```python
def submit(round_id, seed, pred_array):
    return requests.post(f"{BASE}/submit", headers=headers, json={
        "round_id": round_id, "seed": seed, "predictions": pred_array.tolist()
    }).json()
```

## Query-strategi (50 queries totalt for 5 seeds)
- **10 queries per seed** (eller fokuser på 2-3 seeds med flest dynamiske celler)
- Observer **rundt settlements/ports** — de endrer seg mest
- **År 25 og 50** — midtveis og slutt gir mest info
- Bruk alltid **15x15 viewports** for maks dekning

### Oppdater prediksjoner fra observasjoner
```python
def update_from_observation(pred, obs_grid, x, y):
    """Oppdater prediksjon basert på observert tilstand."""
    for dr in range(len(obs_grid)):
        for dc in range(len(obs_grid[0])):
            cell = obs_grid[dr][dc]
            r, c = y + dr, x + dc
            # Observert celle ved dette årstallet — sterk indikator
            class_idx = cell if cell <= 5 else 0
            pred[r][c] = np.full(6, 0.02)
            pred[r][c][class_idx] = 0.90
            pred[r][c] /= pred[r][c].sum()
    return pred
```

## Renormalisering (KRITISK)
```python
def ensure_valid(pred):
    pred = np.maximum(pred, 0.01)
    for r in range(40):
        for c in range(40):
            pred[r][c] /= pred[r][c].sum()
    return pred
```

## Runde-håndtering
- Runder er tidsbegrensede — sjekk `round_info["ends_at"]`
- Submit baseline TIDLIG, forbedre etterpå (best score beholdes)
- Nye runder kan åpne — sjekk GET /rounds regelmessig

## Strategi-nivåer
1. **Baseline:** initial_states → bygg baseline → submit alle 5 seeds. Gjør dette FØRST.
2. **Observer:** Bruk 50 queries strategisk → oppdater → resubmit.
3. **Simuleringsmodell:** Modeller Growth/Conflict/Trade/Winter. Prediker uten queries.

## Sjekkliste
- [ ] Ny runde aktiv? (GET /rounds)
- [ ] Baseline submitted for alle seeds?
- [ ] Queries brukt på dynamiske områder?
- [ ] Sjekk score på app.ainm.no
