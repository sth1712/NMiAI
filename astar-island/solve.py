"""
Astar Island — Norse World Prediction Solver v7
NM i AI 2026

v7: TWO-PARAMETER model (expansion + survival)

Key insight from R2-R7 analysis:
- Settlement survival is NOT fully determined by expansion (R²=0.84)
- Detect BOTH expansion AND survival from 10 queries
- Use expansion for forest/plains distance profiles
- Use survival DIRECTLY for settlement/port predictions
- Invariant: when settlements die, 45% → forest, 55% → empty

Scores: R2=49.3, R3=71.2(#10), R4=81.1(#25), R5=68.9, R6=68.8, R7=63.0
"""

import os
import time
import numpy as np
import requests

BASE = "https://api.ainm.no/astar-island"
JWT_TOKEN = os.environ.get("JWT_TOKEN", "")
MIN_FLOOR = 0.01  # REGELKRAV: minimum 0.01, ALDRI lavere
DEAD_SETT_FOREST_RATIO = 0.316  # sf/(sf+se) share, NOT sf/se ratio. Avg R2-R8.


def get_session():
    s = requests.Session()
    if JWT_TOKEN:
        s.cookies.set("access_token", JWT_TOKEN)
    return s


def cell_to_class(cell):
    if cell in (10, 11, 0):
        return 0
    if 1 <= cell <= 5:
        return cell
    return 0


# ============================================================
# Distance profiles indexed by EXPANSION only
# (forest/plains behavior is expansion-driven)
# ============================================================

FOREST_PROFILES = {
    0.003: {1:[.026,.012,0,.001,.960,0],2:[.026,.005,0,.001,.968,0],3:[.026,0,0,.001,.973,0],4:[.026,0,0,.001,.973,0],5:[.026,0,0,.001,.973,0],6:[.026,0,0,.001,.973,0],7:[.026,0,0,.001,.973,0]},
    0.10:  {1:[.074,.131,.007,.009,.780,0],2:[.074,.122,.006,.009,.789,0],3:[.074,.111,.007,.009,.799,0],4:[.066,.092,.007,.010,.825,0],5:[.040,.055,.009,.007,.889,0],6:[.021,.040,.006,.005,.928,0],7:[.018,.020,.006,.005,.951,0]},
    0.136: {1:[.154,.249,.008,.025,.564,0],2:[.143,.215,.015,.022,.605,0],3:[.047,.112,.009,.013,.819,0],4:[.034,.079,.009,.009,.869,0],5:[.005,.019,.002,.003,.971,0],6:[.002,.008,.001,.001,.988,0],7:[0,0,0,0,1,0]},
    0.21:  {1:[.123,.237,.009,.019,.612,0],2:[.125,.230,.010,.021,.614,0],3:[.123,.214,.012,.020,.630,0],4:[.119,.206,.013,.020,.642,0],5:[.091,.204,.023,.015,.667,0],6:[.045,.163,.014,.012,.766,0],7:[.045,.114,.014,.012,.815,0]},
    0.265: {1:[.167,.315,.011,.035,.472,0],2:[.170,.287,.016,.036,.491,0],3:[.178,.259,.012,.037,.514,0],4:[.163,.234,.016,.036,.550,0],5:[.099,.208,.017,.029,.647,0],6:[.077,.178,.025,.027,.693,0],7:[.037,.114,.020,.012,.816,0]},
}

PLAINS_PROFILES = {
    0.003: {1:[.987,.011,0,0,.002,0],2:[.987,.004,0,0,.009,0],3:[.987,0,0,0,.013,0],4:[.987,0,0,0,.013,0],5:[.987,0,0,0,.013,0],6:[.987,0,0,0,.013,0],7:[.987,0,0,0,.013,0]},
    0.10:  {1:[.808,.126,.004,.011,.051,0],2:[.837,.122,.007,.011,.023,0],3:[.846,.113,.008,.011,.022,0],4:[.859,.093,.009,.009,.030,0],5:[.877,.054,.010,.008,.051,0],6:[.917,.043,.009,.006,.025,0],7:[.949,.021,.007,.004,.019,0]},
    0.136: {1:[.657,.236,.012,.024,.071,0],2:[.693,.203,.015,.021,.069,0],3:[.850,.106,.010,.012,.022,0],4:[.896,.072,.008,.008,.016,0],5:[.978,.017,.001,.002,.002,0],6:[.988,.009,.001,.001,.001,0],7:[1,0,0,0,0,0]},
    0.21:  {1:[.701,.226,.008,.020,.045,0],2:[.700,.225,.013,.021,.041,0],3:[.710,.214,.015,.021,.040,0],4:[.719,.209,.018,.019,.035,0],5:[.735,.203,.021,.018,.023,0],6:[.809,.153,.021,.013,.004,0],7:[.874,.105,.018,.009,0,0]},
    0.265: {1:[.587,.300,.010,.034,.069,0],2:[.597,.281,.013,.035,.074,0],3:[.614,.261,.016,.035,.074,0],4:[.632,.240,.025,.034,.069,0],5:[.707,.197,.025,.029,.042,0],6:[.738,.175,.029,.024,.034,0],7:[.843,.108,.022,.012,.015,0]},
}

# Port survival relative to settlement survival (from R2-R7 data)
# ps/ss ratio: 0.50, 0.56, 0.60, 0.70, 0.76 → avg ~0.62
PORT_SURVIVAL_RATIO = 0.62


def interpolate_dist_profile(profiles, rate):
    """Interpolate distance profiles with extrapolation support."""
    ks = sorted(profiles.keys())
    rate = max(ks[0] * 0.5, rate)  # Allow slight below-range

    if rate <= ks[0]:
        return {d: list(v) for d, v in profiles[ks[0]].items()}
    if rate >= ks[-1]:
        lo, hi = ks[-2], ks[-1]
        t = min((rate - lo) / (hi - lo), 2.0)
        result = {}
        for d in range(1, 8):
            lv, hv = np.array(profiles[lo][d]), np.array(profiles[hi][d])
            v = np.maximum((1 - t) * lv + t * hv, 0)
            s = v.sum()
            result[d] = (v / s).tolist() if s > 0 else list(profiles[hi][d])
        return result

    for i in range(len(ks) - 1):
        if ks[i] <= rate <= ks[i + 1]:
            lo, hi = ks[i], ks[i + 1]
            t = (rate - lo) / (hi - lo)
            result = {}
            for d in range(1, 8):
                lv, hv = np.array(profiles[lo][d]), np.array(profiles[hi][d])
                result[d] = ((1 - t) * lv + t * hv).tolist()
            return result

    return {d: list(v) for d, v in profiles[ks[len(ks) // 2]].items()}


# ============================================================
# Settlement prediction from survival rate (DIRECT, not interpolated)
# ============================================================

def settlement_probs(surv_rate, exp_rate, is_port=False, adj_ocean=False, adj_forest=0):
    """
    Build probability distribution for a settlement/port cell.
    Uses detected survival DIRECTLY + invariant death distribution.
    """
    ruin_rate = max(0.01, min(0.15 * exp_rate + 0.008, 0.05))  # Fix 2: calibrated ruin formula
    surv = surv_rate

    if is_port:
        surv = max(surv_rate * PORT_SURVIVAL_RATIO, 0.02)  # Missing Fix B: port floor

    dead_fraction = max(1.0 - surv - ruin_rate, 0.05)
    to_empty = dead_fraction * (1 - DEAD_SETT_FOREST_RATIO)
    to_forest = dead_fraction * DEAD_SETT_FOREST_RATIO

    if is_port:
        # Port: split survival between port-stays and port-to-sett
        port_stays = surv * 0.7
        port_to_sett = surv * 0.3
        base = np.array([to_empty, port_to_sett, port_stays, ruin_rate, to_forest, 0])
    else:
        port_prob = surv * 0.1 if adj_ocean else surv * 0.01
        sett_stays = surv - port_prob
        base = np.array([to_empty, max(sett_stays, 0.01), port_prob, ruin_rate, to_forest, 0])

    # Weak forest adjacency effect (~6%)
    if adj_forest >= 3:
        base[1] *= 1.06
        base[0] *= 0.96
    elif adj_forest == 0:
        base[1] *= 0.94
        base[0] *= 1.04

    return base


# ============================================================
# Adjacency features
# ============================================================

def compute_features(grid):
    H, W = len(grid), len(grid[0])
    sett_pos = {(r, c) for r in range(H) for c in range(W) if grid[r][c] in (1, 2)}
    features = {}
    for r in range(H):
        for c in range(W):
            af, ao = 0, 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if grid[nr][nc] == 4: af += 1
                        if grid[nr][nc] == 10: ao += 1
            ds = 7
            for d in range(1, 7):
                found = any((r + dr, c + dc) in sett_pos
                            for dr in range(-d, d + 1) for dc in range(-d, d + 1)
                            if abs(dr) + abs(dc) == d)
                if found:
                    ds = d
                    break
            features[(r, c)] = {"adj_forest": af, "adj_ocean": ao, "dist_sett": ds}
    return features


# ============================================================
# Build prior with two parameters
# ============================================================

def build_prior(grid, settlements, features, exp_rate, surv_rate):
    """Build predictions using BOTH expansion and survival."""
    H, W = len(grid), len(grid[0])
    alpha = np.full((H, W, 6), 0.1)
    port_pos = {(s["y"], s["x"]) for s in settlements if s.get("has_port")}

    forest_prof = interpolate_dist_profile(FOREST_PROFILES, exp_rate)
    plains_prof = interpolate_dist_profile(PLAINS_PROFILES, exp_rate)

    # Survival correction for dist=1-2 settlement probability
    expected_surv = 1.588 * exp_rate + 0.074
    surv_correction = (surv_rate / max(expected_surv, 0.01)) ** 0.5 if expected_surv > 0.01 else 1.0
    surv_correction = max(0.5, min(2.0, surv_correction))

    for r in range(H):
        for c in range(W):
            cell = grid[r][c]
            f = features[(r, c)]

            if cell == 10:
                alpha[r][c] = np.array([100, .01, .01, .01, .01, .01])
                continue
            if cell == 5:
                alpha[r][c] = np.array([.01, .01, .01, .01, .01, 100])
                continue

            dist = min(f["dist_sett"], 7)
            adj_o = f["adj_ocean"]
            adj_f = min(f["adj_forest"], 5)

            if cell == 4:  # Forest
                base = np.array(forest_prof[dist])
                if adj_o > 0 and dist <= 3:
                    base[2] *= 2.0
                # Apply survival correction at close distances
                if dist <= 2:
                    base[1] *= surv_correction
                strength = 7.0  # Deep Research V2: økt fra 5.0 for sterkere prior (demper stokastisk støy)

            elif cell == 11:  # Plains
                base = np.array(plains_prof[dist])
                if adj_o > 0 and dist <= 4:
                    base[2] *= 2.0
                if dist <= 2:
                    base[1] *= surv_correction
                strength = 7.0  # Deep Research V2: økt fra 5.0

            elif cell in (1, 2):  # Settlement or Port
                is_port = (r, c) in port_pos or cell == 2
                base = settlement_probs(surv_rate, exp_rate, is_port, adj_o > 0, adj_f)
                strength = 5.0  # Deep Research V2: økt fra 3.5 (sterkere prior demper enkeltobservasjons-støy)

            elif cell == 3:  # Ruin
                base = np.array([0.35, 0.10, 0.03, 0.12, 0.40, 0])
                if dist <= 2:
                    base[1] *= 1.5
                    if adj_o > 0:
                        base[2] *= 2.0
                if adj_f >= 2:
                    base[4] *= 1.3
                strength = 4.5  # Deep Research V2: økt fra 3.0

            else:  # Empty
                base = np.array(plains_prof.get(dist, plains_prof[7]))
                strength = 7.0  # Deep Research V2: økt fra 5.0

            base = np.maximum(base, 0.001)
            base = base / base.sum()
            alpha[r][c] = base * strength

    return alpha


def alpha_to_prediction(alpha, grid=None):
    """Convert Dirichlet alpha to predictions. Floor 0.01 per REGLER."""
    pred = alpha / alpha.sum(axis=-1, keepdims=True)
    pred = np.maximum(pred, MIN_FLOOR)  # 0.01 for ALL cells per rules
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


# ============================================================
# Two-parameter detection
# ============================================================

def detect_parameters(observations, grids):
    """Detect BOTH expansion and survival from observations."""
    f_total, f_sett = 0, 0
    p_total, p_sett = 0, 0
    s_total, s_surv = 0, 0

    for seed_idx, obs_list in observations.items():
        grid = grids[seed_idx]
        H, W = len(grid), len(grid[0])
        for obs in obs_list:
            vp = obs["viewport"]
            og = obs["grid"]
            for dr in range(len(og)):
                for dc in range(len(og[0])):
                    r, c = vp["y"] + dr, vp["x"] + dc
                    if r >= H or c >= W:
                        continue
                    init = grid[r][c]
                    obs_cls = cell_to_class(og[dr][dc])

                    if init == 4:
                        f_total += 1
                        if obs_cls in (1, 2): f_sett += 1
                    elif init in (1, 2):
                        s_total += 1
                        if obs_cls in (1, 2): s_surv += 1
                    elif init == 11:
                        p_total += 1
                        if obs_cls in (1, 2): p_sett += 1

    # Expansion from forest + plains
    exp_rates = []
    if f_total > 30: exp_rates.append(f_sett / f_total)
    if p_total > 30: exp_rates.append(p_sett / p_total)
    exp_rate = sum(exp_rates) / len(exp_rates) if exp_rates else 0.136

    # Survival DIRECTLY from settlement observations
    surv_rate = s_surv / s_total if s_total > 10 else 1.588 * exp_rate + 0.074

    print(f"  Expansion: f={f_sett}/{f_total} p={p_sett}/{p_total} → exp={exp_rate:.4f}")
    print(f"  Survival: {s_surv}/{s_total} → surv={surv_rate:.4f}")
    print(f"  Expected surv from exp: {1.588*exp_rate+0.074:.4f}, actual: {surv_rate:.4f}")

    return exp_rate, surv_rate


# ============================================================
# Bayesian update
# ============================================================

def bayesian_update(alpha, observations, grid, H=40, W=40):
    for obs in observations:
        vp = obs["viewport"]
        og = obs["grid"]
        for dr in range(len(og)):
            for dc in range(len(og[0])):
                r, c = vp["y"] + dr, vp["x"] + dc
                if r >= H or c >= W or grid[r][c] in (10, 5):
                    continue
                alpha[r][c][cell_to_class(og[dr][dc])] += 1.0

        for s in obs.get("settlements", []):
            if s["y"] >= H or s["x"] >= W:
                continue
            if grid[s["y"]][s["x"]] not in (1, 2):
                continue
            if not s.get("alive", True):
                # Fix 1: match 55:45 invariant (was 67:33)
                alpha[s["y"]][s["x"]][0] += 2.75
                alpha[s["y"]][s["x"]][4] += 2.25
            elif s.get("population", 1) > 3 and s.get("food", 0.5) > 0.7:
                # Fix 7A: Very strong → +4.0 (was +3.0)
                idx = 2 if s.get("has_port") else 1
                alpha[s["y"]][s["x"]][idx] += 4.0
            elif s.get("population", 1) > 2 and s.get("food", 0.5) > 0.5:
                idx = 2 if s.get("has_port") else 1
                alpha[s["y"]][s["x"]][idx] += 2.0
            elif s.get("food", 0.5) < 0.3 or s.get("population", 1) < 0.8:
                alpha[s["y"]][s["x"]][0] += 2.0
                alpha[s["y"]][s["x"]][4] += 1.0

    return np.maximum(alpha, 0.05)


# ============================================================
# Viewport planning
# ============================================================

def plan_viewports(grid, settlements, num_vp, map_w=40, map_h=40):
    H, W = map_h, map_w
    dynamic = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if grid[r][c] in (1, 2, 3, 4):
                dynamic[r][c] = True
    for s in settlements:
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                r, c = s["y"] + dr, s["x"] + dc
                if 0 <= r < H and 0 <= c < W and grid[r][c] == 11:
                    dynamic[r][c] = True

    covered = np.zeros((H, W), dtype=bool)
    viewports = []
    for _ in range(num_vp):
        bx, by, bs = 0, 0, 0
        for y in range(0, H - 14, 2):
            for x in range(0, W - 14, 2):
                sc = sum(1 for dr in range(15) for dc in range(15)
                         if y+dr < H and x+dc < W and dynamic[y+dr][x+dc] and not covered[y+dr][x+dc])
                if sc > bs:
                    bs, bx, by = sc, x, y
        if bs == 0:
            break
        viewports.append({"x": bx, "y": by, "w": 15, "h": 15, "score": bs})
        for dr in range(15):
            for dc in range(15):
                if by + dr < H and bx + dc < W:
                    covered[by + dr][bx + dc] = True
    return viewports


# ============================================================
# API helpers
# ============================================================

def simulate(session, round_id, seed_index, vp):
    resp = session.post(f"{BASE}/simulate", json={
        "round_id": round_id, "seed_index": seed_index,
        "viewport_x": vp["x"], "viewport_y": vp["y"],
        "viewport_w": vp["w"], "viewport_h": vp["h"],
    })
    resp.raise_for_status()
    return resp.json()


def submit(session, round_id, seed_index, prediction):
    resp = session.post(f"{BASE}/submit", json={
        "round_id": round_id, "seed_index": seed_index,
        "prediction": prediction.tolist(),
    })
    resp.raise_for_status()
    return resp.json()


# ============================================================
# Main pipeline
# ============================================================

def main():
    session = get_session()
    if not JWT_TOKEN:
        print("FEIL: export JWT_TOKEN='...'")
        return

    rounds = session.get(f"{BASE}/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("Ingen aktiv runde.")
        return

    round_id = active["id"]
    detail = session.get(f"{BASE}/rounds/{round_id}").json()
    H, W = detail["map_height"], detail["map_width"]
    seeds = detail["seeds_count"]

    budget = session.get(f"{BASE}/budget").json()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"Runde #{active['round_number']}, {W}x{H}, {seeds} seeds, {remaining}q")

    if remaining == 0:
        print("Ingen queries igjen.")
        return

    # Features
    all_features = {}
    for si in range(seeds):
        all_features[si] = compute_features(detail["initial_states"][si]["grid"])

    # 1. Baseline with medium estimates
    print("\n--- 1/4: Baseline ---")
    for si in range(seeds):
        state = detail["initial_states"][si]
        alpha = build_prior(state["grid"], state["settlements"], all_features[si], 0.136, 0.30)
        r = submit(session, round_id, si, alpha_to_prediction(alpha, state["grid"]))
        print(f"  S{si}: {r['status']}")
        time.sleep(0.6)

    # 2. Detection queries (Fix 4: reduced from 10 to 6)
    print("\n--- 2/4: Detection ---")
    plans = {si: plan_viewports(detail["initial_states"][si]["grid"],
             detail["initial_states"][si]["settlements"], 10, W, H) for si in range(seeds)}
    all_obs = {si: [] for si in range(seeds)}
    detect_n = min(6, remaining)  # Fix 4: 6 instead of 10
    q_used = 0
    for si in range(seeds):
        for vp in plans[si][:2]:
            if q_used >= detect_n:
                break
            obs = simulate(session, round_id, si, vp)
            all_obs[si].append(obs)
            q_used += 1
            time.sleep(0.23)

    # 3. Detect both parameters
    print("\n--- 3/4: Detect & resubmit ---")
    grids = {si: detail["initial_states"][si]["grid"] for si in range(seeds)}
    exp_rate, surv_rate = detect_parameters(all_obs, grids)

    for si in range(seeds):
        state = detail["initial_states"][si]
        alpha = build_prior(state["grid"], state["settlements"], all_features[si], exp_rate, surv_rate)
        alpha = bayesian_update(alpha, all_obs[si], state["grid"], H, W)
        r = submit(session, round_id, si, alpha_to_prediction(alpha, state["grid"]))
        print(f"  S{si}: {r['status']}")
        time.sleep(0.6)

    # 4. Refinement (Fix 4: 44 queries instead of 40)
    rem_q = remaining - q_used
    print(f"\n--- 4/4: Refine ({rem_q}q) ---")
    allocs = []
    left = rem_q
    seed_order = sorted(range(seeds), key=lambda s: sum(v["score"] for v in plans[s]), reverse=True)
    for si in seed_order:
        for vp in plans[si][2:6]:
            if left <= 0: break
            allocs.append({"seed": si, "vp": vp})
            left -= 1
    # Repeats
    all_vps = [(si, vp) for si in seed_order for vp in plans[si][:3]]
    all_vps.sort(key=lambda x: x[1]["score"], reverse=True)
    idx = 0
    while left > 0 and all_vps:
        si, vp = all_vps[idx % len(all_vps)]
        allocs.append({"seed": si, "vp": vp})
        left -= 1
        idx += 1

    for i, a in enumerate(allocs):
        obs = simulate(session, round_id, a["seed"], a["vp"])
        all_obs[a["seed"]].append(obs)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(allocs)}")
        time.sleep(0.23)

    # Re-detect with all data
    exp2, surv2 = detect_parameters(all_obs, grids)

    # Fix 5: Recompute dist_sett from observed NEW settlements
    print("\n--- Fix 5: Update features from observations ---")
    for si in range(seeds):
        state = detail["initial_states"][si]
        grid = state["grid"]
        # Find new settlement positions from observations
        new_setts = set()
        for obs in all_obs[si]:
            vp = obs["viewport"]
            og = obs["grid"]
            for dr in range(len(og)):
                for dc in range(len(og[0])):
                    r, c = vp["y"] + dr, vp["x"] + dc
                    if r >= H or c >= W:
                        continue
                    if grid[r][c] in (4, 11, 0) and cell_to_class(og[dr][dc]) in (1, 2):
                        new_setts.add((r, c))
        if new_setts:
            # Update dist_sett for cells near new settlements
            for (nr, nc) in new_setts:
                for d in range(1, 7):
                    for dr2 in range(-d, d + 1):
                        for dc2 in range(-d, d + 1):
                            if abs(dr2) + abs(dc2) != d:
                                continue
                            r2, c2 = nr + dr2, nc + dc2
                            if 0 <= r2 < H and 0 <= c2 < W and (r2, c2) in all_features[si]:
                                if d < all_features[si][(r2, c2)]["dist_sett"]:
                                    all_features[si][(r2, c2)]["dist_sett"] = d
            print(f"  S{si}: {len(new_setts)} new settlements detected, features updated")

    # Final resubmit with updated features
    print("\n--- Final resubmit ---")
    for si in range(seeds):
        state = detail["initial_states"][si]
        alpha = build_prior(state["grid"], state["settlements"], all_features[si], exp2, surv2)
        alpha = bayesian_update(alpha, all_obs[si], state["grid"], H, W)
        r = submit(session, round_id, si, alpha_to_prediction(alpha, state["grid"]))
        print(f"  S{si}: {r['status']} ({len(all_obs[si])} obs)")
        time.sleep(0.6)

    b = session.get(f"{BASE}/budget").json()
    print(f"\nDone! {b['queries_used']}/{b['queries_max']}q, exp={exp2:.4f}, surv={surv2:.4f}")


if __name__ == "__main__":
    main()
