"""
strategy.py — Høynivå-strategi og ordre-planlegging for Grocery Bot

Strategiprioriteringer:
1. Fullfør pågående ordre hvis inventaret er fullt
2. Plukk varer for den mest lønnsomme ordren tilgjengelig
3. Lever varer hvis alle er plukket
4. Unngå å gå tomhendende til levering
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

MAKS_INVENTAR = 3


@dataclass
class Vare:
    item_id: str
    navn: str
    posisjon: tuple[int, int]


@dataclass
class Ordre:
    ordre_id: str
    varer: list[str]          # liste av item_id
    leveringssted: tuple[int, int]
    poengverdi: int = 0       # beregnes ved parsing

    def __post_init__(self):
        # +1 per vare levert + +5 for fullført ordre
        self.poengverdi = len(self.varer) + 5


@dataclass
class BotTilstand:
    posisjon: tuple[int, int] = (0, 0)
    inventar: list[str] = field(default_factory=list)   # item_id-er
    aktiv_ordre: Optional[str] = None
    mål_sekvens: list[str] = field(default_factory=list)  # planlagte aksjoner


def parse_spilltilstand(game_state: dict) -> tuple[BotTilstand, list[Ordre], list[Vare], set]:
    """
    Tolk JSON game_state fra serveren.
    Returnerer (bot_tilstand, ordrer, varer, hindringer).

    Tilpass felt-navnene til faktisk API-respons fra serveren.
    """
    bot_raw = game_state.get("bot", {})
    bot = BotTilstand(
        posisjon=tuple(bot_raw.get("position", [0, 0])),
        inventar=bot_raw.get("inventory", []),
        aktiv_ordre=bot_raw.get("active_order"),
    )

    ordrer = []
    for o in game_state.get("orders", []):
        ordrer.append(Ordre(
            ordre_id=o["id"],
            varer=o.get("items", []),
            leveringssted=tuple(o.get("delivery_position", [0, 0])),
        ))

    varer = []
    for v in game_state.get("items", []):
        varer.append(Vare(
            item_id=v["id"],
            navn=v.get("name", "ukjent"),
            posisjon=tuple(v.get("position", [0, 0])),
        ))

    hindringer: set[tuple[int, int]] = set()
    for h in game_state.get("obstacles", []):
        hindringer.add(tuple(h))

    return bot, ordrer, varer, hindringer


def score_ordre(ordre: Ordre, bot_pos: tuple[int, int], tilgjengelige_varer: list[Vare]) -> float:
    """
    Scorer en ordre basert på poengverdi vs. estimert reisekostnad.
    Høyere score = mer attraktiv.
    """
    vare_map = {v.item_id: v for v in tilgjengelige_varer}
    manglende = [vid for vid in ordre.varer if vid in vare_map]

    if not manglende:
        return -999  # Ingen tilgjengelige varer — skip

    # Grovt estimat: avstand til første vare + avstand fra vare til levering
    første_vare_pos = vare_map[manglende[0]].posisjon
    avstand = (
        abs(bot_pos[0] - første_vare_pos[0]) + abs(bot_pos[1] - første_vare_pos[1])
        + abs(første_vare_pos[0] - ordre.leveringssted[0])
        + abs(første_vare_pos[1] - ordre.leveringssted[1])
    )
    if avstand == 0:
        avstand = 1
    return ordre.poengverdi / avstand


def velg_beste_ordre(
    ordrer: list[Ordre],
    bot: BotTilstand,
    tilgjengelige_varer: list[Vare],
) -> Optional[Ordre]:
    """Velg ordren med best score/avstand-ratio."""
    if not ordrer:
        return None

    rangert = sorted(
        ordrer,
        key=lambda o: score_ordre(o, bot.posisjon, tilgjengelige_varer),
        reverse=True,
    )
    beste = rangert[0]
    if score_ordre(beste, bot.posisjon, tilgjengelige_varer) < 0:
        return None
    return beste


def planlegg_aksjoner(
    game_state: dict,
    grid_bredde: int,
    grid_høyde: int,
) -> list[dict]:
    """
    Hovedstrategi. Returnerer liste med aksjons-dicts klar for sending til server.
    Kaller pathfinding internt.

    Aksjonformat: {"action": "move_right"} eller {"action": "pick_up", "item_id": "..."}
    """
    from pathfinding import finn_sti, sti_til_aksjoner, nærmeste_mål

    bot, ordrer, varer, hindringer = parse_spilltilstand(game_state)
    vare_map = {v.item_id: v for v in varer}

    aksjoner: list[dict] = []

    # --- Prioritet 1: Lever hvis inventaret matcher en ordre ---
    for ordre in ordrer:
        if alle_varer_i_inventar(ordre, bot.inventar):
            sti = finn_sti(bot.posisjon, ordre.leveringssted, hindringer, grid_bredde, grid_høyde)
            bevegelser = sti_til_aksjoner(sti, bot.posisjon)
            aksjoner.extend({"action": a} for a in bevegelser)
            aksjoner.append({"action": "drop_off"})
            logger.info(f"Leverer ordre {ordre.ordre_id} — {len(bevegelser)} steg")
            return aksjoner[:20]  # Sikkerhetsgrense

    # --- Prioritet 2: Plukk varer hvis plass ---
    if len(bot.inventar) < MAKS_INVENTAR:
        beste_ordre = velg_beste_ordre(ordrer, bot, varer)
        if beste_ordre:
            manglende_vare_ids = [
                vid for vid in beste_ordre.varer
                if vid not in bot.inventar and vid in vare_map
            ]
            # Begrens til maks inventar-kapasitet
            plukk_ids = manglende_vare_ids[: MAKS_INVENTAR - len(bot.inventar)]
            bot_pos = bot.posisjon

            for item_id in plukk_ids:
                vare = vare_map[item_id]
                sti = finn_sti(bot_pos, vare.posisjon, hindringer, grid_bredde, grid_høyde)
                bevegelser = sti_til_aksjoner(sti, bot_pos)
                aksjoner.extend({"action": a} for a in bevegelser)
                aksjoner.append({"action": "pick_up", "item_id": item_id})
                bot_pos = vare.posisjon
                logger.info(f"Plukker {item_id} for ordre {beste_ordre.ordre_id}")

            return aksjoner[:20]

    # --- Prioritet 3: Lever det vi har til nærmeste ordre ---
    if bot.inventar:
        for ordre in ordrer:
            sti = finn_sti(bot.posisjon, ordre.leveringssted, hindringer, grid_bredde, grid_høyde)
            if sti is not None:
                bevegelser = sti_til_aksjoner(sti, bot.posisjon)
                aksjoner.extend({"action": a} for a in bevegelser)
                aksjoner.append({"action": "drop_off"})
                return aksjoner[:20]

    # --- Fallback ---
    aksjoner.append({"action": "wait"})
    return aksjoner


def alle_varer_i_inventar(ordre: Ordre, inventar: list[str]) -> bool:
    """Sjekk om alle varer i ordren er i inventaret."""
    return all(vid in inventar for vid in ordre.varer)
