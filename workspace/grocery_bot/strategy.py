"""
strategy.py — Høynivå-strategi og ordre-planlegging for Grocery Bot

Strategiprioriteringer:
1. Fullfør pågående ordre hvis inventaret er fullt
2. Plukk varer for den mest lønnsomme ordren tilgjengelig
3. Lever varer hvis alle er plukket
4. Unngå å gå tomhendende til levering

Multi-bot-støtte:
- Serveren kan sende flere bots i game_state["bots"]
- Forskjellige bots tildeles forskjellige ordrer
- Kollisjonsunngåelse mellom egne bots
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

MAKS_INVENTAR = 3

# Vektparametere for ordrescoring
VEKT_NESTEN_FERDIG = 3.0      # Bonus for ordrer med bare 1 vare igjen
VEKT_VARER_I_INVENTAR = 2.0   # Bonus per vare vi allerede har


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
    bot_id: str = "default"
    posisjon: tuple[int, int] = (0, 0)
    inventar: list[str] = field(default_factory=list)   # item_id-er
    aktiv_ordre: Optional[str] = None
    mål_sekvens: list[str] = field(default_factory=list)  # planlagte aksjoner


# Holder styr på hvilke ordrer som er tildelt hvilken bot (per tur)
_tildelte_ordrer: dict[str, str] = {}  # ordre_id -> bot_id


def parse_spilltilstand(
    game_state: dict,
) -> tuple[list[BotTilstand], list[Ordre], list[Vare], set[tuple[int, int]]]:
    """
    Tolk JSON game_state fra serveren.
    Returnerer (bots, ordrer, varer, hindringer).

    Støtter to formater:
    - Enkelt-bot: game_state["bot"] (bakoverkompatibelt)
    - Multi-bot:  game_state["bots"] (liste)
    """
    bots: list[BotTilstand] = []

    # Multi-bot-format: game_state["bots"] er en liste
    if "bots" in game_state and isinstance(game_state["bots"], list):
        for bot_raw in game_state["bots"]:
            bots.append(BotTilstand(
                bot_id=bot_raw.get("id", f"bot_{len(bots)}"),
                posisjon=tuple(bot_raw.get("position", [0, 0])),
                inventar=bot_raw.get("inventory", []),
                aktiv_ordre=bot_raw.get("active_order"),
            ))
    # Enkelt-bot-format: game_state["bot"] (bakoverkompatibelt)
    elif "bot" in game_state:
        bot_raw = game_state["bot"]
        bots.append(BotTilstand(
            bot_id=bot_raw.get("id", "default"),
            posisjon=tuple(bot_raw.get("position", [0, 0])),
            inventar=bot_raw.get("inventory", []),
            aktiv_ordre=bot_raw.get("active_order"),
        ))

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

    return bots, ordrer, varer, hindringer


def _beregn_rutelengde(
    bot_pos: tuple[int, int],
    vare_posisjoner: list[tuple[int, int]],
    leveringssted: tuple[int, int],
) -> int:
    """
    Beregn faktisk rutelengde: sum av Manhattan-avstander gjennom alle
    varer i sekvens, pluss avstand fra siste vare til leveringssted.
    """
    if not vare_posisjoner:
        # Bare levering igjen
        return abs(bot_pos[0] - leveringssted[0]) + abs(bot_pos[1] - leveringssted[1])

    total = 0
    nåværende = bot_pos
    for vare_pos in vare_posisjoner:
        total += abs(nåværende[0] - vare_pos[0]) + abs(nåværende[1] - vare_pos[1])
        nåværende = vare_pos

    # Fra siste vare til levering
    total += abs(nåværende[0] - leveringssted[0]) + abs(nåværende[1] - leveringssted[1])
    return total


def score_ordre(
    ordre: Ordre,
    bot_pos: tuple[int, int],
    tilgjengelige_varer: list[Vare],
    inventar: list[str],
) -> float:
    """
    Forbedret ordrescoring:
    - Beregner faktisk rutelengde (alle varer + levering)
    - Vekter med antall varer allerede i inventar for ordren
    - Prioriterer ordrer som nesten er ferdige (1 vare igjen)
    Høyere score = mer attraktiv.
    """
    vare_map = {v.item_id: v for v in tilgjengelige_varer}

    # Finn manglende varer (ikke i inventar og finnes på kartet)
    manglende = [vid for vid in ordre.varer if vid not in inventar and vid in vare_map]
    # Varer vi allerede har i inventar for denne ordren
    har_i_inventar = [vid for vid in ordre.varer if vid in inventar]

    # Hvis alle varer er hentet (enten i inventar eller utilgjengelige)
    if not manglende and not har_i_inventar:
        return -999  # Ingen tilgjengelige varer og ingenting i inventar — skip

    # Beregn faktisk rutelengde gjennom alle manglende varer
    vare_posisjoner = [vare_map[vid].posisjon for vid in manglende]
    rutelengde = _beregn_rutelengde(bot_pos, vare_posisjoner, ordre.leveringssted)
    if rutelengde == 0:
        rutelengde = 1

    base_score = ordre.poengverdi / rutelengde

    # Bonus: varer vi allerede har i inventar for denne ordren
    inventar_bonus = len(har_i_inventar) * VEKT_VARER_I_INVENTAR

    # Bonus: ordrer som nesten er ferdige (bare 1 vare igjen å plukke)
    nesten_ferdig_bonus = 0.0
    if len(manglende) == 1:
        nesten_ferdig_bonus = VEKT_NESTEN_FERDIG
    elif len(manglende) == 0 and har_i_inventar:
        # Alt er i inventar — bare levering gjenstår!
        nesten_ferdig_bonus = VEKT_NESTEN_FERDIG * 2

    return base_score + inventar_bonus + nesten_ferdig_bonus


def velg_beste_ordre(
    ordrer: list[Ordre],
    bot: BotTilstand,
    tilgjengelige_varer: list[Vare],
    reserverte_ordrer: set[str] | None = None,
) -> Optional[Ordre]:
    """
    Velg ordren med best score/avstand-ratio.
    Ekskluderer ordrer som allerede er reservert av andre bots.
    """
    if not ordrer:
        return None

    if reserverte_ordrer is None:
        reserverte_ordrer = set()

    tilgjengelige = [o for o in ordrer if o.ordre_id not in reserverte_ordrer]
    if not tilgjengelige:
        # Fallback: alle ordrer reservert, prøv å ta den beste uansett
        tilgjengelige = ordrer

    rangert = sorted(
        tilgjengelige,
        key=lambda o: score_ordre(o, bot.posisjon, tilgjengelige_varer, bot.inventar),
        reverse=True,
    )
    beste = rangert[0]
    if score_ordre(beste, bot.posisjon, tilgjengelige_varer, bot.inventar) < 0:
        return None
    return beste


def _planlegg_for_enkelt_bot(
    bot: BotTilstand,
    ordrer: list[Ordre],
    varer: list[Vare],
    hindringer: set[tuple[int, int]],
    grid_bredde: int,
    grid_høyde: int,
    reserverte_ordrer: set[str] | None = None,
    andre_bot_posisjoner: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """
    Planlegg aksjoner for én bot. Brukes av multi-bot-koordinator
    og direkte i enkelt-bot-modus.
    """
    from pathfinding import finn_sti, sti_til_aksjoner

    vare_map = {v.item_id: v for v in varer}

    # Bygg dynamiske hindringer: statiske + andre bots sine posisjoner
    dynamiske_hindringer = set(hindringer)
    if andre_bot_posisjoner:
        for pos in andre_bot_posisjoner:
            if pos != bot.posisjon:
                dynamiske_hindringer.add(pos)

    aksjoner: list[dict] = []

    # --- Prioritet 1: Lever hvis inventaret matcher en ordre ---
    for ordre in ordrer:
        if alle_varer_i_inventar(ordre, bot.inventar):
            sti = finn_sti(
                bot.posisjon, ordre.leveringssted,
                dynamiske_hindringer, grid_bredde, grid_høyde,
            )
            if not sti and bot.posisjon != ordre.leveringssted:
                # Sti blokkert av annen bot — prøv uten dynamiske hindringer
                sti = finn_sti(
                    bot.posisjon, ordre.leveringssted,
                    hindringer, grid_bredde, grid_høyde,
                )
                if not sti:
                    continue
                # Kollisjonsrisiko — legg inn venting først
                aksjoner.append({"action": "wait"})
                logger.info(
                    f"Bot {bot.bot_id}: Venter 1 tur pga. kollisjonsrisiko "
                    f"ved levering av {ordre.ordre_id}"
                )
                return aksjoner

            bevegelser = sti_til_aksjoner(sti, bot.posisjon)
            aksjoner.extend({"action": a} for a in bevegelser)
            aksjoner.append({"action": "drop_off"})
            logger.info(
                f"Bot {bot.bot_id}: Leverer ordre {ordre.ordre_id} "
                f"— {len(bevegelser)} steg"
            )
            return aksjoner[:20]

    # --- Prioritet 2: Plukk varer hvis plass ---
    if len(bot.inventar) < MAKS_INVENTAR:
        beste_ordre = velg_beste_ordre(ordrer, bot, varer, reserverte_ordrer)
        if beste_ordre:
            # Registrer denne ordren som reservert for denne boten
            if reserverte_ordrer is not None:
                reserverte_ordrer.add(beste_ordre.ordre_id)
            _tildelte_ordrer[beste_ordre.ordre_id] = bot.bot_id

            manglende_vare_ids = [
                vid for vid in beste_ordre.varer
                if vid not in bot.inventar and vid in vare_map
            ]
            plukk_ids = manglende_vare_ids[: MAKS_INVENTAR - len(bot.inventar)]
            bot_pos = bot.posisjon

            for item_id in plukk_ids:
                vare = vare_map[item_id]
                sti = finn_sti(
                    bot_pos, vare.posisjon,
                    dynamiske_hindringer, grid_bredde, grid_høyde,
                )
                if not sti and bot_pos != vare.posisjon:
                    # Sti blokkert — prøv uten dynamiske hindringer
                    sti = finn_sti(
                        bot_pos, vare.posisjon,
                        hindringer, grid_bredde, grid_høyde,
                    )
                    if not sti:
                        continue
                    # Kollisjonskurs — vent
                    aksjoner.append({"action": "wait"})
                    logger.info(
                        f"Bot {bot.bot_id}: Venter 1 tur — kollisjonskurs "
                        f"mot vare {item_id}"
                    )
                    return aksjoner

                bevegelser = sti_til_aksjoner(sti, bot_pos)
                aksjoner.extend({"action": a} for a in bevegelser)
                aksjoner.append({"action": "pick_up", "item_id": item_id})
                bot_pos = vare.posisjon
                logger.info(
                    f"Bot {bot.bot_id}: Plukker {item_id} "
                    f"for ordre {beste_ordre.ordre_id}"
                )

            if aksjoner:
                return aksjoner[:20]

    # --- Prioritet 3: Lever det vi har til nærmeste ordre ---
    if bot.inventar:
        for ordre in ordrer:
            sti = finn_sti(
                bot.posisjon, ordre.leveringssted,
                dynamiske_hindringer, grid_bredde, grid_høyde,
            )
            if sti is not None and (sti or bot.posisjon == ordre.leveringssted):
                bevegelser = sti_til_aksjoner(sti, bot.posisjon)
                aksjoner.extend({"action": a} for a in bevegelser)
                aksjoner.append({"action": "drop_off"})
                return aksjoner[:20]

    # --- Fallback ---
    aksjoner.append({"action": "wait"})
    return aksjoner


def planlegg_aksjoner(
    game_state: dict,
    grid_bredde: int,
    grid_høyde: int,
) -> dict[str, list[dict]] | list[dict]:
    """
    Hovedstrategi. Returnerer:
    - Multi-bot:  dict[str, list[dict]]  (bot_id -> aksjonsliste)
    - Enkelt-bot: list[dict]             (bakoverkompatibelt)

    Kaller pathfinding internt.
    Aksjonformat: {"action": "move_right"} eller {"action": "pick_up", "item_id": "..."}
    """
    global _tildelte_ordrer
    _tildelte_ordrer = {}  # Nullstill tildelinger hver tur

    bots, ordrer, varer, hindringer = parse_spilltilstand(game_state)

    if not bots:
        logger.warning("Ingen bots i game_state — returnerer wait")
        return [{"action": "wait"}]

    enkelt_bot_modus = len(bots) == 1 and bots[0].bot_id in ("default", bots[0].bot_id)
    # Sjekk om vi kom fra enkelt-bot-format (game_state["bot"])
    er_enkelt_format = "bot" in game_state and "bots" not in game_state

    if len(bots) == 1 and er_enkelt_format:
        # Bakoverkompatibelt: returner flat liste
        aksjoner = _planlegg_for_enkelt_bot(
            bots[0], ordrer, varer, hindringer, grid_bredde, grid_høyde,
        )
        return aksjoner

    # --- Multi-bot-modus ---
    alle_aksjoner: dict[str, list[dict]] = {}
    reserverte_ordrer: set[str] = set()

    # Sorter bots etter hvem som har mest i inventaret (de bør prioriteres)
    bots_sortert = sorted(bots, key=lambda b: len(b.inventar), reverse=True)

    for bot in bots_sortert:
        # Samle andre bots sine posisjoner som dynamiske hindringer
        andre_posisjoner = [
            b.posisjon for b in bots if b.bot_id != bot.bot_id
        ]

        aksjoner = _planlegg_for_enkelt_bot(
            bot, ordrer, varer, hindringer, grid_bredde, grid_høyde,
            reserverte_ordrer=reserverte_ordrer,
            andre_bot_posisjoner=andre_posisjoner,
        )
        alle_aksjoner[bot.bot_id] = aksjoner

    return alle_aksjoner


def alle_varer_i_inventar(ordre: Ordre, inventar: list[str]) -> bool:
    """Sjekk om alle varer i ordren er i inventaret."""
    return all(vid in inventar for vid in ordre.varer)
