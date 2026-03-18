"""
bot.py — Hoved-bot med WebSocket-tilkobling for NM i AI 2026 — Grocery Bot

Tilkobling: wss://game.ainm.no/ws?token=<jwt_token>
Server sender game_state JSON → bot svarer med aksjonsliste innen 2 sekunder.

Støtter multi-bot: Hvis serveren sender game_state med "bots" (liste),
returneres aksjoner per bot.

Kjør:
    python bot.py

Sett JWT_TOKEN som miljøvariabel eller direkte i koden (se TODO nedenfor).
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

import websockets

from strategy import planlegg_aksjoner

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Konfigurasjon ---
# TODO: Sett JWT_TOKEN som miljøvariabel: export JWT_TOKEN="ditt_token_her"
JWT_TOKEN: str = os.environ.get("JWT_TOKEN", "SETT_TOKEN_HER")
WS_URL: str = f"wss://game.ainm.no/ws?token={JWT_TOKEN}"

# Gridstørrelse — oppdater etter første game_state hvis serveren sender dette
GRID_BREDDE: int = 20
GRID_HØYDE: int = 20

# Maks tid per tur i sekunder (serveren krever svar innen 2 sek)
MAKS_RESPONSTID: float = 1.8

# Antall forsøk ved tilkoblingsfeil
MAKS_RECONNECTS: int = 10


def _formater_svar(resultat: dict[str, list[dict]] | list[dict]) -> dict:
    """
    Formater strategiresultatet til et svar serveren aksepterer.

    Enkelt-bot (list[dict]):
        {"type": "actions", "actions": [...]}

    Multi-bot (dict[str, list[dict]]):
        {"type": "actions", "actions": {"bot_0": [...], "bot_1": [...]}}

    Fallback: Hvis serveren forventer en flat liste med bot_id per aksjon,
    konverterer vi dict til: [{"bot_id": "bot_0", "action": "move_right"}, ...]
    """
    if isinstance(resultat, list):
        # Enkelt-bot-format — bakoverkompatibelt
        return {
            "type": "actions",
            "actions": resultat,
        }

    # Multi-bot: returner dict-format som primært format
    return {
        "type": "actions",
        "actions": resultat,
    }


def _formater_svar_flat(resultat: dict[str, list[dict]]) -> dict:
    """
    Alternativt format: flat liste med bot_id per aksjon.
    Bruk dette hvis serveren ikke støtter dict-format for aksjoner.

    Returnerer: {"type": "actions", "actions": [
        {"bot_id": "bot_0", "action": "move_right"},
        {"bot_id": "bot_1", "action": "wait"},
        ...
    ]}
    """
    flat_aksjoner = []
    for bot_id, aksjoner in resultat.items():
        for aksjon in aksjoner:
            flat_aksjon = dict(aksjon)
            flat_aksjon["bot_id"] = bot_id
            flat_aksjoner.append(flat_aksjon)

    return {
        "type": "actions",
        "actions": flat_aksjoner,
    }


def behandle_game_state(melding: dict) -> Optional[dict]:
    """
    Tar imot en game_state-melding og returnerer ferdig formatert svar.
    Returnerer None hvis meldingen ikke er en game_state.

    Håndterer både enkelt-bot og multi-bot game_state.
    """
    if melding.get("type") != "game_state":
        logger.debug(f"Ignorerer meldingstype: {melding.get('type')}")
        return None

    # Dynamisk oppdatering av gridstørrelse hvis serveren sender den
    global GRID_BREDDE, GRID_HØYDE
    if "grid_width" in melding:
        GRID_BREDDE = melding["grid_width"]
    if "grid_height" in melding:
        GRID_HØYDE = melding["grid_height"]

    start = time.monotonic()
    resultat = planlegg_aksjoner(melding, GRID_BREDDE, GRID_HØYDE)
    elapsed = time.monotonic() - start

    # Logg antall aksjoner
    if isinstance(resultat, dict):
        totalt = sum(len(a) for a in resultat.values())
        logger.info(
            f"Planla aksjoner for {len(resultat)} bots "
            f"({totalt} aksjoner totalt) på {elapsed:.3f}s"
        )
    else:
        logger.info(f"Planla {len(resultat)} aksjoner på {elapsed:.3f}s")

    if elapsed > MAKS_RESPONSTID:
        logger.warning(f"ADVARSEL: Planlegging tok {elapsed:.2f}s — over grensen!")

    return _formater_svar(resultat)


async def kjør_bot():
    """Hovedløkke: kobler til server og håndterer meldinger."""
    reconnect_forsøk = 0

    while reconnect_forsøk < MAKS_RECONNECTS:
        try:
            logger.info(f"Kobler til {WS_URL}")
            async with websockets.connect(
                WS_URL,
                ping_interval=10,
                ping_timeout=5,
                close_timeout=5,
            ) as ws:
                logger.info("Tilkoblet! Venter på game_state...")
                reconnect_forsøk = 0  # Nullstill ved vellykket tilkobling

                async for rå_melding in ws:
                    try:
                        melding = json.loads(rå_melding)
                    except json.JSONDecodeError as e:
                        logger.error(f"Ugyldig JSON: {e}")
                        continue

                    meldingstype = melding.get("type", "ukjent")

                    if meldingstype == "game_over":
                        score = melding.get("score", "?")
                        logger.info(f"Spillet over! Score: {score}")
                        return

                    if meldingstype == "error":
                        logger.error(f"Serverfeil: {melding.get('message')}")
                        continue

                    svar_dict = behandle_game_state(melding)
                    if svar_dict is None:
                        continue

                    svar = json.dumps(svar_dict)
                    await ws.send(svar)
                    logger.debug(f"Sendte: {svar[:200]}")

        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Tilkobling lukket: {e}. Forsøker igjen...")
        except websockets.exceptions.InvalidURI:
            logger.error(f"Ugyldig WebSocket-URL: {WS_URL}")
            break
        except OSError as e:
            logger.error(f"Nettverksfeil: {e}")
        except Exception as e:
            logger.exception(f"Uventet feil: {e}")

        reconnect_forsøk += 1
        ventetid = min(2 ** reconnect_forsøk, 30)
        logger.info(f"Venter {ventetid}s før neste forsøk ({reconnect_forsøk}/{MAKS_RECONNECTS})...")
        await asyncio.sleep(ventetid)

    logger.error("Maks antall reconnect-forsøk nådd. Avslutter.")


def valider_konfigurasjon():
    """Sjekk at nødvendig konfigurasjon er satt."""
    if JWT_TOKEN == "SETT_TOKEN_HER":
        raise ValueError(
            "JWT_TOKEN er ikke satt!\n"
            "Kjør: export JWT_TOKEN='ditt_token_her'\n"
            "Eller rediger JWT_TOKEN direkte i bot.py"
        )


if __name__ == "__main__":
    valider_konfigurasjon()
    try:
        asyncio.run(kjør_bot())
    except KeyboardInterrupt:
        logger.info("Bot stoppet manuelt.")
