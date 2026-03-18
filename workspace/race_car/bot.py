"""
bot.py — WebSocket/HTTP-wrapper for Race Car-agenten (NM i AI 2026)

Kobler RegelbasertAgent til spillserveren via WebSocket (primær) eller HTTP-polling (fallback).

Tilkobling: wss://game.ainm.no/ws?token=<jwt_token>

Miljøvariabler:
    JWT_TOKEN   — Påkrevd. JWT-token for autentisering.
    MODE        — Valgfri. "ws" (standard) eller "http" for HTTP-polling.
    HTTP_URL    — Valgfri. URL for HTTP-polling (standard: https://game.ainm.no/api/race).
    LOG_LEVEL   — Valgfri. Loggnivå (standard: INFO).

Kjør:
    export JWT_TOKEN="ditt_token_her"
    python bot.py              # WebSocket-modus (standard)
    MODE=http python bot.py    # HTTP-polling fallback
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional

import websockets

try:
    import aiohttp
    AIOHTTP_TILGJENGELIG = True
except ImportError:
    AIOHTTP_TILGJENGELIG = False

from agent import RegelbasertAgent

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("race_car.bot")

# --- Konfigurasjon ---
JWT_TOKEN: str = os.environ.get("JWT_TOKEN", "SETT_TOKEN_HER")
WS_URL: str = f"wss://game.ainm.no/ws?token={JWT_TOKEN}"
HTTP_URL: str = os.environ.get("HTTP_URL", "https://game.ainm.no/api/race")
MODE: str = os.environ.get("MODE", "ws").lower()

# Tidsgrenser
MAKS_RESPONSTID_MS: float = 100.0  # Advarsel hvis over 100 ms
MAKS_RECONNECTS: int = 10
POLL_INTERVALL: float = 0.05  # 50 ms mellom HTTP-polls


# --- Hjelpefunksjoner ---

def valider_konfigurasjon():
    """Sjekk at nødvendig konfigurasjon er satt."""
    if JWT_TOKEN == "SETT_TOKEN_HER":
        raise ValueError(
            "JWT_TOKEN er ikke satt!\n"
            "Kjør: export JWT_TOKEN='ditt_token_her'\n"
            "Eller rediger JWT_TOKEN direkte i bot.py"
        )
    if MODE not in ("ws", "http"):
        raise ValueError(
            f"Ugyldig MODE: '{MODE}'. Bruk 'ws' (WebSocket) eller 'http' (HTTP-polling)."
        )
    if MODE == "http" and not AIOHTTP_TILGJENGELIG:
        raise ImportError(
            "HTTP-modus krever aiohttp. Installer med: pip install aiohttp"
        )


def ekstraher_sensorer(melding: dict) -> Optional[list[float]]:
    """
    Ekstraher sensordata fra servermelding.

    Støtter ulike meldingsformater:
    - {"sensors": [float, ...]}
    - {"type": "game_state", "sensors": [...]}
    - {"state": {"sensors": [...]}}
    """
    # Direkte sensors-felt
    if "sensors" in melding:
        return melding["sensors"]

    # Nestet i state-objekt
    state = melding.get("state", {})
    if isinstance(state, dict) and "sensors" in state:
        return state["sensors"]

    return None


def bygg_svar(aksjon: dict) -> str:
    """Bygg JSON-svar fra agentens aksjon."""
    return json.dumps({
        "type": "action",
        "steering": aksjon["steering"],
        "throttle": aksjon["throttle"],
        "brake": aksjon["brake"],
    })


class TidsMåler:
    """Holder oversikt over responstider og logger advarsler."""

    def __init__(self, grense_ms: float = MAKS_RESPONSTID_MS):
        self.grense_ms = grense_ms
        self.total_turer = 0
        self.trege_turer = 0
        self.total_tid_ms = 0.0
        self.maks_tid_ms = 0.0

    def mål(self, tid_ms: float):
        self.total_turer += 1
        self.total_tid_ms += tid_ms
        self.maks_tid_ms = max(self.maks_tid_ms, tid_ms)

        if tid_ms > self.grense_ms:
            self.trege_turer += 1
            logger.warning(
                f"Treg respons: {tid_ms:.1f} ms (grense: {self.grense_ms} ms) "
                f"— {self.trege_turer}/{self.total_turer} trege turer"
            )

    def oppsummering(self) -> str:
        if self.total_turer == 0:
            return "Ingen turer registrert."
        snitt = self.total_tid_ms / self.total_turer
        return (
            f"Tidsmåling: {self.total_turer} turer, "
            f"snitt {snitt:.1f} ms, maks {self.maks_tid_ms:.1f} ms, "
            f"{self.trege_turer} trege (>{self.grense_ms} ms)"
        )


# --- WebSocket-modus ---

async def kjør_ws():
    """Hovedløkke for WebSocket-modus med eksponentiell backoff ved reconnect."""
    agent = RegelbasertAgent()
    tidsmåler = TidsMåler()
    reconnect_forsøk = 0

    while reconnect_forsøk < MAKS_RECONNECTS:
        try:
            logger.info(f"Kobler til {WS_URL[:60]}...")
            async with websockets.connect(
                WS_URL,
                ping_interval=10,
                ping_timeout=5,
                close_timeout=5,
            ) as ws:
                logger.info("Tilkoblet! Venter på sensordata fra serveren...")
                reconnect_forsøk = 0  # Nullstill ved vellykket tilkobling

                async for rå_melding in ws:
                    try:
                        melding = json.loads(rå_melding)
                    except json.JSONDecodeError as e:
                        logger.error(f"Ugyldig JSON fra server: {e}")
                        continue

                    meldingstype = melding.get("type", "ukjent")

                    # --- Game over ---
                    if meldingstype == "game_over":
                        score = melding.get("score", "?")
                        tid = melding.get("time", "?")
                        logger.info(f"Spillet over! Score: {score}, tid: {tid}")
                        logger.info(tidsmåler.oppsummering())
                        return

                    # --- Serverfeil ---
                    if meldingstype == "error":
                        feilmelding = melding.get("message", melding.get("error", "ukjent"))
                        logger.error(f"Serverfeil: {feilmelding}")
                        continue

                    # --- Ekstraher sensorer ---
                    sensorer = ekstraher_sensorer(melding)
                    if sensorer is None:
                        logger.debug(f"Ingen sensordata i melding av type '{meldingstype}' — ignorerer")
                        continue

                    # --- Kjør agent med tidsmåling ---
                    start = time.monotonic()
                    aksjon = agent.velg_aksjon(sensorer)
                    tid_ms = (time.monotonic() - start) * 1000
                    tidsmåler.mål(tid_ms)

                    # --- Send aksjon ---
                    svar = bygg_svar(aksjon)
                    await ws.send(svar)
                    logger.debug(
                        f"Tur {agent.tur_teller}: "
                        f"steering={aksjon['steering']:.3f}, "
                        f"throttle={aksjon['throttle']:.3f}, "
                        f"brake={aksjon['brake']:.3f} "
                        f"({tid_ms:.1f} ms)"
                    )

        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Tilkobling lukket: {e}")
        except websockets.exceptions.InvalidURI:
            logger.error(f"Ugyldig WebSocket-URL: {WS_URL}")
            break
        except websockets.exceptions.InvalidStatusCode as e:
            logger.error(f"Server avviste tilkobling (HTTP {e.status_code}). Sjekk JWT_TOKEN.")
            if e.status_code in (401, 403):
                break  # Ikke prøv igjen med ugyldig token
        except OSError as e:
            logger.error(f"Nettverksfeil: {e}")
        except Exception as e:
            logger.exception(f"Uventet feil: {e}")

        reconnect_forsøk += 1
        ventetid = min(2 ** reconnect_forsøk, 30)
        logger.info(
            f"Reconnect om {ventetid}s "
            f"(forsøk {reconnect_forsøk}/{MAKS_RECONNECTS})..."
        )
        await asyncio.sleep(ventetid)

    logger.error("Maks antall reconnect-forsøk nådd. Avslutter.")
    logger.info(tidsmåler.oppsummering())


# --- HTTP-polling fallback ---

async def kjør_http():
    """Fallback-modus: HTTP-polling mot REST-API."""
    if not AIOHTTP_TILGJENGELIG:
        raise ImportError("HTTP-modus krever aiohttp. Installer med: pip install aiohttp")

    agent = RegelbasertAgent()
    tidsmåler = TidsMåler()
    reconnect_forsøk = 0

    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(headers=headers) as sesjon:
        logger.info(f"HTTP-polling startet mot {HTTP_URL}")

        while reconnect_forsøk < MAKS_RECONNECTS:
            try:
                # --- Hent tilstand ---
                async with sesjon.get(HTTP_URL) as resp:
                    if resp.status == 401:
                        logger.error("Ugyldig token (HTTP 401). Avslutter.")
                        break
                    if resp.status != 200:
                        logger.warning(f"HTTP {resp.status} — prøver igjen")
                        reconnect_forsøk += 1
                        await asyncio.sleep(min(2 ** reconnect_forsøk, 30))
                        continue

                    melding = await resp.json()

                # Nullstill reconnect ved suksess
                reconnect_forsøk = 0
                meldingstype = melding.get("type", "ukjent")

                # --- Game over ---
                if meldingstype == "game_over":
                    score = melding.get("score", "?")
                    logger.info(f"Spillet over! Score: {score}")
                    logger.info(tidsmåler.oppsummering())
                    return

                # --- Serverfeil ---
                if meldingstype == "error":
                    logger.error(f"Serverfeil: {melding.get('message', 'ukjent')}")
                    await asyncio.sleep(POLL_INTERVALL)
                    continue

                # --- Ekstraher sensorer ---
                sensorer = ekstraher_sensorer(melding)
                if sensorer is None:
                    await asyncio.sleep(POLL_INTERVALL)
                    continue

                # --- Kjør agent med tidsmåling ---
                start = time.monotonic()
                aksjon = agent.velg_aksjon(sensorer)
                tid_ms = (time.monotonic() - start) * 1000
                tidsmåler.mål(tid_ms)

                # --- Send aksjon ---
                async with sesjon.post(HTTP_URL, json={
                    "type": "action",
                    "steering": aksjon["steering"],
                    "throttle": aksjon["throttle"],
                    "brake": aksjon["brake"],
                }) as resp:
                    if resp.status != 200:
                        logger.warning(f"Aksjon avvist: HTTP {resp.status}")

                logger.debug(
                    f"Tur {agent.tur_teller}: "
                    f"steering={aksjon['steering']:.3f}, "
                    f"throttle={aksjon['throttle']:.3f}, "
                    f"brake={aksjon['brake']:.3f} "
                    f"({tid_ms:.1f} ms)"
                )

                await asyncio.sleep(POLL_INTERVALL)

            except aiohttp.ClientError as e:
                logger.error(f"HTTP-feil: {e}")
                reconnect_forsøk += 1
                ventetid = min(2 ** reconnect_forsøk, 30)
                logger.info(f"Prøver igjen om {ventetid}s ({reconnect_forsøk}/{MAKS_RECONNECTS})...")
                await asyncio.sleep(ventetid)
            except Exception as e:
                logger.exception(f"Uventet feil i HTTP-modus: {e}")
                reconnect_forsøk += 1
                await asyncio.sleep(min(2 ** reconnect_forsøk, 30))

    logger.error("Maks antall reconnect-forsøk nådd (HTTP). Avslutter.")
    logger.info(tidsmåler.oppsummering())


# --- Oppstart ---

async def main():
    """Velg modus og start boten."""
    valider_konfigurasjon()

    if MODE == "ws":
        logger.info("Starter i WebSocket-modus")
        await kjør_ws()
    else:
        logger.info("Starter i HTTP-polling fallback-modus")
        await kjør_http()


if __name__ == "__main__":
    valider_konfigurasjon()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stoppet manuelt (Ctrl+C).")
