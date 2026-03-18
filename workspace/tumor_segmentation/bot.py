"""
bot.py — Server-integrasjon for tumor-segmenteringskonkurranse

WebSocket-klient som kobler til konkurranse-serveren, mottar bilder,
kjører segmentering og sender tilbake maskeresultater.

Protokoll:
    1. Koble til wss://game.ainm.no/ws?token=JWT_TOKEN
    2. Motta meldinger (JSON):
       - {"type": "image", "data": "<base64>", "image_id": "..."}
       - {"type": "image_url", "url": "...", "image_id": "..."}
       - {"type": "game_over", "scores": {...}}
       - {"type": "error", "message": "..."}
    3. Send svar:
       - {"type": "segmentation", "image_id": "...", "mask": "<base64 PNG>"}
       - {"type": "segmentation", "image_id": "...", "rle": [start, len, ...]}

Bruk:
    export JWT_TOKEN="din_token_her"
    python bot.py [--url wss://game.ainm.no/ws] [--no-tta] [--format png|rle]

Krav: pip install websockets requests
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch

from model import last_modell, BILDE_STØRRELSE
from inference import (
    forbehandle_base64,
    forbehandle_numpy,
    segmenter_tensor,
    maske_til_base64_png,
    maske_til_rle,
    ENHET,
)

logger = logging.getLogger(__name__)

# --- Konfigurasjon ---
STANDARD_SERVER_URL = "wss://game.ainm.no/ws"
MAKS_RECONNECT_FORSØK = 10
RECONNECT_FORSINKELSE_SEK = 2.0
PING_INTERVALL_SEK = 20
PING_TIMEOUT_SEK = 10


class TumorSegmenteringsBot:
    """
    WebSocket-klient for tumor-segmenteringskonkurranse.

    Håndterer tilkobling, bildeprosessering og resultatlevering
    med automatisk reconnect ved tilkoblingsproblemer.
    """

    def __init__(
        self,
        server_url: str = STANDARD_SERVER_URL,
        token: Optional[str] = None,
        bruk_tta: bool = True,
        respons_format: str = "png",  # "png" eller "rle"
    ):
        self.server_url = server_url
        self.token = token or os.environ.get("JWT_TOKEN")
        if not self.token:
            raise ValueError(
                "JWT_TOKEN er ikke satt! Sett miljøvariabel JWT_TOKEN "
                "eller bruk --token argumentet."
            )

        self.bruk_tta = bruk_tta
        self.respons_format = respons_format
        self.modell = None
        self.ws = None
        self.kjører = False
        self.reconnect_forsøk = 0

        # Statistikk
        self.antall_behandlet = 0
        self.total_inferenstid = 0.0
        self.feil_teller = 0

    def _last_modell(self):
        """Last segmenteringsmodell med fallback-hierarki."""
        logger.info("Laster segmenteringsmodell...")
        self.modell = last_modell(enhet=ENHET)
        self.modell.eval()
        logger.info(f"Modell klar på enhet: {ENHET}")

    def _bygg_tilkoblings_url(self) -> str:
        """Bygg WebSocket-URL med token."""
        separator = "&" if "?" in self.server_url else "?"
        return f"{self.server_url}{separator}token={self.token}"

    async def _prosesser_bilde_melding(self, melding: dict) -> dict:
        """
        Prosesser en bildemelding og returner segmenteringsresultat.

        Støtter to formater:
            - "data": base64-kodet bildedata direkte
            - "url": URL til bilde som lastes ned
        """
        bilde_id = melding.get("image_id", "ukjent")
        logger.info(f"Mottok bilde: {bilde_id}")

        try:
            # Hent bildedata
            if "data" in melding:
                tensor = forbehandle_base64(melding["data"])
            elif "url" in melding:
                tensor = await self._last_bilde_fra_url(melding["url"])
            else:
                raise ValueError("Melding inneholder verken 'data' eller 'url'")

            # Kjør segmentering
            maske, inferenstid = segmenter_tensor(
                tensor,
                self.modell,
                bruk_tta=self.bruk_tta,
            )

            # Oppdater statistikk
            self.antall_behandlet += 1
            self.total_inferenstid += inferenstid

            # Bygg respons
            respons = {
                "type": "segmentation",
                "image_id": bilde_id,
            }

            if self.respons_format == "rle":
                respons["rle"] = maske_til_rle(maske)
            else:
                respons["mask"] = maske_til_base64_png(maske)

            logger.info(
                f"Bilde {bilde_id} segmentert på {inferenstid:.2f}s "
                f"({self.antall_behandlet} totalt, "
                f"snitt: {self.total_inferenstid / self.antall_behandlet:.2f}s)"
            )

            return respons

        except Exception as e:
            self.feil_teller += 1
            logger.error(f"Feil ved prosessering av bilde {bilde_id}: {e}")
            return {
                "type": "error",
                "image_id": bilde_id,
                "message": str(e),
            }

    async def _last_bilde_fra_url(self, url: str) -> torch.Tensor:
        """Last ned bilde fra URL og konverter til tensor."""
        from PIL import Image
        import io

        logger.debug(f"Laster ned bilde fra: {url}")

        # Kjør nedlasting i en executor for å ikke blokkere event-løkken
        loop = asyncio.get_event_loop()
        respons = await loop.run_in_executor(
            None,
            lambda: requests.get(url, timeout=15)
        )
        respons.raise_for_status()

        bilde = Image.open(io.BytesIO(respons.content))
        if bilde.mode != 'L':
            bilde = bilde.convert('L')

        arr = np.array(bilde, dtype=np.float32)
        return forbehandle_numpy(arr)

    async def _håndter_melding(self, rå_melding: str):
        """Parse og håndter en innkommende melding fra serveren."""
        try:
            melding = json.loads(rå_melding)
        except json.JSONDecodeError as e:
            logger.error(f"Ugyldig JSON fra server: {e}")
            return

        meldingstype = melding.get("type", "ukjent")

        if meldingstype in ("image", "image_url"):
            respons = await self._prosesser_bilde_melding(melding)
            await self._send_melding(respons)

        elif meldingstype == "game_over":
            logger.info("=" * 50)
            logger.info("KONKURRANSE AVSLUTTET!")
            logger.info(f"Resultater: {json.dumps(melding.get('scores', {}), indent=2)}")
            logger.info(f"Bilder behandlet: {self.antall_behandlet}")
            if self.antall_behandlet > 0:
                logger.info(
                    f"Gjennomsnittlig inferenstid: "
                    f"{self.total_inferenstid / self.antall_behandlet:.2f}s"
                )
            logger.info(f"Feil: {self.feil_teller}")
            logger.info("=" * 50)
            self.kjører = False

        elif meldingstype == "error":
            logger.error(f"Serverfeil: {melding.get('message', 'Ingen detaljer')}")

        elif meldingstype == "ping":
            await self._send_melding({"type": "pong"})

        elif meldingstype == "connected":
            logger.info(f"Tilkoblet server: {melding.get('message', '')}")
            self.reconnect_forsøk = 0

        else:
            logger.warning(f"Ukjent meldingstype: {meldingstype}")
            logger.debug(f"Full melding: {rå_melding[:500]}")

    async def _send_melding(self, melding: dict):
        """Send en melding til serveren via WebSocket."""
        if self.ws is None:
            logger.error("Kan ikke sende — ingen WebSocket-tilkobling!")
            return

        try:
            rå = json.dumps(melding)
            await self.ws.send(rå)
            logger.debug(f"Sendt melding type={melding.get('type')}")
        except Exception as e:
            logger.error(f"Feil ved sending av melding: {e}")

    async def koble_til_og_kjør(self):
        """
        Hovedløkke: koble til serveren og prosesser meldinger.

        Implementerer automatisk reconnect med eksponentiell backoff
        opptil MAKS_RECONNECT_FORSØK ganger.
        """
        try:
            import websockets
        except ImportError:
            logger.error(
                "websockets-pakken er ikke installert!\n"
                "Installer med: pip install websockets"
            )
            sys.exit(1)

        self._last_modell()
        self.kjører = True

        url = self._bygg_tilkoblings_url()
        logger.info(f"Kobler til: {self.server_url}")

        while self.kjører and self.reconnect_forsøk < MAKS_RECONNECT_FORSØK:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=PING_INTERVALL_SEK,
                    ping_timeout=PING_TIMEOUT_SEK,
                    max_size=50 * 1024 * 1024,  # 50 MB maks meldingsstørrelse
                ) as ws:
                    self.ws = ws
                    self.reconnect_forsøk = 0
                    logger.info("WebSocket-tilkobling opprettet!")

                    async for melding in ws:
                        await self._håndter_melding(melding)

            except asyncio.CancelledError:
                logger.info("Bot avbrutt — avslutter...")
                self.kjører = False
                break

            except Exception as e:
                self.reconnect_forsøk += 1
                forsinkelse = min(
                    RECONNECT_FORSINKELSE_SEK * (2 ** (self.reconnect_forsøk - 1)),
                    60.0,  # Maks 60 sekunders forsinkelse
                )
                logger.warning(
                    f"Tilkobling tapt: {e}. "
                    f"Forsøker reconnect {self.reconnect_forsøk}/{MAKS_RECONNECT_FORSØK} "
                    f"om {forsinkelse:.0f}s..."
                )
                await asyncio.sleep(forsinkelse)

            finally:
                self.ws = None

        if self.reconnect_forsøk >= MAKS_RECONNECT_FORSØK:
            logger.error(
                f"Ga opp etter {MAKS_RECONNECT_FORSØK} reconnect-forsøk. "
                "Sjekk nettverkstilkobling og server-status."
            )

    def _skriv_statistikk(self):
        """Skriv ut sluttstatistikk."""
        logger.info("\n--- Sesjonsstatistikk ---")
        logger.info(f"Bilder behandlet: {self.antall_behandlet}")
        if self.antall_behandlet > 0:
            logger.info(
                f"Gjennomsnittlig inferenstid: "
                f"{self.total_inferenstid / self.antall_behandlet:.2f}s"
            )
        logger.info(f"Feil: {self.feil_teller}")
        logger.info(f"Enhet: {ENHET}")
        logger.info(f"TTA: {'på' if self.bruk_tta else 'av'}")
        logger.info(f"Responsformat: {self.respons_format}")


# --- HTTP-basert fallback (for servere uten WebSocket) ---

class HTTPSegmenteringsBot:
    """
    HTTP-basert klient som alternativ til WebSocket.

    Bruker polling: spør serveren etter nye bilder, sender resultater tilbake.
    Enklere, men høyere latens enn WebSocket.
    """

    def __init__(
        self,
        base_url: str = "https://game.ainm.no",
        token: Optional[str] = None,
        bruk_tta: bool = True,
        respons_format: str = "png",
        poll_intervall: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("JWT_TOKEN")
        if not self.token:
            raise ValueError("JWT_TOKEN er ikke satt!")

        self.bruk_tta = bruk_tta
        self.respons_format = respons_format
        self.poll_intervall = poll_intervall
        self.modell = None
        self.sesjon = requests.Session()
        self.sesjon.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        })

        # Statistikk
        self.antall_behandlet = 0
        self.total_inferenstid = 0.0

    def _last_modell(self):
        """Last segmenteringsmodell."""
        logger.info("Laster segmenteringsmodell...")
        self.modell = last_modell(enhet=ENHET)
        self.modell.eval()

    def hent_neste_bilde(self) -> Optional[dict]:
        """Hent neste bilde fra serveren via HTTP GET."""
        try:
            respons = self.sesjon.get(
                f"{self.base_url}/api/next-image",
                timeout=10,
            )
            if respons.status_code == 200:
                return respons.json()
            elif respons.status_code == 204:
                return None  # Ingen nye bilder
            else:
                logger.warning(f"Uventet statuskode: {respons.status_code}")
                return None
        except requests.RequestException as e:
            logger.error(f"HTTP-feil ved henting av bilde: {e}")
            return None

    def send_resultat(self, bilde_id: str, maske: np.ndarray) -> bool:
        """Send segmenteringsresultat tilbake til serveren."""
        try:
            data = {
                "image_id": bilde_id,
            }

            if self.respons_format == "rle":
                data["rle"] = maske_til_rle(maske)
            else:
                data["mask"] = maske_til_base64_png(maske)

            respons = self.sesjon.post(
                f"{self.base_url}/api/submit",
                json=data,
                timeout=10,
            )
            return respons.status_code == 200

        except requests.RequestException as e:
            logger.error(f"HTTP-feil ved sending av resultat: {e}")
            return False

    def kjør(self):
        """Hovedløkke for HTTP-basert polling."""
        self._last_modell()
        logger.info(f"HTTP-bot startet. Poller {self.base_url} hvert {self.poll_intervall}s")

        try:
            while True:
                bilde_data = self.hent_neste_bilde()

                if bilde_data is None:
                    time.sleep(self.poll_intervall)
                    continue

                if bilde_data.get("type") == "game_over":
                    logger.info("Konkurranse avsluttet!")
                    logger.info(f"Resultater: {json.dumps(bilde_data.get('scores', {}), indent=2)}")
                    break

                bilde_id = bilde_data.get("image_id", "ukjent")

                # Prosesser bilde
                if "data" in bilde_data:
                    tensor = forbehandle_base64(bilde_data["data"])
                elif "url" in bilde_data:
                    from PIL import Image
                    import io as _io
                    resp = requests.get(bilde_data["url"], timeout=15)
                    resp.raise_for_status()
                    bilde = Image.open(_io.BytesIO(resp.content)).convert('L')
                    arr = np.array(bilde, dtype=np.float32)
                    tensor = forbehandle_numpy(arr)
                else:
                    logger.warning(f"Ukjent bildeformat for {bilde_id}")
                    continue

                maske, tid = segmenter_tensor(
                    tensor, self.modell, bruk_tta=self.bruk_tta,
                )

                self.antall_behandlet += 1
                self.total_inferenstid += tid

                if self.send_resultat(bilde_id, maske):
                    logger.info(
                        f"Bilde {bilde_id} sendt OK ({tid:.2f}s, "
                        f"#{self.antall_behandlet})"
                    )
                else:
                    logger.error(f"Klarte ikke sende resultat for {bilde_id}")

        except KeyboardInterrupt:
            logger.info("Avbrutt av bruker.")

        logger.info(f"Behandlet {self.antall_behandlet} bilder totalt.")


# --- Kommandolinjegrensesnitt ---

def parse_argumenter() -> argparse.Namespace:
    """Parse kommandolinjeargumenter."""
    parser = argparse.ArgumentParser(
        description="Tumor-segmenteringsbot for konkurranse-server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Eksempler:
    # WebSocket-tilkobling (standard)
    export JWT_TOKEN="din_token"
    python bot.py

    # Med egendefinert URL og uten TTA (raskere)
    python bot.py --url wss://annen-server.no/ws --no-tta

    # HTTP-polling som fallback
    python bot.py --http --url https://game.ainm.no

    # Med RLE-format for mindre datamengde
    python bot.py --format rle
        """,
    )

    parser.add_argument(
        "--url",
        default=STANDARD_SERVER_URL,
        help=f"Server-URL (standard: {STANDARD_SERVER_URL})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="JWT-token (alternativ til JWT_TOKEN miljøvariabel)",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Deaktiver test-time augmentation (raskere, lavere nøyaktighet)",
    )
    parser.add_argument(
        "--format",
        choices=["png", "rle"],
        default="png",
        help="Responsformat for masker (standard: png)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Bruk HTTP-polling i stedet for WebSocket",
    )
    parser.add_argument(
        "--poll-intervall",
        type=float,
        default=1.0,
        help="Polling-intervall i sekunder for HTTP-modus (standard: 1.0)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Aktiver detaljert logging (DEBUG-nivå)",
    )

    return parser.parse_args()


def main():
    """Hovedinngang for boten."""
    args = parse_argumenter()

    # Sett opp logging
    log_nivå = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_nivå,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    token = args.token or os.environ.get("JWT_TOKEN")
    if not token:
        logger.error(
            "Ingen JWT-token funnet!\n"
            "Sett miljøvariabel: export JWT_TOKEN='din_token'\n"
            "Eller bruk: python bot.py --token din_token"
        )
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Tumor-segmenteringsbot")
    logger.info(f"Server: {args.url}")
    logger.info(f"Modus: {'HTTP' if args.http else 'WebSocket'}")
    logger.info(f"TTA: {'av' if args.no_tta else 'på'}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Enhet: {ENHET}")
    logger.info("=" * 50)

    if args.http:
        # HTTP-modus
        bot = HTTPSegmenteringsBot(
            base_url=args.url,
            token=token,
            bruk_tta=not args.no_tta,
            respons_format=args.format,
            poll_intervall=args.poll_intervall,
        )
        bot.kjør()
    else:
        # WebSocket-modus (standard)
        bot = TumorSegmenteringsBot(
            server_url=args.url,
            token=token,
            bruk_tta=not args.no_tta,
            respons_format=args.format,
        )

        # Håndter graceful shutdown
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def _shutdown_handler(sig, frame):
            logger.info(f"\nMottok signal {sig} — avslutter...")
            bot.kjører = False
            bot._skriv_statistikk()

        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)

        try:
            loop.run_until_complete(bot.koble_til_og_kjør())
        except KeyboardInterrupt:
            logger.info("Avbrutt med Ctrl+C")
        finally:
            bot._skriv_statistikk()
            loop.close()


if __name__ == "__main__":
    main()
