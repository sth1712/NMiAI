"""
agent.py — Regelbasert kjøreagent for NM i AI 2026 — Race Car

Bilen har 16 sensorer som måler avstand til veggen i ulike vinkler.
Mål: holde bilen på banen så lenge som mulig (1 minutt).

Sensor-layout (antatt): 16 stråler jevnt fordelt 0–360° rundt bilen,
eller 0–180° foran. Tilpass SENSOR_VINKLER til faktisk API.

TODO: Bytt ut regelbasert logikk med RL-agent (f.eks. PPO via stable-baselines3).
"""

import math
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# --- Konfigurasjon ---
ANTALL_SENSORER = 16

# Sensorvinkler i grader relativt til kjøreretning (0° = rett frem)
# Antatt symmetrisk layout — juster til faktisk API
SENSOR_VINKLER = [i * (360 / ANTALL_SENSORER) for i in range(ANTALL_SENSORER)]

# Terskelverdi: hvis sensor < FARESONE er vi nær veggen
FARESONE = 50.0
KRITISK_SONE = 20.0

# Gass- og styringsverdier (typisk -1.0 til 1.0)
FULL_GASS = 1.0
HALV_GASS = 0.5
LAV_GASS = 0.2
FULL_SVING = 1.0


class RegelbasertAgent:
    """
    Regelbasert kjøreagent.

    Input: sensorverdier (liste med 16 avstander)
    Output: {"steering": float, "throttle": float, "brake": float}
        - steering: -1.0 (venstre) til 1.0 (høyre)
        - throttle: 0.0 til 1.0
        - brake:    0.0 til 1.0
    """

    def __init__(self):
        self.tur_teller = 0
        self.forrige_aksjon = {"steering": 0.0, "throttle": FULL_GASS, "brake": 0.0}

    def velg_aksjon(self, sensorer: list[float]) -> dict:
        """
        Hovedmetode. Tar sensorverdier og returnerer styrekommando.
        """
        self.tur_teller += 1

        if len(sensorer) != ANTALL_SENSORER:
            logger.warning(f"Forventet {ANTALL_SENSORER} sensorer, fikk {len(sensorer)}")

        # Del sensorene i sektorer
        foran = self._sektor(sensorer, start=14, slutt=2)     # Foran (±45°)
        venstre = self._sektor(sensorer, start=10, slutt=14)  # Venstre side
        høyre = self._sektor(sensorer, start=2, slutt=6)      # Høyre side
        skrå_venstre = self._sektor(sensorer, start=12, slutt=14)
        skrå_høyre = self._sektor(sensorer, start=2, slutt=4)

        min_foran = min(foran) if foran else 999
        min_venstre = min(venstre) if venstre else 999
        min_høyre = min(høyre) if høyre else 999

        steering = 0.0
        throttle = FULL_GASS
        brake = 0.0

        # --- Kritisk fare foran → bremse og sving mot åpning ---
        if min_foran < KRITISK_SONE:
            brake = 0.8
            throttle = 0.0
            if min_venstre > min_høyre:
                steering = -FULL_SVING  # Sving venstre
            else:
                steering = FULL_SVING   # Sving høyre
            logger.debug(f"Tur {self.tur_teller}: KRITISK — brems og sving")

        # --- Fare foran → reduser fart og styr unna ---
        elif min_foran < FARESONE:
            throttle = LAV_GASS
            if min_venstre > min_høyre:
                steering = -0.7
            else:
                steering = 0.7
            logger.debug(f"Tur {self.tur_teller}: Fare foran — styrer unna")

        # --- Veggen nær til venstre → styr høyre ---
        elif min_venstre < FARESONE:
            steering = 0.5 * (1 - min_venstre / FARESONE)
            throttle = HALV_GASS
            logger.debug(f"Tur {self.tur_teller}: Nær vegg venstre")

        # --- Veggen nær til høyre → styr venstre ---
        elif min_høyre < FARESONE:
            steering = -0.5 * (1 - min_høyre / FARESONE)
            throttle = HALV_GASS
            logger.debug(f"Tur {self.tur_teller}: Nær vegg høyre")

        # --- Normalt: kjør rett frem med maksimal fart ---
        else:
            # Finjustering: hold midten basert på venstre/høyre balanse
            diff = min_høyre - min_venstre
            steering = -diff / (min_venstre + min_høyre + 1e-6) * 0.3
            throttle = FULL_GASS
            logger.debug(f"Tur {self.tur_teller}: Kjør fremover")

        # Glatt overgang (unngå brå rykk)
        steering = self._glatt(self.forrige_aksjon["steering"], steering, alpha=0.4)

        aksjon = {
            "steering": round(max(-1.0, min(1.0, steering)), 4),
            "throttle": round(max(0.0, min(1.0, throttle)), 4),
            "brake": round(max(0.0, min(1.0, brake)), 4),
        }
        self.forrige_aksjon = aksjon
        return aksjon

    def _sektor(self, sensorer: list[float], start: int, slutt: int) -> list[float]:
        """
        Hent sensorverdier for en sektor (indekser, støtter wrap-around).
        """
        n = len(sensorer)
        if start <= slutt:
            return sensorer[start:slutt]
        else:
            return sensorer[start:] + sensorer[:slutt]

    def _glatt(self, forrige: float, ny: float, alpha: float = 0.3) -> float:
        """Eksponensiell utjevning for å unngå rykk."""
        return forrige * (1 - alpha) + ny * alpha


# --- Integrasjonsgrensesnitt ---
# TODO: Tilpass til faktisk spillserver-API (REST/WebSocket/gym-env)

def kjør_episode(agent: RegelbasertAgent, hent_tilstand, send_aksjon, maks_tid: float = 60.0):
    """
    Kjør én episode (1 minutt).

    hent_tilstand: callable som returnerer {"sensors": [float, ...], "speed": float, ...}
    send_aksjon:   callable som tar {"steering": float, "throttle": float, "brake": float}
    """
    start = time.monotonic()
    tur = 0

    while time.monotonic() - start < maks_tid:
        tilstand = hent_tilstand()
        sensorer = tilstand.get("sensors", [0.0] * ANTALL_SENSORER)
        aksjon = agent.velg_aksjon(sensorer)
        send_aksjon(aksjon)
        tur += 1

    logger.info(f"Episode ferdig etter {tur} turer og {maks_tid}s")


# --- Enkel testmodus ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    agent = RegelbasertAgent()

    # Simuler noen testtilstander
    testkasus = [
        ([100.0] * 16, "Åpen bane"),
        ([5.0] * 4 + [100.0] * 8 + [5.0] * 4, "Nær vegger foran/bak"),
        ([10.0] * 2 + [100.0] * 12 + [10.0] * 2, "Nær vegger høyre/venstre"),
        ([8.0] * 16, "Kritisk — vegger overalt"),
    ]

    for sensorer, beskrivelse in testkasus:
        aksjon = agent.velg_aksjon(sensorer)
        print(f"\n{beskrivelse}")
        print(f"  Sensorer (min): {min(sensorer):.1f}")
        print(f"  Aksjon: {aksjon}")
