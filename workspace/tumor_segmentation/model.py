"""
model.py — U-Net segmenteringsmodell for svulster i MIP-PET-bilder

Bruker MONAI sin innebygde U-Net implementasjon.
Input: 2D MIP-PET-bilde (grånivå eller normalisert float)
Output: Binært segmenteringskart (0 = bakgrunn, 1 = svulst)

Krav: pip install monai torch torchvision
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm

logger = logging.getLogger(__name__)

# --- Konfigurasjon ---
# TODO: Tilpass til faktisk inputstørrelse fra konkurransen
BILDE_STØRRELSE = (512, 512)     # (H, W) — juster ved behov
ANTALL_KANALER_INN = 1           # Grånivå PET-bilde
ANTALL_KLASSER = 2               # Bakgrunn + svulst

# Standardsti for lagrede modellvekter
MODELL_STI = Path("modell_vekter.pth")


def bygg_unet(
    spatial_dims: int = 2,
    inn_kanaler: int = ANTALL_KANALER_INN,
    ut_kanaler: int = ANTALL_KLASSER,
    kanaler: tuple = (16, 32, 64, 128, 256),
    strides: tuple = (2, 2, 2, 2),
    normalisering: str = Norm.BATCH,
) -> UNet:
    """
    Bygg MONAI U-Net.

    Arkitektur:
    - 5 enkodernivåer med batch normalisering
    - Standardstørrelse for 512×512 PET-bilder
    - Kan justeres for større/mindre bilder
    """
    modell = UNet(
        spatial_dims=spatial_dims,
        in_channels=inn_kanaler,
        out_channels=ut_kanaler,
        channels=kanaler,
        strides=strides,
        num_res_units=2,
        norm=normalisering,
        dropout=0.1,
    )
    logger.info(
        f"U-Net bygget: inn={inn_kanaler}, ut={ut_kanaler}, "
        f"kanaler={kanaler}, strides={strides}"
    )
    return modell


def last_modell(sti: Path = MODELL_STI, enhet: Optional[str] = None) -> UNet:
    """
    Last modellvekter fra fil.

    TODO: Tren modell på PET-treningsdata og lagre vekter til MODELL_STI.
    """
    if enhet is None:
        enhet = "cuda" if torch.cuda.is_available() else "cpu"

    modell = bygg_unet()
    if not sti.exists():
        logger.warning(
            f"Ingen forhåndstrente vekter funnet på {sti}. "
            "Bruker tilfeldig initialisering (IKKE klar for produksjon!).\n"
            "TODO: Tren modellen og lagre vekter."
        )
        return modell.to(enhet)

    tilstand = torch.load(sti, map_location=enhet)
    modell.load_state_dict(tilstand)
    modell.eval()
    logger.info(f"Modellvekter lastet fra {sti} på enhet: {enhet}")
    return modell.to(enhet)


def lagre_modell(modell: UNet, sti: Path = MODELL_STI):
    """Lagre modellvekter til disk."""
    torch.save(modell.state_dict(), sti)
    logger.info(f"Modellvekter lagret til {sti}")


def tren_modell(
    modell: UNet,
    treningsdata,    # DataLoader
    valideringsdata, # DataLoader
    epoker: int = 50,
    læringshastighet: float = 1e-4,
    enhet: str = "cuda",
) -> UNet:
    """
    Treningstløkke med Dice-tap.

    TODO: Koble til faktisk PET-treningsdata.
    Se inference.py for datasett-oppsett.
    """
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    tap_funksjon = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metrikk = DiceMetric(include_background=False, reduction="mean")
    optimaliserer = AdamW(modell.parameters(), lr=læringshastighet, weight_decay=1e-5)
    planlegger = CosineAnnealingLR(optimaliserer, T_max=epoker)

    beste_dice = 0.0
    modell = modell.to(enhet)

    for epoke in range(epoker):
        # --- Trening ---
        modell.train()
        epoketap = 0.0
        for batch in treningsdata:
            bilder = batch["bilde"].to(enhet)
            masker = batch["maske"].to(enhet)

            optimaliserer.zero_grad()
            prediksjoner = modell(bilder)
            tap = tap_funksjon(prediksjoner, masker)
            tap.backward()
            optimaliserer.step()
            epoketap += tap.item()

        planlegger.step()

        # --- Validering ---
        modell.eval()
        with torch.no_grad():
            for batch in valideringsdata:
                bilder = batch["bilde"].to(enhet)
                masker = batch["maske"].to(enhet)
                prediksjoner = torch.argmax(modell(bilder), dim=1, keepdim=True)
                dice_metrikk(y_pred=prediksjoner, y=masker)

        gjennomsnitt_dice = dice_metrikk.aggregate().item()
        dice_metrikk.reset()

        logger.info(
            f"Epoke {epoke+1}/{epoker} — Tap: {epoketap/len(treningsdata):.4f}, "
            f"Dice: {gjennomsnitt_dice:.4f}"
        )

        if gjennomsnitt_dice > beste_dice:
            beste_dice = gjennomsnitt_dice
            lagre_modell(modell)
            logger.info(f"Ny beste Dice: {beste_dice:.4f} — Lagret!")

    logger.info(f"Trening ferdig. Beste Dice: {beste_dice:.4f}")
    return modell


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    modell = bygg_unet()
    antall_parametere = sum(p.numel() for p in modell.parameters())
    logger.info(f"Modellparametere: {antall_parametere:,}")

    # Test forward pass
    enhet = "cuda" if torch.cuda.is_available() else "cpu"
    modell = modell.to(enhet)
    dummy_input = torch.randn(1, ANTALL_KANALER_INN, *BILDE_STØRRELSE).to(enhet)

    with torch.no_grad():
        output = modell(dummy_input)

    logger.info(f"Input form: {dummy_input.shape}")
    logger.info(f"Output form: {output.shape}")
    logger.info("Forward pass OK!")
