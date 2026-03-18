"""
model.py — U-Net segmenteringsmodell for svulster i MIP-PET-bilder

Bruker MONAI sin innebygde U-Net implementasjon.
Input: 2D MIP-PET-bilde (grånivå eller normalisert float)
Output: Binært segmenteringskart (0 = bakgrunn, 1 = svulst)

Fallback-hierarki for modellvekter:
    1. Egne trente vekter (modell_vekter.pth)
    2. Pretrained modell fra MONAI Model Zoo / HuggingFace
    3. Tilfeldig initialisering (siste utvei)

Krav: pip install monai torch torchvision huggingface_hub
"""

import logging
import os
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

# Katalog for nedlastede pretrained-modeller
PRETRAINED_CACHE = Path("pretrained_cache")


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


def last_pretrained_modell(enhet: str = "cpu") -> Optional[UNet]:
    """
    Forsøk å laste en pretrained segmenteringsmodell som fallback.

    Prøver følgende kilder i rekkefølge:
        1. MONAI Model Zoo (bundles) — medisinsk bildesegmentering
        2. HuggingFace Hub — brukermodeller for medisinsk segmentering

    Returnerer None hvis ingen pretrained modell kan lastes.
    Vekter tilpasses automatisk til vår U-Net-arkitektur (partial loading).
    """
    PRETRAINED_CACHE.mkdir(exist_ok=True)
    cached_weights = PRETRAINED_CACHE / "pretrained_unet.pth"

    # Sjekk om vi har cachet pretrained-vekter fra før
    if cached_weights.exists():
        try:
            modell = bygg_unet()
            tilstand = torch.load(cached_weights, map_location=enhet, weights_only=True)
            modell.load_state_dict(tilstand)
            modell.eval()
            logger.info(f"Pretrained vekter lastet fra cache: {cached_weights}")
            return modell.to(enhet)
        except Exception as e:
            logger.warning(f"Kunne ikke laste cachet pretrained-vekter: {e}")

    # --- Forsøk 1: MONAI Model Zoo bundle ---
    try:
        from monai.bundle import download, load

        bundle_name = "spleen_ct_segmentation"  # Nærmeste tilgjengelige 2D-segmentering
        bundle_dir = PRETRAINED_CACHE / "monai_bundles"
        bundle_dir.mkdir(exist_ok=True)

        logger.info(f"Laster ned MONAI bundle '{bundle_name}'...")
        download(name=bundle_name, bundle_dir=str(bundle_dir))

        # Last nettverket fra bundlen
        bundle_modell = load(
            name=bundle_name,
            bundle_dir=str(bundle_dir),
            source="monaimodel",
            load_ts_module=False,
        )

        # Prøv å overføre kompatible lag (partial loading)
        vår_modell = bygg_unet()
        vår_tilstand = vår_modell.state_dict()
        pretrained_tilstand = bundle_modell.state_dict() if hasattr(bundle_modell, 'state_dict') else {}

        overført = 0
        for nøkkel in vår_tilstand:
            if nøkkel in pretrained_tilstand and vår_tilstand[nøkkel].shape == pretrained_tilstand[nøkkel].shape:
                vår_tilstand[nøkkel] = pretrained_tilstand[nøkkel]
                overført += 1

        if overført > 0:
            vår_modell.load_state_dict(vår_tilstand)
            vår_modell.eval()
            # Cache for raskere lasting neste gang
            torch.save(vår_modell.state_dict(), cached_weights)
            logger.info(
                f"MONAI bundle lastet! Overførte {overført}/{len(vår_tilstand)} "
                f"lagvekter fra '{bundle_name}'"
            )
            return vår_modell.to(enhet)
        else:
            logger.warning("Ingen kompatible lagvekter funnet i MONAI bundle.")

    except ImportError:
        logger.warning("MONAI bundle-modulen er ikke tilgjengelig.")
    except Exception as e:
        logger.warning(f"Kunne ikke laste MONAI bundle: {e}")

    # --- Forsøk 2: HuggingFace Hub ---
    try:
        from huggingface_hub import hf_hub_download

        # Søk etter medisinsk segmenteringsmodell på HuggingFace
        repo_id = "katielink/monai_unet_pet_segmentation"
        filename = "model.pt"

        logger.info(f"Laster ned fra HuggingFace: {repo_id}...")
        nedlastet_sti = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(PRETRAINED_CACHE / "hf_cache"),
        )

        vår_modell = bygg_unet()
        vår_tilstand = vår_modell.state_dict()
        pretrained_tilstand = torch.load(nedlastet_sti, map_location=enhet, weights_only=True)

        # Håndter ulike formater (state_dict direkte eller innpakket)
        if isinstance(pretrained_tilstand, dict) and "state_dict" in pretrained_tilstand:
            pretrained_tilstand = pretrained_tilstand["state_dict"]
        elif isinstance(pretrained_tilstand, dict) and "model_state_dict" in pretrained_tilstand:
            pretrained_tilstand = pretrained_tilstand["model_state_dict"]

        overført = 0
        for nøkkel in vår_tilstand:
            if nøkkel in pretrained_tilstand and vår_tilstand[nøkkel].shape == pretrained_tilstand[nøkkel].shape:
                vår_tilstand[nøkkel] = pretrained_tilstand[nøkkel]
                overført += 1

        if overført > 0:
            vår_modell.load_state_dict(vår_tilstand)
            vår_modell.eval()
            torch.save(vår_modell.state_dict(), cached_weights)
            logger.info(
                f"HuggingFace-modell lastet! Overførte {overført}/{len(vår_tilstand)} "
                f"lagvekter fra '{repo_id}'"
            )
            return vår_modell.to(enhet)
        else:
            logger.warning("Ingen kompatible lagvekter funnet i HuggingFace-modellen.")

    except ImportError:
        logger.warning("huggingface_hub er ikke installert. Installer med: pip install huggingface_hub")
    except Exception as e:
        logger.warning(f"Kunne ikke laste fra HuggingFace: {e}")

    logger.warning("Ingen pretrained-modeller tilgjengelig.")
    return None


def last_modell(sti: Path = MODELL_STI, enhet: Optional[str] = None) -> UNet:
    """
    Last modellvekter med fallback-hierarki:
        1. Egne trente vekter (fra sti)
        2. Pretrained modell (MONAI Zoo / HuggingFace)
        3. Tilfeldig initialisering (siste utvei)

    Logger tydelig hvilken kilde som brukes.
    """
    if enhet is None:
        enhet = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Nivå 1: Egne trente vekter ---
    if sti.exists():
        try:
            modell = bygg_unet()
            tilstand = torch.load(sti, map_location=enhet, weights_only=True)
            modell.load_state_dict(tilstand)
            modell.eval()
            logger.info(f"[MODELLKILDE: EGNE VEKTER] Lastet fra {sti} på enhet: {enhet}")
            return modell.to(enhet)
        except Exception as e:
            logger.warning(f"Feil ved lasting av egne vekter fra {sti}: {e}")

    logger.info(f"Ingen egne vekter funnet på {sti}. Forsøker pretrained fallback...")

    # --- Nivå 2: Pretrained modell ---
    pretrained = last_pretrained_modell(enhet=enhet)
    if pretrained is not None:
        logger.info("[MODELLKILDE: PRETRAINED] Bruker pretrained modell som fallback.")
        return pretrained

    # --- Nivå 3: Tilfeldig initialisering ---
    modell = bygg_unet()
    logger.warning(
        "[MODELLKILDE: TILFELDIG INITIALISERING] Ingen vekter tilgjengelig!\n"
        "Modellen er IKKE klar for produksjon. Tren modellen eller skaff pretrained-vekter."
    )
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
