"""
inference.py — Rask inferens for svulstsegmentering i MIP-PET-bilder

Mål: under 10 sekunder per bilde.
Bruker MONAI's sliding window inference for store bilder.

Pipeline:
    1. Last bilde (DICOM / NIfTI / PNG)
    2. Forbehandling (normalisering, resize)
    3. Modellkjøring
    4. Etterbehandling (terskling, morfologiske operasjoner)
    5. Returner binær maske + visualisering

Krav: pip install monai torch torchvision nibabel pydicom pillow
"""

import logging
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    NormalizeIntensity,
    Resize,
    ToTensor,
    AsDiscrete,
)
from monai.data import Dataset, DataLoader

from model import bygg_unet, last_modell, BILDE_STØRRELSE, ANTALL_KLASSER

logger = logging.getLogger(__name__)

# --- Konfigurasjon ---
TERSKEL = 0.5          # Sannsynlighetsterskel for svulst
ENHET = "cuda" if torch.cuda.is_available() else "cpu"

# Forhåndstransformer for inferens
FORBEHANDLING = Compose([
    LoadImage(image_only=True),          # Laster DICOM/NIfTI/PNG
    EnsureChannelFirst(),                 # (H, W) → (1, H, W)
    Resize(spatial_size=BILDE_STØRRELSE), # Skalerer til modellens inputstørrelse
    NormalizeIntensity(nonzero=True),    # Nullpunkts-normalisering
    ToTensor(),
])


def forbehandle_bilde(bildesti: Union[str, Path]) -> torch.Tensor:
    """
    Last og forbehandle ett PET-bilde til tensor klar for modellen.
    Returner tensor av form (1, 1, H, W) (batch + kanal).
    """
    tensor = FORBEHANDLING(str(bildesti))
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # Legg til batch-dimensjon
    return tensor.to(ENHET)


def etterbehandle_maske(rå_output: torch.Tensor) -> np.ndarray:
    """
    Konverter modelloutput til binær maske.

    rå_output: (1, 2, H, W) — sannsynlighetskart
    Returnerer: (H, W) numpy-array med 0/1
    """
    sannsynlighet = torch.softmax(rå_output, dim=1)
    svulst_sannsynlighet = sannsynlighet[0, 1]  # Klasse 1 = svulst
    maske = (svulst_sannsynlighet > TERSKEL).cpu().numpy().astype(np.uint8)
    return maske


def segmenter_bilde(
    bildesti: Union[str, Path],
    modell: torch.nn.Module,
    bruk_sliding_window: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Kjør fullstendig segmenteringspipeline på ett bilde.

    Returnerer (maske, prosesseringstid_sek).
    Maske er numpy-array (H, W) med 0 (bakgrunn) eller 1 (svulst).
    """
    start = time.monotonic()

    # Forbehandling
    tensor = forbehandle_bilde(bildesti)

    # Inferens
    modell.eval()
    with torch.no_grad():
        if bruk_sliding_window:
            inferer = SlidingWindowInferer(
                roi_size=BILDE_STØRRELSE,
                sw_batch_size=1,
                overlap=0.25,
            )
            output = inferer(tensor, modell)
        else:
            output = modell(tensor)

    # Etterbehandling
    maske = etterbehandle_maske(output)

    prosesseringstid = time.monotonic() - start
    logger.info(
        f"Segmentert {Path(bildesti).name} på {prosesseringstid:.2f}s. "
        f"Svulstpiksler: {maske.sum():,} av {maske.size:,}"
    )

    if prosesseringstid > 10.0:
        logger.warning(f"ADVARSEL: Inferens tok {prosesseringstid:.2f}s — over 10-sekundersmålet!")

    return maske, prosesseringstid


def segmenter_batch(
    bildesier: list[Union[str, Path]],
    modell: torch.nn.Module,
) -> list[tuple[np.ndarray, float]]:
    """
    Segmenter en liste bilder sekvensielt.
    Returnerer liste av (maske, tid)-tupler.
    """
    resultater = []
    for i, sti in enumerate(bildesier):
        logger.info(f"Behandler bilde {i+1}/{len(bildesier)}: {sti}")
        maske, tid = segmenter_bilde(sti, modell)
        resultater.append((maske, tid))
    return resultater


def lagre_maske(maske: np.ndarray, utsti: Union[str, Path]):
    """Lagre binær maske som PNG."""
    from PIL import Image
    img = Image.fromarray((maske * 255).astype(np.uint8))
    img.save(str(utsti))
    logger.info(f"Maske lagret: {utsti}")


def lag_pet_datasett(data_mappe: Union[str, Path]) -> Dataset:
    """
    Lag MONAI Dataset fra en mappe med PET-bilder og masker.

    Forventet mappestruktur:
        data_mappe/
            bilder/   — PET-bilder (*.dcm, *.nii, *.png)
            masker/   — Tilsvarende segmenteringsmasker

    TODO: Tilpass til faktisk dataformat fra konkurransen.
    """
    data_mappe = Path(data_mappe)
    bilder = sorted((data_mappe / "bilder").glob("*"))
    masker = sorted((data_mappe / "masker").glob("*"))

    if len(bilder) != len(masker):
        raise ValueError(
            f"Antall bilder ({len(bilder)}) stemmer ikke med masker ({len(masker)})"
        )

    data = [{"bilde": str(b), "maske": str(m)} for b, m in zip(bilder, masker)]

    transforms = Compose([
        LoadImage(image_only=False),
        EnsureChannelFirst(),
        Resize(spatial_size=BILDE_STØRRELSE),
        NormalizeIntensity(nonzero=True),
    ])

    return Dataset(data=data, transform=transforms)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Last modell
    modell = last_modell(enhet=ENHET).to(ENHET)
    logger.info(f"Kjører på: {ENHET}")

    # TODO: Pek til faktisk bildefil
    if len(sys.argv) > 1:
        bildesti = sys.argv[1]
        maske, tid = segmenter_bilde(bildesti, modell)
        logger.info(f"Ferdig! Tid: {tid:.2f}s")

        utsti = Path(bildesti).stem + "_maske.png"
        lagre_maske(maske, utsti)
    else:
        logger.info(
            "Bruk: python inference.py <bildesti>\n"
            "Eksempel: python inference.py pasient_001.dcm"
        )

        # Ytelsestest med syntetisk data
        logger.info("Kjører ytelsestest med syntetisk data...")
        dummy = torch.randn(1, 1, *BILDE_STØRRELSE).to(ENHET)
        modell.eval()
        start = time.monotonic()
        with torch.no_grad():
            _ = modell(dummy)
        tid = time.monotonic() - start
        logger.info(f"Syntetisk inferens: {tid:.3f}s")
        if tid < 10.0:
            logger.info("OK — under 10-sekundersmålet")
        else:
            logger.warning("ADVARSEL — over 10-sekundersmålet!")
