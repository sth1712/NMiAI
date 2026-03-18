"""
inference.py — Rask inferens for svulstsegmentering i MIP-PET-bilder

Mål: under 10 sekunder per bilde.
Bruker MONAI's sliding window inference for store bilder.

Pipeline:
    1. Last bilde (DICOM / NIfTI / PNG / numpy-array / base64)
    2. Forbehandling (normalisering, resize)
    3. Modellkjøring med valgfri Test-Time Augmentation (TTA)
    4. Etterbehandling (terskeloptimering, morfologiske operasjoner)
    5. Returner binær maske + visualisering

Forbedringer:
    - TTA: horisontale/vertikale flips, gjennomsnitt av prediksjoner
    - Terskeloptimering: prøver flere verdier, velger beste kontur
    - Morfologisk etterbehandling: fjern artefakter (opening), fyll hull (closing)

Krav: pip install monai torch torchvision nibabel pydicom pillow scipy
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import Optional, Union

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
from scipy import ndimage as ndi

from model import bygg_unet, last_modell, BILDE_STØRRELSE, ANTALL_KLASSER

logger = logging.getLogger(__name__)

# --- Konfigurasjon ---
TERSKEL = 0.5          # Standard sannsynlighetsterskel for svulst
ENHET = "cuda" if torch.cuda.is_available() else "cpu"

# TTA-konfigurasjon
TTA_AKTIVERT = True    # Aktiver test-time augmentation som standard

# Morfologisk etterbehandling
MORFOLOGI_ÅPNING_RADIUS = 3    # Fjern små artefakter (støy)
MORFOLOGI_LUKKING_RADIUS = 5   # Fyll hull i segmenteringen
MIN_KOMPONENT_STØRRELSE = 50   # Minimum piksler for gyldig komponent

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


def forbehandle_numpy(bilde: np.ndarray) -> torch.Tensor:
    """
    Forbehandle et numpy-array (f.eks. fra base64-dekoding) til modellinput.

    bilde: numpy-array av form (H, W) eller (H, W, C)
    Returnerer: tensor av form (1, 1, H, W)
    """
    from PIL import Image

    if bilde.ndim == 3 and bilde.shape[2] > 1:
        # Konverter til grånivå
        bilde = np.mean(bilde, axis=2)

    # Resize til modellens inputstørrelse
    img = Image.fromarray(bilde.astype(np.float32), mode='F')
    img = img.resize((BILDE_STØRRELSE[1], BILDE_STØRRELSE[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # Normaliser
    if arr.std() > 0:
        arr = (arr - arr.mean()) / arr.std()

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor.to(ENHET)


def forbehandle_base64(base64_str: str) -> torch.Tensor:
    """
    Dekod base64-kodet bilde til tensor klar for modellen.

    Støtter PNG, JPEG, DICOM (som PNG-konvertert).
    Returnerer: tensor av form (1, 1, H, W)
    """
    from PIL import Image

    bildedata = base64.b64decode(base64_str)
    bilde = Image.open(io.BytesIO(bildedata))

    if bilde.mode != 'L':
        bilde = bilde.convert('L')

    arr = np.array(bilde, dtype=np.float32)
    return forbehandle_numpy(arr)


# --- Test-Time Augmentation (TTA) ---

def _kjør_tta_inferens(
    tensor: torch.Tensor,
    modell: torch.nn.Module,
    bruk_sliding_window: bool = False,
) -> torch.Tensor:
    """
    Kjør inferens med test-time augmentation (TTA).

    Augmenteringer:
        1. Originalbilde
        2. Horisontal flip
        3. Vertikal flip
        4. Horisontal + vertikal flip

    Returnerer gjennomsnittlig sannsynlighetskart (1, 2, H, W).
    """
    modell.eval()
    prediksjoner = []

    def _kjør_inferens(inp: torch.Tensor) -> torch.Tensor:
        if bruk_sliding_window:
            inferer = SlidingWindowInferer(
                roi_size=BILDE_STØRRELSE,
                sw_batch_size=1,
                overlap=0.25,
            )
            return inferer(inp, modell)
        return modell(inp)

    with torch.no_grad():
        # 1. Original
        out = _kjør_inferens(tensor)
        prediksjoner.append(torch.softmax(out, dim=1))

        # 2. Horisontal flip
        flipped_h = torch.flip(tensor, dims=[3])
        out_h = _kjør_inferens(flipped_h)
        prediksjoner.append(torch.flip(torch.softmax(out_h, dim=1), dims=[3]))

        # 3. Vertikal flip
        flipped_v = torch.flip(tensor, dims=[2])
        out_v = _kjør_inferens(flipped_v)
        prediksjoner.append(torch.flip(torch.softmax(out_v, dim=1), dims=[2]))

        # 4. Horisontal + vertikal flip
        flipped_hv = torch.flip(tensor, dims=[2, 3])
        out_hv = _kjør_inferens(flipped_hv)
        prediksjoner.append(torch.flip(torch.softmax(out_hv, dim=1), dims=[2, 3]))

    # Gjennomsnitt av alle prediksjoner
    gjennomsnitt = torch.stack(prediksjoner).mean(dim=0)
    logger.debug(f"TTA: {len(prediksjoner)} augmenteringer, gjennomsnitt beregnet")
    return gjennomsnitt


# --- Terskeloptimering ---

def _optimer_terskel(sannsynlighet: np.ndarray) -> float:
    """
    Finn optimal terskelverdi basert på konturkvalitet.

    Prøver flere terskelverdier og velger den som gir beste balanse
    mellom kompakthet og dekning. Bruker konturfylde (solidity) som mål.

    sannsynlighet: (H, W) numpy-array med sannsynligheter [0, 1]
    Returnerer: optimal terskelverdi
    """
    from scipy.ndimage import label

    terskelverdier = np.arange(0.3, 0.8, 0.05)
    beste_terskel = TERSKEL
    beste_score = -1.0

    for t in terskelverdier:
        maske = (sannsynlighet > t).astype(np.uint8)

        if maske.sum() == 0:
            continue

        # Beregn score basert på konturkvalitet
        labeled, n_komponenter = label(maske)

        if n_komponenter == 0:
            continue

        # Beregn fylde: andel av piksler i bounding box som er del av masken
        total_fylde = 0.0
        total_størrelse = 0
        for komp_id in range(1, n_komponenter + 1):
            komponent = (labeled == komp_id)
            størrelse = komponent.sum()

            if størrelse < MIN_KOMPONENT_STØRRELSE:
                continue

            # Bounding box
            rader = np.any(komponent, axis=1)
            kolonner = np.any(komponent, axis=0)
            r_min, r_max = np.where(rader)[0][[0, -1]]
            k_min, k_max = np.where(kolonner)[0][[0, -1]]
            bb_størrelse = (r_max - r_min + 1) * (k_max - k_min + 1)

            fylde = størrelse / bb_størrelse if bb_størrelse > 0 else 0
            total_fylde += fylde * størrelse
            total_størrelse += størrelse

        if total_størrelse > 0:
            vektet_fylde = total_fylde / total_størrelse
            # Foretrekk masker med rimelig størrelse — straff ekstremt store/små
            størrelse_ratio = total_størrelse / sannsynlighet.size
            størrelse_score = 1.0 - abs(np.log10(størrelse_ratio + 1e-6) + 2)  # Optimal rundt 1%
            score = vektet_fylde * 0.7 + max(0, størrelse_score) * 0.3

            if score > beste_score:
                beste_score = score
                beste_terskel = t

    logger.info(f"Terskeloptimering: valgte {beste_terskel:.2f} (score: {beste_score:.3f})")
    return beste_terskel


# --- Morfologisk etterbehandling ---

def _morfologisk_etterbehandling(maske: np.ndarray) -> np.ndarray:
    """
    Morfologisk etterbehandling av binær segmenteringsmaske.

    Steg:
        1. Opening (erosjon → dilasjon): fjern små artefakter og støy
        2. Closing (dilasjon → erosjon): fyll hull inni segmenteringen
        3. Fjern små tilkoblede komponenter under MIN_KOMPONENT_STØRRELSE
    """
    from scipy.ndimage import binary_opening, binary_closing, label

    if maske.sum() == 0:
        return maske

    # 1. Åpning — fjern små støypiksler
    åpning_struct = ndi.generate_binary_structure(2, 1)
    åpnet = binary_opening(
        maske,
        structure=ndi.iterate_structure(åpning_struct, MORFOLOGI_ÅPNING_RADIUS),
    ).astype(np.uint8)

    # 2. Lukking — fyll hull
    lukking_struct = ndi.generate_binary_structure(2, 1)
    lukket = binary_closing(
        åpnet,
        structure=ndi.iterate_structure(lukking_struct, MORFOLOGI_LUKKING_RADIUS),
    ).astype(np.uint8)

    # 3. Fjern små komponenter
    labeled, n_komponenter = label(lukket)
    renset = np.zeros_like(maske)

    for komp_id in range(1, n_komponenter + 1):
        komponent = (labeled == komp_id)
        if komponent.sum() >= MIN_KOMPONENT_STØRRELSE:
            renset[komponent] = 1

    fjernet = maske.sum() - renset.sum()
    if fjernet > 0:
        logger.debug(
            f"Morfologisk etterbehandling: fjernet {fjernet} piksler "
            f"({n_komponenter} komponenter → {renset.max()} etter filtrering)"
        )

    return renset


def etterbehandle_maske(
    rå_output: torch.Tensor,
    bruk_terskeloptimering: bool = True,
    bruk_morfologi: bool = True,
) -> np.ndarray:
    """
    Konverter modelloutput til binær maske med avansert etterbehandling.

    rå_output: (1, 2, H, W) — sannsynlighetskart (allerede softmax hvis TTA)
    Returnerer: (H, W) numpy-array med 0/1

    Etterbehandlingssteg:
        1. Terskeloptimering (valgfri): finn beste terskelverdi
        2. Binær terskling
        3. Morfologisk etterbehandling (valgfri): opening + closing + komponentfiltrering
    """
    # Sjekk om output allerede er softmax-normalisert (fra TTA)
    if rå_output.shape[1] == ANTALL_KLASSER:
        sannsynlighet = torch.softmax(rå_output, dim=1)
    else:
        sannsynlighet = rå_output

    svulst_sannsynlighet = sannsynlighet[0, 1].cpu().numpy()

    # Terskelvalg
    if bruk_terskeloptimering:
        terskel = _optimer_terskel(svulst_sannsynlighet)
    else:
        terskel = TERSKEL

    maske = (svulst_sannsynlighet > terskel).astype(np.uint8)

    # Morfologisk etterbehandling
    if bruk_morfologi:
        maske = _morfologisk_etterbehandling(maske)

    return maske


def segmenter_bilde(
    bildesti: Union[str, Path],
    modell: torch.nn.Module,
    bruk_sliding_window: bool = False,
    bruk_tta: bool = TTA_AKTIVERT,
    bruk_terskeloptimering: bool = True,
    bruk_morfologi: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Kjør fullstendig segmenteringspipeline på ett bilde.

    Returnerer (maske, prosesseringstid_sek).
    Maske er numpy-array (H, W) med 0 (bakgrunn) eller 1 (svulst).

    Parametere:
        bruk_tta: Aktiver test-time augmentation (4x inferens, bedre nøyaktighet)
        bruk_terskeloptimering: Automatisk terskelvalg basert på konturkvalitet
        bruk_morfologi: Morfologisk etterbehandling (fjern støy, fyll hull)
    """
    start = time.monotonic()

    # Forbehandling
    tensor = forbehandle_bilde(bildesti)

    # Inferens (med eller uten TTA)
    modell.eval()
    if bruk_tta:
        output = _kjør_tta_inferens(tensor, modell, bruk_sliding_window)
    else:
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
    maske = etterbehandle_maske(
        output,
        bruk_terskeloptimering=bruk_terskeloptimering,
        bruk_morfologi=bruk_morfologi,
    )

    prosesseringstid = time.monotonic() - start
    logger.info(
        f"Segmentert {Path(bildesti).name} på {prosesseringstid:.2f}s "
        f"(TTA={'på' if bruk_tta else 'av'}). "
        f"Svulstpiksler: {maske.sum():,} av {maske.size:,}"
    )

    if prosesseringstid > 10.0:
        logger.warning(f"ADVARSEL: Inferens tok {prosesseringstid:.2f}s — over 10-sekundersmålet!")

    return maske, prosesseringstid


def segmenter_tensor(
    tensor: torch.Tensor,
    modell: torch.nn.Module,
    bruk_tta: bool = TTA_AKTIVERT,
    bruk_terskeloptimering: bool = True,
    bruk_morfologi: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Kjør segmentering på en ferdig forbehandlet tensor.
    Brukes av bot.py for direkte tensor-input (fra base64-dekoding).

    tensor: (1, 1, H, W) input-tensor
    Returnerer: (maske, prosesseringstid_sek)
    """
    start = time.monotonic()

    modell.eval()
    if bruk_tta:
        output = _kjør_tta_inferens(tensor, modell)
    else:
        with torch.no_grad():
            output = modell(tensor)

    maske = etterbehandle_maske(
        output,
        bruk_terskeloptimering=bruk_terskeloptimering,
        bruk_morfologi=bruk_morfologi,
    )

    prosesseringstid = time.monotonic() - start
    logger.info(
        f"Segmentert tensor på {prosesseringstid:.2f}s "
        f"(TTA={'på' if bruk_tta else 'av'}). "
        f"Svulstpiksler: {maske.sum():,} av {maske.size:,}"
    )

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


def maske_til_base64_png(maske: np.ndarray) -> str:
    """
    Konverter binær maske til base64-kodet PNG-streng.
    Brukes for å sende resultater tilbake via WebSocket/HTTP.
    """
    from PIL import Image

    img = Image.fromarray((maske * 255).astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def maske_til_rle(maske: np.ndarray) -> list[int]:
    """
    Konverter binær maske til Run-Length Encoding (RLE).

    Format: [start1, lengde1, start2, lengde2, ...]
    der start er 0-indeksert posisjon i flatet array.

    Mer kompakt enn base64 PNG for sparsomme masker.
    """
    flat = maske.flatten()
    rle = []
    i = 0
    while i < len(flat):
        if flat[i] == 1:
            start = i
            while i < len(flat) and flat[i] == 1:
                i += 1
            rle.extend([start, i - start])
        else:
            i += 1
    return rle


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
