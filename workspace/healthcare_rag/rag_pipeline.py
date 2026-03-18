"""
rag_pipeline.py — RAG-pipeline med sentence-transformers og FAISS

Brukes til å klassifisere medisinske påstander som sant/usant
og knytte dem til ett av 115 temaer.

Flyt:
    1. Bygg FAISS-indeks fra kunnskapsbase (én gang)
    2. For hver påstand: embed → hent top-k kontekst → klassifiser

Krav: pip install sentence-transformers faiss-cpu
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# --- Konfigurasjon ---
# TODO: Velg modell — større er mer nøyaktig, mindre er raskere
EMBEDDING_MODELL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Støtter norsk

TOP_K = 5                    # Antall dokumenter å hente
INDEKS_STI = Path("faiss_index.bin")
METADATA_STI = Path("faiss_metadata.pkl")

# TODO: Fyll inn stien til din kunnskapsbase (JSON-fil)
# Format: liste med {"tekst": "...", "tema_id": 1, "tema_navn": "...", "kilde": "..."}
KUNNSKAPSBASE_STI = Path("kunnskapsbase.json")


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for medisinsk faktasjekk.
    """

    def __init__(self, modell_navn: str = EMBEDDING_MODELL):
        logger.info(f"Laster embedding-modell: {modell_navn}")
        self.modell = SentenceTransformer(modell_navn)
        self.indeks: Optional[faiss.Index] = None
        self.metadata: list[dict] = []
        self.dimensjon: int = self.modell.get_sentence_embedding_dimension()

    def bygg_indeks(self, dokumenter: list[dict]):
        """
        Bygg FAISS-indeks fra liste av dokumenter.

        dokumenter: [{"tekst": "...", "tema_id": int, "tema_navn": "...", "sant": bool}, ...]
        """
        logger.info(f"Bygger indeks for {len(dokumenter)} dokumenter...")

        tekster = [d["tekst"] for d in dokumenter]
        embeddings = self.modell.encode(
            tekster,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype(np.float32)

        # Bruk indre produkt (kosinus-likhet siden vi normaliserte)
        self.indeks = faiss.IndexFlatIP(self.dimensjon)
        self.indeks.add(embeddings)
        self.metadata = dokumenter

        logger.info(f"Indeks bygget. Totalt: {self.indeks.ntotal} vektorer")

    def lagre_indeks(self, indeks_sti: Path = INDEKS_STI, meta_sti: Path = METADATA_STI):
        """Lagre FAISS-indeks og metadata til disk."""
        faiss.write_index(self.indeks, str(indeks_sti))
        with open(meta_sti, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Indeks lagret: {indeks_sti}, {meta_sti}")

    def last_indeks(self, indeks_sti: Path = INDEKS_STI, meta_sti: Path = METADATA_STI):
        """Last FAISS-indeks og metadata fra disk."""
        if not indeks_sti.exists() or not meta_sti.exists():
            raise FileNotFoundError(
                f"Indeksfiler ikke funnet: {indeks_sti}, {meta_sti}\n"
                "Kjør bygg_indeks() først."
            )
        self.indeks = faiss.read_index(str(indeks_sti))
        with open(meta_sti, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"Indeks lastet: {self.indeks.ntotal} vektorer")

    def hent_kontekst(self, spørsmål: str, top_k: int = TOP_K) -> list[dict]:
        """
        Hent de top_k mest relevante dokumentene for et spørsmål.
        Returnerer liste med dokumenter sortert etter relevans.
        """
        if self.indeks is None:
            raise RuntimeError("Indeks ikke lastet. Kall last_indeks() eller bygg_indeks() først.")

        embedding = self.modell.encode(
            [spørsmål],
            normalize_embeddings=True,
        ).astype(np.float32)

        skårer, indekser = self.indeks.search(embedding, top_k)

        resultater = []
        for skår, idx in zip(skårer[0], indekser[0]):
            if idx == -1:
                continue
            doc = self.metadata[idx].copy()
            doc["relevans_skår"] = float(skår)
            resultater.append(doc)

        return resultater

    def bygg_fra_fil(self, sti: Path = KUNNSKAPSBASE_STI):
        """Bygg indeks direkte fra JSON-fil."""
        if not sti.exists():
            raise FileNotFoundError(
                f"Kunnskapsbase ikke funnet: {sti}\n"
                "TODO: Opprett kunnskapsbase.json med medisinsk innhold."
            )
        with open(sti, "r", encoding="utf-8") as f:
            dokumenter = json.load(f)
        self.bygg_indeks(dokumenter)


def lag_eksempel_kunnskapsbase() -> list[dict]:
    """
    Lag en liten eksempel-kunnskapsbase for testing.
    TODO: Erstatt med faktisk medisinsk kunnskapsbase (115 temaer).
    """
    return [
        {
            "tekst": "Paracetamol er trygt å ta i normale doser for voksne.",
            "tema_id": 1,
            "tema_navn": "Smertestillende",
            "sant": True,
            "kilde": "Felleskatalogen",
        },
        {
            "tekst": "Antibiotika kurerer virusinfeksjoner effektivt.",
            "tema_id": 2,
            "tema_navn": "Antibiotika",
            "sant": False,
            "kilde": "Folkehelseinstituttet",
        },
        {
            "tekst": "Høyt blodtrykk øker risikoen for hjerteinfarkt.",
            "tema_id": 3,
            "tema_navn": "Hjerte-karsykdommer",
            "sant": True,
            "kilde": "Helsedirektoratet",
        },
        # TODO: Legg til alle 115 temaer her
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = RAGPipeline()

    # Bygg fra eksempeldata hvis ingen indeks finnes
    if not INDEKS_STI.exists():
        logger.info("Ingen eksisterende indeks — bygger fra eksempeldata")
        eksempel = lag_eksempel_kunnskapsbase()
        pipeline.bygg_indeks(eksempel)
        pipeline.lagre_indeks()
    else:
        pipeline.last_indeks()

    # Test et spørsmål
    testspørsmål = "Er det trygt å bruke paracetamol mot hodepine?"
    kontekst = pipeline.hent_kontekst(testspørsmål)

    print(f"\nSpørsmål: {testspørsmål}")
    print(f"Hentet {len(kontekst)} relevante dokumenter:")
    for i, doc in enumerate(kontekst, 1):
        print(f"  {i}. [{doc['tema_navn']}] {doc['tekst']} (skår: {doc['relevans_skår']:.3f})")
