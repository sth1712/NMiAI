"""
classifier.py — Klassifisering av medisinske påstander

Bruker RAG-kontekst til å bestemme:
  1. Sant eller usant (binær klassifisering)
  2. Tema-ID (1–115)

Strategi:
  - Rask regelbasert avgjørelse basert på hentede dokumenter (primær)
  - Fallback: LLM-kall hvis regelbasert er usikker
  - Hastighet prioriteres over perfekt nøyaktighet

TODO: Sett opp LLM-tilgang (OpenAI / lokal modell) for fallback-klassifisering.
"""

import logging
import os
import time
from collections import Counter
from typing import Optional

from rag_pipeline import RAGPipeline, INDEKS_STI, METADATA_STI

logger = logging.getLogger(__name__)

# --- Konfigurasjon ---
# TODO: Sett OPENAI_API_KEY som miljøvariabel for LLM-fallback
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# Terskel for sikker regelbasert klassifisering
SIKKER_TERSKEL = 0.75  # Relevans-skår over denne → stol på regelbasert

# Antall kontekstdokumenter å bruke
TOP_K = 5


class KlassifiseringsResultat:
    """Resultat fra klassifisering av én påstand."""
    __slots__ = ["påstand", "sant", "tema_id", "tema_navn", "konfidensgrad",
                 "begrunnelse", "prosesseringstid_ms"]

    def __init__(
        self,
        påstand: str,
        sant: bool,
        tema_id: int,
        tema_navn: str,
        konfidensgrad: float,
        begrunnelse: str = "",
        prosesseringstid_ms: float = 0.0,
    ):
        self.påstand = påstand
        self.sant = sant
        self.tema_id = tema_id
        self.tema_navn = tema_navn
        self.konfidensgrad = konfidensgrad
        self.begrunnelse = begrunnelse
        self.prosesseringstid_ms = prosesseringstid_ms

    def til_dict(self) -> dict:
        return {
            "påstand": self.påstand,
            "sant": self.sant,
            "tema_id": self.tema_id,
            "tema_navn": self.tema_navn,
            "konfidensgrad": round(self.konfidensgrad, 4),
            "begrunnelse": self.begrunnelse,
            "prosesseringstid_ms": round(self.prosesseringstid_ms, 1),
        }


class PåstandsKlassifikator:
    """
    Klassifiserer medisinske påstander som sant/usant og kobler til tema.
    """

    def __init__(self, rag: RAGPipeline):
        self.rag = rag

    def klassifiser(self, påstand: str) -> KlassifiseringsResultat:
        """
        Klassifiser én påstand. Rask og deterministisk.
        """
        start = time.monotonic()

        kontekst = self.rag.hent_kontekst(påstand, top_k=TOP_K)

        if not kontekst:
            logger.warning(f"Ingen kontekst funnet for: {påstand[:60]}")
            resultat = KlassifiseringsResultat(
                påstand=påstand,
                sant=False,
                tema_id=0,
                tema_navn="Ukjent",
                konfidensgrad=0.0,
                begrunnelse="Ingen relevante dokumenter funnet",
            )
        else:
            resultat = self._regelbasert_klassifiser(påstand, kontekst)

        ms = (time.monotonic() - start) * 1000
        resultat.prosesseringstid_ms = ms
        logger.info(f"Klassifisert på {ms:.1f}ms: {påstand[:50]}… → {'SANT' if resultat.sant else 'USANT'} (tema: {resultat.tema_id})")
        return resultat

    def _regelbasert_klassifiser(
        self, påstand: str, kontekst: list[dict]
    ) -> KlassifiseringsResultat:
        """
        Regelbasert klassifisering:
        - Finn det hyppigste tema_id i toppresultater
        - Stemmevoting på sant/usant vektet av relevans-skår
        """
        høy_relevans = [d for d in kontekst if d.get("relevans_skår", 0) > 0.5]
        if not høy_relevans:
            høy_relevans = kontekst[:1]  # Bruk beste uansett

        # Vektet stemming for sant/usant
        sant_skår = sum(d["relevans_skår"] for d in høy_relevans if d.get("sant", False))
        usant_skår = sum(d["relevans_skår"] for d in høy_relevans if not d.get("sant", True))

        total = sant_skår + usant_skår
        if total == 0:
            sant = False
            konfidensgrad = 0.5
        else:
            sant = sant_skår >= usant_skår
            konfidensgrad = max(sant_skår, usant_skår) / total

        # Finn vanligste tema
        tema_teller = Counter(d.get("tema_id", 0) for d in høy_relevans)
        tema_id = tema_teller.most_common(1)[0][0]
        tema_navn = next(
            (d["tema_navn"] for d in høy_relevans if d.get("tema_id") == tema_id),
            "Ukjent"
        )

        begrunnelse = (
            f"Basert på {len(høy_relevans)} relevante dokumenter. "
            f"Sant-skår: {sant_skår:.2f}, Usant-skår: {usant_skår:.2f}."
        )

        return KlassifiseringsResultat(
            påstand=påstand,
            sant=sant,
            tema_id=tema_id,
            tema_navn=tema_navn,
            konfidensgrad=konfidensgrad,
            begrunnelse=begrunnelse,
        )

    def klassifiser_batch(self, påstander: list[str]) -> list[KlassifiseringsResultat]:
        """Klassifiser en liste påstander sekvensielt (FAISS er allerede rask)."""
        resultater = []
        for i, påstand in enumerate(påstander):
            logger.info(f"Behandler {i+1}/{len(påstander)}")
            resultater.append(self.klassifiser(påstand))
        return resultater


def lag_klassifikator() -> PåstandsKlassifikator:
    """Fabrikk-funksjon: lag RAG-pipeline og klassifikator klar til bruk."""
    pipeline = RAGPipeline()

    if INDEKS_STI.exists():
        pipeline.last_indeks()
    else:
        logger.warning(
            "Ingen FAISS-indeks funnet. Kjør rag_pipeline.py for å bygge indeks."
        )
        from rag_pipeline import lag_eksempel_kunnskapsbase
        pipeline.bygg_indeks(lag_eksempel_kunnskapsbase())

    return PåstandsKlassifikator(pipeline)


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    klassifikator = lag_klassifikator()

    testpåstander = [
        "Paracetamol er et trygt smertestillende middel.",
        "Antibiotika dreper virus like effektivt som bakterier.",
        "Røyking øker risikoen for lungekreft betydelig.",
        # TODO: Legg til faktiske testpåstander fra konkurransen
    ]

    print("\n--- Klassifiseringsresultater ---")
    for påstand in testpåstander:
        res = klassifikator.klassifiser(påstand)
        print(json.dumps(res.til_dict(), ensure_ascii=False, indent=2))
        print()
