"""
classifier.py — Klassifisering av medisinske påstander

Bruker RAG-kontekst til å bestemme:
  1. Sant eller usant (binær klassifisering)
  2. Tema-ID (1–115)

Strategier (i prioritert rekkefølge):
  1. NLI-basert klassifisering (cross-encoder / DeBERTa) — sterkest for sant/usant
  2. LLM-basert fallback (OpenAI / Anthropic) — brukes når regelbasert er usikker
  3. Regelbasert (embedding-likhet + stemming) — raskest, alltid tilgjengelig

Server-tilkobling:
  - WebSocket eller HTTP POST mot konkurranse-server
  - Autentisering via JWT_TOKEN fra miljøvariabel
"""

import abc
import asyncio
import json
import logging
import os
import time
from collections import Counter
from typing import Optional

from rag_pipeline import RAGPipeline, INDEKS_STI, METADATA_STI

logger = logging.getLogger(__name__)

# --- Konfigurasjon ---
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
JWT_TOKEN: str = os.environ.get("JWT_TOKEN", "")

# Terskel for sikker regelbasert klassifisering
SIKKER_TERSKEL = 0.75  # Relevans-skår over denne → stol på regelbasert

# Antall kontekstdokumenter å bruke
TOP_K = 5

# NLI-modell (cross-encoder for natural language inference)
NLI_MODELL = "cross-encoder/nli-deberta-v3-base"

# Server-konfigurasjon
SERVER_URL: str = os.environ.get("SERVER_URL", "ws://localhost:8080/ws")
SERVER_HTTP_URL: str = os.environ.get("SERVER_HTTP_URL", "http://localhost:8080/classify")


# ---------------------------------------------------------------------------
# Resultatklasse
# ---------------------------------------------------------------------------

class KlassifiseringsResultat:
    """Resultat fra klassifisering av én påstand."""
    __slots__ = ["påstand", "sant", "tema_id", "tema_navn", "konfidensgrad",
                 "begrunnelse", "prosesseringstid_ms", "strategi"]

    def __init__(
        self,
        påstand: str,
        sant: bool,
        tema_id: int,
        tema_navn: str,
        konfidensgrad: float,
        begrunnelse: str = "",
        prosesseringstid_ms: float = 0.0,
        strategi: str = "regelbasert",
    ):
        self.påstand = påstand
        self.sant = sant
        self.tema_id = tema_id
        self.tema_navn = tema_navn
        self.konfidensgrad = konfidensgrad
        self.begrunnelse = begrunnelse
        self.prosesseringstid_ms = prosesseringstid_ms
        self.strategi = strategi

    def til_dict(self) -> dict:
        return {
            "påstand": self.påstand,
            "sant": self.sant,
            "tema_id": self.tema_id,
            "tema_navn": self.tema_navn,
            "konfidensgrad": round(self.konfidensgrad, 4),
            "begrunnelse": self.begrunnelse,
            "prosesseringstid_ms": round(self.prosesseringstid_ms, 1),
            "strategi": self.strategi,
        }


# ---------------------------------------------------------------------------
# Abstrakt basisklasse for klassifiseringsstrategier
# ---------------------------------------------------------------------------

class KlassifiseringsStrategi(abc.ABC):
    """Felles grensesnitt for alle klassifiseringsstrategier."""

    @abc.abstractmethod
    def klassifiser(
        self, påstand: str, kontekst: list[dict]
    ) -> KlassifiseringsResultat:
        """Klassifiser én påstand med gitt RAG-kontekst."""
        ...

    @property
    @abc.abstractmethod
    def navn(self) -> str:
        """Navnet på strategien (for logging/sporing)."""
        ...


# ---------------------------------------------------------------------------
# 1. Regelbasert strategi (alltid tilgjengelig)
# ---------------------------------------------------------------------------

class RegelbasertStrategi(KlassifiseringsStrategi):
    """
    Regelbasert klassifisering:
    - Vektet stemming på sant/usant basert på relevans-skår
    - Finn hyppigste tema_id i toppresultater
    """

    @property
    def navn(self) -> str:
        return "regelbasert"

    def klassifiser(
        self, påstand: str, kontekst: list[dict]
    ) -> KlassifiseringsResultat:
        høy_relevans = [d for d in kontekst if d.get("relevans_skår", 0) > 0.5]
        if not høy_relevans:
            høy_relevans = kontekst[:1]  # Bruk beste uansett

        # Vektet stemming for sant/usant
        sant_skår = sum(
            d["relevans_skår"] for d in høy_relevans if d.get("sant", False)
        )
        usant_skår = sum(
            d["relevans_skår"] for d in høy_relevans if not d.get("sant", True)
        )

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
            "Ukjent",
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
            strategi=self.navn,
        )


# ---------------------------------------------------------------------------
# 2. NLI-basert strategi (cross-encoder / DeBERTa)
# ---------------------------------------------------------------------------

class NLIStrategi(KlassifiseringsStrategi):
    """
    Natural Language Inference-strategi.

    Bruker en cross-encoder (f.eks. DeBERTa) trent på NLI til å vurdere
    om en påstand støttes (entailment), motsis (contradiction) eller er
    nøytral i forhold til hvert hentet dokument.

    Sterkere enn ren embedding-likhet for sant/usant-klassifisering fordi
    modellen eksplisitt trenes på logiske relasjoner mellom setningspar.
    """

    def __init__(self, modell_navn: str = NLI_MODELL):
        from sentence_transformers import CrossEncoder

        logger.info(f"Laster NLI cross-encoder: {modell_navn}")
        self.modell = CrossEncoder(modell_navn)
        # Etikettrekkefølge for nli-deberta-v3-base: contradiction, entailment, neutral
        self._etikett_indeks = {"contradiction": 0, "entailment": 1, "neutral": 2}

    @property
    def navn(self) -> str:
        return "nli"

    def klassifiser(
        self, påstand: str, kontekst: list[dict]
    ) -> KlassifiseringsResultat:
        if not kontekst:
            return KlassifiseringsResultat(
                påstand=påstand,
                sant=False,
                tema_id=0,
                tema_navn="Ukjent",
                konfidensgrad=0.0,
                begrunnelse="Ingen kontekst for NLI",
                strategi=self.navn,
            )

        # Bygg setningspar: (premiss=dokument, hypotese=påstand)
        setningspar = [(d["tekst"], påstand) for d in kontekst]
        skårer = self.modell.predict(setningspar)  # shape: (n, 3)

        # Aggreger NLI-resultatene vektet med relevans-skår
        total_entailment = 0.0
        total_contradiction = 0.0
        total_neutral = 0.0

        for i, dok in enumerate(kontekst):
            relevans = dok.get("relevans_skår", 1.0)
            logits = skårer[i]
            # Softmax for å få sannsynligheter
            exp_logits = _softmax(logits)

            total_contradiction += exp_logits[self._etikett_indeks["contradiction"]] * relevans
            total_entailment += exp_logits[self._etikett_indeks["entailment"]] * relevans
            total_neutral += exp_logits[self._etikett_indeks["neutral"]] * relevans

        total = total_entailment + total_contradiction + total_neutral
        if total == 0:
            sant = False
            konfidensgrad = 0.5
        else:
            sant = total_entailment > total_contradiction
            konfidensgrad = max(total_entailment, total_contradiction) / total

        # Finn tema fra kontekst
        tema_teller = Counter(d.get("tema_id", 0) for d in kontekst)
        tema_id = tema_teller.most_common(1)[0][0]
        tema_navn = next(
            (d["tema_navn"] for d in kontekst if d.get("tema_id") == tema_id),
            "Ukjent",
        )

        begrunnelse = (
            f"NLI-analyse over {len(kontekst)} dokumenter. "
            f"Entailment: {total_entailment:.2f}, "
            f"Contradiction: {total_contradiction:.2f}, "
            f"Neutral: {total_neutral:.2f}."
        )

        return KlassifiseringsResultat(
            påstand=påstand,
            sant=sant,
            tema_id=tema_id,
            tema_navn=tema_navn,
            konfidensgrad=konfidensgrad,
            begrunnelse=begrunnelse,
            strategi=self.navn,
        )


def _softmax(logits) -> list[float]:
    """Beregn softmax over en vektor av logits."""
    import numpy as np

    exp_vals = np.exp(logits - np.max(logits))
    return (exp_vals / exp_vals.sum()).tolist()


# ---------------------------------------------------------------------------
# 3. LLM-basert fallback (OpenAI / Anthropic)
# ---------------------------------------------------------------------------

class LLMFallbackStrategi(KlassifiseringsStrategi):
    """
    LLM-basert klassifisering via OpenAI eller Anthropic API.

    Brukes som fallback når regelbasert/NLI har lav konfidensgrad.
    Sender påstanden + RAG-kontekst til LLM og ber om strukturert JSON-svar.
    """

    def __init__(self):
        self._klient = None
        self._leverandør: Optional[str] = None
        self._opprett_klient()

    def _opprett_klient(self):
        """Opprett API-klient basert på tilgjengelige nøkler."""
        if OPENAI_API_KEY:
            try:
                import openai

                self._klient = openai.OpenAI(api_key=OPENAI_API_KEY)
                self._leverandør = "openai"
                logger.info("LLM-fallback: Bruker OpenAI API")
                return
            except ImportError:
                logger.warning("openai-pakke ikke installert, prøver Anthropic")

        if ANTHROPIC_API_KEY:
            try:
                import anthropic

                self._klient = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                self._leverandør = "anthropic"
                logger.info("LLM-fallback: Bruker Anthropic API")
                return
            except ImportError:
                logger.warning("anthropic-pakke ikke installert")

        raise RuntimeError(
            "Ingen LLM API tilgjengelig. Sett OPENAI_API_KEY eller "
            "ANTHROPIC_API_KEY som miljøvariabel, og installer tilhørende pakke."
        )

    @property
    def navn(self) -> str:
        return f"llm-{self._leverandør}"

    def _bygg_prompt(self, påstand: str, kontekst: list[dict]) -> str:
        """Bygg klassifiseringsprompt med RAG-kontekst."""
        kontekst_tekst = "\n".join(
            f"  {i+1}. [Tema: {d.get('tema_navn', 'Ukjent')} (ID: {d.get('tema_id', 0)})] "
            f"(Relevans: {d.get('relevans_skår', 0):.2f}) — {d['tekst']}"
            for i, d in enumerate(kontekst)
        )

        return f"""Du er en medisinsk faktasjekker. Vurder om følgende påstand er SANN eller USANN
basert på den medisinske konteksten nedenfor.

PÅSTAND:
  "{påstand}"

MEDISINSK KONTEKST (hentet fra verifiserte kilder):
{kontekst_tekst}

Svar BARE med gyldig JSON i dette formatet — ingen annen tekst:
{{
  "sant": true/false,
  "tema_id": <heltall mellom 1 og 115>,
  "begrunnelse": "<kort begrunnelse på norsk, maks 2 setninger>"
}}

Viktig:
- Baser svaret på konteksten over, ikke generell kunnskap.
- Hvis konteksten er tvetydig, velg det mest sannsynlige svaret.
- tema_id skal matche det mest relevante temaet fra konteksten."""

    def _kall_openai(self, prompt: str) -> dict:
        """Kall OpenAI API og parse JSON-svar."""
        respons = self._klient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Du er en medisinsk faktasjekker. Svar alltid med gyldig JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        return json.loads(respons.choices[0].message.content)

    def _kall_anthropic(self, prompt: str) -> dict:
        """Kall Anthropic API og parse JSON-svar."""
        respons = self._klient.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
            system="Du er en medisinsk faktasjekker. Svar alltid med gyldig JSON og ingenting annet.",
        )
        tekst = respons.content[0].text.strip()
        # Fjern eventuell markdown-innpakning
        if tekst.startswith("```"):
            tekst = tekst.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(tekst)

    def klassifiser(
        self, påstand: str, kontekst: list[dict]
    ) -> KlassifiseringsResultat:
        prompt = self._bygg_prompt(påstand, kontekst)

        try:
            if self._leverandør == "openai":
                svar = self._kall_openai(prompt)
            elif self._leverandør == "anthropic":
                svar = self._kall_anthropic(prompt)
            else:
                raise RuntimeError(f"Ukjent leverandør: {self._leverandør}")
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Feil ved parsing av LLM-svar: {e}")
            return KlassifiseringsResultat(
                påstand=påstand,
                sant=False,
                tema_id=0,
                tema_navn="Ukjent",
                konfidensgrad=0.0,
                begrunnelse=f"LLM-parsing feilet: {e}",
                strategi=self.navn,
            )
        except Exception as e:
            logger.error(f"LLM API-kall feilet: {e}")
            return KlassifiseringsResultat(
                påstand=påstand,
                sant=False,
                tema_id=0,
                tema_navn="Ukjent",
                konfidensgrad=0.0,
                begrunnelse=f"LLM-feil: {e}",
                strategi=self.navn,
            )

        tema_id = int(svar.get("tema_id", 0))
        tema_navn = next(
            (d["tema_navn"] for d in kontekst if d.get("tema_id") == tema_id),
            "Ukjent",
        )

        return KlassifiseringsResultat(
            påstand=påstand,
            sant=bool(svar.get("sant", False)),
            tema_id=tema_id,
            tema_navn=tema_navn,
            konfidensgrad=0.85,  # LLM-basert → høy men ikke perfekt konfidensgrad
            begrunnelse=svar.get("begrunnelse", ""),
            strategi=self.navn,
        )


# ---------------------------------------------------------------------------
# Hovedklassifikator med strategi-kaskade
# ---------------------------------------------------------------------------

class PåstandsKlassifikator:
    """
    Klassifiserer medisinske påstander som sant/usant og kobler til tema.

    Bruker en kaskade av strategier:
      1. Primærstrategi (NLI eller regelbasert)
      2. LLM-fallback hvis konfidensgraden er under SIKKER_TERSKEL
    """

    def __init__(
        self,
        rag: RAGPipeline,
        primær_strategi: KlassifiseringsStrategi,
        fallback_strategi: Optional[KlassifiseringsStrategi] = None,
    ):
        self.rag = rag
        self.primær = primær_strategi
        self.fallback = fallback_strategi
        logger.info(
            f"Klassifikator initialisert. "
            f"Primær: {self.primær.navn}, "
            f"Fallback: {self.fallback.navn if self.fallback else 'ingen'}"
        )

    def klassifiser(self, påstand: str) -> KlassifiseringsResultat:
        """
        Klassifiser én påstand med automatisk fallback.
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
            # Prøv primærstrategi
            resultat = self.primær.klassifiser(påstand, kontekst)

            # Fallback hvis lav konfidensgrad
            if resultat.konfidensgrad < SIKKER_TERSKEL and self.fallback:
                logger.info(
                    f"Lav konfidensgrad ({resultat.konfidensgrad:.2f}) — "
                    f"bruker fallback ({self.fallback.navn})"
                )
                resultat = self.fallback.klassifiser(påstand, kontekst)

        ms = (time.monotonic() - start) * 1000
        resultat.prosesseringstid_ms = ms
        logger.info(
            f"Klassifisert på {ms:.1f}ms [{resultat.strategi}]: "
            f"{påstand[:50]}… → {'SANT' if resultat.sant else 'USANT'} "
            f"(tema: {resultat.tema_id}, konf: {resultat.konfidensgrad:.2f})"
        )
        return resultat

    def klassifiser_batch(
        self, påstander: list[str]
    ) -> list[KlassifiseringsResultat]:
        """Klassifiser en liste påstander sekvensielt."""
        resultater = []
        for i, påstand in enumerate(påstander):
            logger.info(f"Behandler {i+1}/{len(påstander)}")
            resultater.append(self.klassifiser(påstand))
        return resultater


# ---------------------------------------------------------------------------
# Server-tilkobling (WebSocket + HTTP POST)
# ---------------------------------------------------------------------------

class ServerTilkobling:
    """
    Kobler til konkurranse-server for å motta påstander og sende
    klassifiseringer tilbake.

    Støtter:
      - WebSocket (primær, for sanntidsstrømming)
      - HTTP POST (fallback, for enkeltforespørsler)

    Autentisering via JWT_TOKEN fra miljøvariabel.
    """

    def __init__(self, klassifikator: PåstandsKlassifikator):
        self.klassifikator = klassifikator
        self._token = JWT_TOKEN
        if not self._token:
            logger.warning(
                "JWT_TOKEN ikke satt. Server-tilkobling vil feile ved autentisering."
            )

    # --- WebSocket-modus ---

    async def koble_til_websocket(self, url: str = SERVER_URL):
        """
        Koble til server via WebSocket.
        Mottar påstander som JSON, sender klassifiseringer tilbake.

        Forventet innkommende format:
            {"type": "påstand", "id": "abc123", "tekst": "Paracetamol er trygt."}
        Utgående format:
            {"type": "svar", "id": "abc123", "sant": true, "tema_id": 1, ...}
        """
        import websockets

        headers = {"Authorization": f"Bearer {self._token}"}

        logger.info(f"Kobler til WebSocket: {url}")
        async for ws in websockets.connect(url, additional_headers=headers):
            try:
                logger.info("WebSocket tilkoblet. Venter på påstander…")
                async for melding in ws:
                    await self._behandle_ws_melding(ws, melding)
            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket frakoblet: {e}. Prøver igjen om 3 sek…")
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"WebSocket-feil: {e}")
                await asyncio.sleep(5)

    async def _behandle_ws_melding(self, ws, melding: str):
        """Parse innkommende melding, klassifiser, og send svar."""
        try:
            data = json.loads(melding)
        except json.JSONDecodeError:
            logger.warning(f"Ugyldig JSON fra server: {melding[:100]}")
            return

        if data.get("type") != "påstand":
            logger.debug(f"Ignorerer meldingstype: {data.get('type')}")
            return

        påstand_id = data.get("id", "ukjent")
        tekst = data.get("tekst", "")

        if not tekst:
            logger.warning(f"Tom påstand mottatt (id: {påstand_id})")
            return

        resultat = self.klassifikator.klassifiser(tekst)

        svar = {
            "type": "svar",
            "id": påstand_id,
            "sant": resultat.sant,
            "tema_id": resultat.tema_id,
            "tema_navn": resultat.tema_navn,
            "konfidensgrad": round(resultat.konfidensgrad, 4),
            "begrunnelse": resultat.begrunnelse,
            "prosesseringstid_ms": round(resultat.prosesseringstid_ms, 1),
            "strategi": resultat.strategi,
        }
        await ws.send(json.dumps(svar, ensure_ascii=False))
        logger.info(f"Svar sendt for påstand {påstand_id}")

    # --- HTTP POST-modus ---

    def send_http_klassifisering(
        self,
        påstand: str,
        påstand_id: str = "",
        url: str = SERVER_HTTP_URL,
    ) -> dict:
        """
        Klassifiser og send resultat via HTTP POST.
        Returnerer server-responsen som dict.
        """
        import requests

        resultat = self.klassifikator.klassifiser(påstand)

        payload = {
            "id": påstand_id,
            "sant": resultat.sant,
            "tema_id": resultat.tema_id,
            "tema_navn": resultat.tema_navn,
            "konfidensgrad": round(resultat.konfidensgrad, 4),
            "begrunnelse": resultat.begrunnelse,
            "prosesseringstid_ms": round(resultat.prosesseringstid_ms, 1),
            "strategi": resultat.strategi,
        }

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        respons = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=10,
        )
        respons.raise_for_status()

        logger.info(f"HTTP-svar sendt for påstand {påstand_id}: {respons.status_code}")
        return respons.json()


# ---------------------------------------------------------------------------
# Factory-funksjon: velger beste tilgjengelige strategi
# ---------------------------------------------------------------------------

def _kan_bruke_nli() -> bool:
    """Sjekk om NLI-modell er tilgjengelig."""
    try:
        from sentence_transformers import CrossEncoder
        return True
    except ImportError:
        return False


def _kan_bruke_llm() -> bool:
    """Sjekk om LLM API er tilgjengelig og konfigurert."""
    if OPENAI_API_KEY:
        try:
            import openai
            return True
        except ImportError:
            pass
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            return True
        except ImportError:
            pass
    return False


def velg_beste_strategi() -> tuple[KlassifiseringsStrategi, Optional[KlassifiseringsStrategi]]:
    """
    Velg beste tilgjengelige strategi basert på hva som er installert
    og konfigurert.

    Returnerer (primærstrategi, fallback_strategi).

    Prioritering:
      1. NLI (primær) + LLM (fallback) — sterkeste kombinasjon
      2. NLI (primær) + ingen fallback
      3. Regelbasert (primær) + LLM (fallback)
      4. Regelbasert (primær) + ingen fallback — alltid tilgjengelig
    """
    har_nli = _kan_bruke_nli()
    har_llm = _kan_bruke_llm()

    logger.info(f"Tilgjengelige strategier — NLI: {har_nli}, LLM: {har_llm}")

    # Velg fallback
    fallback: Optional[KlassifiseringsStrategi] = None
    if har_llm:
        try:
            fallback = LLMFallbackStrategi()
        except RuntimeError as e:
            logger.warning(f"Kunne ikke opprette LLM-fallback: {e}")

    # Velg primærstrategi
    if har_nli:
        try:
            primær = NLIStrategi()
            logger.info("Primærstrategi: NLI (cross-encoder)")
            return primær, fallback
        except Exception as e:
            logger.warning(f"Kunne ikke laste NLI-modell: {e}")

    # Regelbasert som siste utvei (alltid tilgjengelig)
    primær = RegelbasertStrategi()
    logger.info("Primærstrategi: Regelbasert (embedding-likhet)")
    return primær, fallback


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

    primær, fallback = velg_beste_strategi()
    return PåstandsKlassifikator(pipeline, primær, fallback)


def lag_server_tilkobling(
    klassifikator: Optional[PåstandsKlassifikator] = None,
) -> ServerTilkobling:
    """Lag en server-tilkobling klar til bruk."""
    if klassifikator is None:
        klassifikator = lag_klassifikator()
    return ServerTilkobling(klassifikator)


# ---------------------------------------------------------------------------
# Hovedprogram
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    klassifikator = lag_klassifikator()

    # --- Modus: Server (WebSocket) ---
    if "--server" in sys.argv:
        tilkobling = lag_server_tilkobling(klassifikator)
        url = SERVER_URL
        # Tillat override av URL via kommandolinje
        for arg in sys.argv:
            if arg.startswith("--url="):
                url = arg.split("=", 1)[1]

        logger.info(f"Starter server-tilkobling mot {url}")
        asyncio.run(tilkobling.koble_til_websocket(url))

    # --- Modus: Lokal test ---
    else:
        testpåstander = [
            "Paracetamol er et trygt smertestillende middel.",
            "Antibiotika dreper virus like effektivt som bakterier.",
            "Røyking øker risikoen for lungekreft betydelig.",
        ]

        print("\n--- Klassifiseringsresultater ---")
        for påstand in testpåstander:
            res = klassifikator.klassifiser(påstand)
            print(json.dumps(res.til_dict(), ensure_ascii=False, indent=2))
            print()
