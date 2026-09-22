"""The dense-retrieval comparator, the editor's requested state-of-the-art baseline.

An unsupervised matcher over the open gazetteer this study already fetches: it
embeds a record's name and the display names of the gazetteer places near its
coordinate, scores each candidate by cosine similarity, and reads a calibrated
threshold against the best score reached. It is an instrument, not an oracle.
Task 2.3 scores it the way every model in the pool is scored, against the same
item sets and gates, and nothing here is asserted about its accuracy before
that measurement runs.

The encoder is served by the same local server the model pool talks to, and
runs on the same 16 GB card under constraint C3: no rate or score this
instrument produces carries an offload qualification, and none is measured on
a second device. Its threshold is calibrated on a split fixed by the
configured seed, drawn before the scores it will be judged against exist, so
that the cut is not chosen after the fact from the scores it is then measured
against.
"""

from __future__ import annotations

import math

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from poi_audit.config import get
from poi_audit.references import Place


class Encoder:
    """A text encoder served by the same local server the model pool uses.

    Attributes:
        base_url: Where the server answers.
        model: The encoder's tag in the local registry.
    """

    def __init__(self, base_url: str, model: str) -> None:
        """Open a client against the local server's embedding endpoint.

        Args:
            base_url: The server's address.
            model: The encoder's tag in the local registry.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(float(get("models.serving.timeout_sec"))),
        )

    async def __aenter__(self) -> Encoder:
        """Enter the client's context."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the client."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying client."""
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type(httpx.HTTPError),
        stop=stop_after_attempt(int(get("models.serving.max_retries", 3))),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in one call, retrying transport failures only.

        Args:
            texts: The strings to embed, in the order their vectors return in.

        Returns:
            One vector per text, in the same order the texts were given.

        Raises:
            httpx.HTTPError: If the server keeps failing.
        """
        response = await self._client.post(
            "/api/embed", json={"model": self.model, "input": texts}
        )
        response.raise_for_status()
        return [list(vector) for vector in response.json()["embeddings"]]


def cosine(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two vectors.

    Plain arithmetic over lists rather than a numeric library, so the tests
    that exercise it need no numeric stack.

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        The cosine similarity, or 0.0 when either vector has zero length,
        which a numeric library would raise on instead of deciding for.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def best_match(
    name: str, candidates: list[Place], vectors: dict[str, list[float]]
) -> tuple[float, str | None]:
    """Return the gazetteer candidate closest to a name, by cosine similarity.

    Args:
        name: The record's name, already embedded and present in ``vectors``.
        candidates: The gazetteer places drawn from the record's neighbourhood.
        vectors: Every embedded text, keyed by its own text: the record's name
            and each candidate's display name alike.

    Returns:
        The best cosine score reached and the display name that reached it.
        An empty neighbourhood has nothing to compare the name against, so it
        returns ``(nan, None)``, an abstention rather than a verdict.
    """
    if not candidates:
        return float("nan"), None
    query = vectors[name]
    best_score = float("-inf")
    best_name: str | None = None
    for candidate in candidates:
        score = cosine(query, vectors[candidate.display_name])
        if score > best_score:
            best_score = score
            best_name = candidate.display_name
    return best_score, best_name


def decide(score: float, threshold: float) -> bool:
    """Decide whether a record is flagged by the retrieval comparator.

    Args:
        score: The best candidate's cosine score, or ``nan`` for an
            abstention from an empty neighbourhood.
        threshold: The calibrated cut.

    Returns:
        True when the record is flagged, that is when the best candidate
        falls below the threshold. A ``nan`` score abstains by returning
        False; the caller counts abstentions separately rather than reading
        them off this return value.
    """
    if math.isnan(score):
        return False
    return score < threshold
