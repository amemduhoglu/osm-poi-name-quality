"""Tests for the dense-retrieval comparator.

The matcher is an instrument like the format rule, so what is fixed here is what
it decides rather than how it embeds: a name with a near neighbour in the
gazetteer is not flagged, a name with none is, and an empty neighbourhood is an
abstention rather than a flag.
"""

from __future__ import annotations

import asyncio
import json
import math

import httpx

from poi_audit import retrieval
from poi_audit.references import Place


def place(name: str, lat: float = 37.0, lon: float = 39.0) -> Place:
    """Return a gazetteer place carrying one name."""
    return Place(
        source="geonames",
        place_id="1",
        names=frozenset({name.casefold()}),
        display_name=name,
        lat=lat,
        lon=lon,
        country="TR",
        population=0,
    )


def test_cosine_of_a_vector_with_itself_is_one() -> None:
    """The similarity is the cosine, not a distance."""
    assert math.isclose(retrieval.cosine([0.3, 0.4], [0.3, 0.4]), 1.0, abs_tol=1e-9)


def test_the_best_match_is_the_highest_scoring_candidate() -> None:
    """The score reported is the best a candidate reaches, with its name."""
    vectors = {
        "Harran Kalesi": [1.0, 0.0],
        "Ulu Camii": [0.0, 1.0],
        "Harran Castle": [0.99, 0.14],
    }
    candidates = [place("Ulu Camii"), place("Harran Castle")]
    score, matched = retrieval.best_match("Harran Kalesi", candidates, vectors)
    assert matched == "Harran Castle"
    assert score > 0.9


def test_a_record_below_the_threshold_is_flagged() -> None:
    """Flagging means the name has no candidate near it, which is the class."""
    assert retrieval.decide(0.42, threshold=0.75) is True
    assert retrieval.decide(0.91, threshold=0.75) is False


def test_an_empty_neighbourhood_abstains_rather_than_flags() -> None:
    """A lookup that saw nothing has not cleared the record and has not condemned it."""
    score, matched = retrieval.best_match("Anywhere", [], {"Anywhere": [1.0, 0.0]})
    assert matched is None
    assert math.isnan(score)


# --- the encoder call -------------------------------------------------------

# The next task runs this comparator against the real card in one long pass, so
# the payload shape is checked here rather than discovered after that pass
# fails partway through.


def encoder_with(recorder: list[dict]) -> retrieval.Encoder:
    """Return an encoder whose client answers from a recording transport.

    Args:
        recorder: The list each decoded request body is appended to.

    Returns:
        The encoder.
    """

    def handle(request: httpx.Request) -> httpx.Response:
        recorder.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    encoder = retrieval.Encoder("http://localhost:11434", "bge-m3")
    encoder._client = httpx.AsyncClient(
        base_url=encoder.base_url, transport=httpx.MockTransport(handle)
    )
    return encoder


def test_the_request_carries_the_model_and_the_texts_in_order() -> None:
    """A wrong payload shape is worth catching before the long run, not after."""
    recorder: list[dict] = []

    async def call() -> list[list[float]]:
        async with encoder_with(recorder) as encoder:
            return await encoder.embed(["Harran Kalesi", "Harran Castle"])

    vectors = asyncio.run(call())
    assert recorder[0] == {
        "model": "bge-m3",
        "input": ["Harran Kalesi", "Harran Castle"],
    }
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
