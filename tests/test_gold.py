import pandas as pd

from ripplekg.extraction.gold import build_sheet, score

CHUNKS = pd.DataFrame(
    {"chunk_id": ["c1"], "text": ["Intro. We buy cells from Panasonic under a supply deal. More."]}
)


def edges(n, model_ok=True):
    return pd.DataFrame(
        {
            "chunk_id": ["c1"] * n,
            "filer": ["TSLA"] * n,
            "source": ["Panasonic"] * n,
            "relation": ["SUPPLIES"] * n,
            "target": [f"T{i}" for i in range(n)],
            "status": ["active"] * n,
            "strength": [None] * n,
            "quote": ["We buy cells from Panasonic"] * n,
            "family": ["trade"] * n,
            "canonical_relation": ["SUPPLIES"] * n,
            "ok": [model_ok] * n,
            "reason": ["ok"] * n,
        }
    )


def test_sheet_is_blind_and_key_maps_back():
    sheet, key = build_sheet({"claude": edges(5), "ollama": edges(3, False)}, CHUNKS, per_model=4, seed=1)
    assert len(sheet) == 7 and "model" not in sheet.columns
    assert sorted(key["model"].value_counts().to_dict().items()) == [("claude", 4), ("ollama", 3)]
    assert "We buy cells from Panasonic" in sheet["context"][0]


def test_score_precision_overall_and_after_verification():
    _, key = build_sheet({"claude": edges(4), "ollama": edges(4, False)}, CHUNKS, per_model=4, seed=1)
    labels = pd.DataFrame({"row": key["row"], "label": ["y", "N", "Y", "?", "Y", "Y", "N", "Y"]})
    s = score(labels, key).set_index("model")
    assert s.loc["claude", "labelled"] + s.loc["ollama", "labelled"] == 7  # "?" is left out
    assert s.loc["ollama", "verified"] == 0
