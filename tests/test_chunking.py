from ripplekg.ingestion.chunking import TARGET_CHARS, chunk_documents, line_hash
from ripplekg.ingestion.documents import DocSection, Document
from ripplekg.ingestion.prefilter import Prefilter
from ripplekg.ontology.schema import load_ontology

LONG = "We purchase battery cells from Panasonic under a supply agreement that runs through 2026."
RISK = "Our business depends on the availability of lithium, nickel and cobalt at reasonable prices."


def doc(doc_id, published, *lines, company="TSLA", section="business"):
    return Document(
        doc_id=doc_id,
        source_type="edgar",
        company_id=company,
        published_at=published,
        url="https://example.com",
        title=doc_id,
        sections=(DocSection(section, "\n".join(lines)),),
        source_sha256="x",
        metadata={"form": "10-K"},
    )


def test_repeated_lines_are_sent_once_and_recorded_as_occurrences():
    d1 = doc("edgar:1", "2023-01-30T21:00:00+00:00", "Item 1. Business", LONG, RISK)
    d2 = doc(
        "edgar:2",
        "2024-01-29T21:00:00+00:00",
        "Item 1. Business",
        LONG,
        "Customer A | 23%",
        RISK + " New text.",
    )
    chunks, occ, stats = chunk_documents([d2, d1])  # order of input does not matter
    texts = [c.text for c in chunks]
    assert LONG in texts[0] and LONG not in texts[1]
    assert "Customer A | 23%" in texts[1]
    assert [o["doc_id"] for o in occ] == ["edgar:2"]
    assert occ[0]["line_hash"] == line_hash(LONG)
    assert stats.repeated_lines == 1


def test_dedup_is_per_company():
    d1 = doc("edgar:1", "2023-01-30T00:00:00+00:00", LONG, company="TSLA")
    d2 = doc("edgar:2", "2023-02-01T00:00:00+00:00", LONG, company="GM")
    chunks, _, _ = chunk_documents([d1, d2])
    assert len(chunks) == 2


def test_chunks_respect_target_size_and_carry_context():
    lines = [f"Sentence {i} about our supply chain and the suppliers we rely on." * 4 for i in range(60)]
    chunks, _, _ = chunk_documents([doc("edgar:1", "2023-01-30T00:00:00+00:00", *lines)])
    assert len(chunks) > 1
    assert all(c.n_chars <= TARGET_CHARS + 400 for c in chunks)
    last_line_of_first = chunks[0].text.split("\n")[-1]
    assert chunks[1].text.startswith(last_line_of_first)


def test_very_long_line_spilling_into_next_chunk_is_kept():
    long_line = "Our suppliers ship cathode material to our plants every week. " * 120
    chunks, _, _ = chunk_documents([doc("edgar:1", "2023-01-30T00:00:00+00:00", long_line)])
    assert len(chunks) >= 2
    assert sum(c.n_chars for c in chunks) >= len(long_line.strip()) - 10


def test_chunk_ids_are_deterministic():
    d = doc("edgar:1", "2023-01-30T00:00:00+00:00", LONG, RISK)
    assert [c.chunk_id for c in chunk_documents([d])[0]] == [c.chunk_id for c in chunk_documents([d])[0]]


PF = Prefilter.from_ontology(load_ontology())


def test_prefilter_tags_families_with_endpoints():
    assert PF.families(LONG, "TSLA") == ["trade"]
    assert PF.families("Sales to our largest customer accounted for 41% of revenue.", "NVX") == ["trade"]
    assert PF.families("We compete with Albemarle and Ganfeng.", "SQM") == ["competition"]
    assert "exposure" in PF.families("China's new export controls on graphite may delay shipments.", "TSLA")
    assert PF.families("We formed a joint venture with Halcyon Materials Inc.", "TSLA") == [
        "ownership_capital"
    ]


def test_prefilter_rejects_cue_words_without_an_endpoint():
    assert PF.families("Our customers value reliability and our suppliers are important.", "TSLA") == []
    assert PF.families("We compete on price and quality.", "TSLA") == []
    assert PF.families("Revenue increased 12% compared with the prior year.", "TSLA") == []


def test_filer_does_not_count_as_another_company():
    assert PF.other_companies("Tesla's suppliers include Panasonic.", "TSLA") == ["PANASONIC"]


def test_filers_own_legal_name_is_not_another_company():
    text = "Albemarle Corporation supplies lithium to customers worldwide."
    assert "trade" not in PF.families(text, "ALB")
    assert "trade" in PF.families(text, "TSLA")


def test_footprint_needs_a_material_component_or_facility_not_just_a_region():
    assert PF.families("We operate a plant in the United States.", "TSLA") == []
    assert PF.families("We operate a lithium hydroxide plant in Nevada.", "TSLA") == ["footprint"]
