from ripplekg.ingestion.sections import split_sections

FILLER = "This paragraph describes our business in detail. " * 10


def doc(*parts: str) -> str:
    return "\n".join(parts)


def test_10k_keeps_longest_span_not_table_of_contents():
    text = doc(
        "Table of contents",
        "Item 1. Business 3",
        "Item 1A. Risk Factors 9",
        "Item 7. Management's Discussion 30",
        "PART I",
        "Item 1. Business",
        "We supply cells to Borealis Motors. " + FILLER,
        "Item 1A. Risk Factors",
        "We rely on a single supplier. " + FILLER,
        "Item 1B. Unresolved Staff Comments",
        "None.",
        "Item 7. Management's Discussion and Analysis",
        "Revenue grew. " + FILLER,
    )
    sections = {s.name: s.text for s in split_sections(text, "10-K")}
    assert list(sections) == ["business", "risk_factors", "mda"]
    assert "Borealis Motors" in sections["business"]
    assert "single supplier" in sections["risk_factors"]
    assert "Unresolved" not in sections["risk_factors"]


def test_10q_distinguishes_part_i_and_part_ii_items():
    text = doc(
        "PART I - FINANCIAL INFORMATION",
        "Item 1. Financial Statements",
        "Balance sheet. " + FILLER,
        "Item 2. Management's Discussion and Analysis",
        "Our offtake partner delayed deliveries. " + FILLER,
        "PART II - OTHER INFORMATION",
        "Item 1. Legal Proceedings",
        "An arbitration claim was filed against us. " + FILLER,
        "Item 1A. Risk Factors",
        "New export controls may affect graphite. " + FILLER,
    )
    sections = {s.name: s.text for s in split_sections(text, "10-Q")}
    assert list(sections) == ["mda", "legal_proceedings", "risk_factors"]
    assert "Balance sheet" not in "".join(sections.values())


def test_8k_items_with_two_digit_subnumbers():
    text = doc(
        "Item 1.01 Entry into a Material Definitive Agreement.",
        "On March 3 we signed an offtake agreement with Acme Lithium. " + FILLER,
        "Item 9.01 Financial Statements and Exhibits.",
        "Exhibit 99.1 Press release.",
    )
    sections = split_sections(text, "8-K")
    assert [s.name for s in sections] == ["material_agreement"]
    assert "Exhibit 99.1" not in sections[0].text


def test_20f_sub_items_fold_into_parent_item():
    text = doc(
        "Item 4. Information on the Company",
        "Short intro.",
        "Item 4.B Business Overview",
        "We mine lithium in Chile and sell to cathode makers. " + FILLER,
        "Item 5. Operating and Financial Review",
        "Prices fell. " + FILLER,
    )
    sections = {s.name: s.text for s in split_sections(text, "20-F")}
    assert "cathode makers" in sections["business"]
    assert "operating_review" in sections


def test_unknown_form_or_no_headings_falls_back_to_full_text():
    assert [s.name for s in split_sections("Annual information form. " + FILLER, "40-F")] == ["full_text"]
    assert [s.name for s in split_sections("No item headings here. " + FILLER, "10-K")] == ["full_text"]
    assert split_sections("   ", "40-F") == []


def test_short_8k_items_are_kept():
    text = doc(
        "Item 8.01. Other Events.",
        "On January 3 we announced a recall of 5,000 vehicles.",
        "Item 9.01. Exhibits.",
    )
    assert [s.name for s in split_sections(text, "8-K")] == ["other_events"]


def test_8k_heading_without_space_after_number():
    text = doc("Item 7.01Regulation FD Disclosure.", "We presented at a conference about our cell supply.")
    assert [s.name for s in split_sections(text, "8-K")] == ["reg_fd"]
