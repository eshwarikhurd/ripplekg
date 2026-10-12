from ripplekg.ingestion.html_text import html_to_text

PAGE = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><head><title>10-K</title><style>p{}</style></head>
<body>
<div style="display:none"><ix:header>dei:EntityName hidden facts</ix:header></div>
<p>Item&#160;1. Business</p>
<p>We sell <b>battery cells</b> to Borealis&nbsp;Motors.</p>
<!-- a comment -->
<table>
  <tr><th>Customer</th><th>Share of revenue</th></tr>
  <tr><td>Borealis Motors</td><td>41%</td></tr>
</table>
<p>12</p>
<p><a href="#toc">Table of Contents</a></p>
<div>Line one<br/>Line two</div>
<script>var x = 1;</script>
</body></html>"""


def test_html_to_text():
    text = html_to_text(PAGE.encode())
    lines = [line for line in text.splitlines() if line]
    assert lines == [
        "Item 1. Business",
        "We sell battery cells to Borealis Motors.",
        "Customer | Share of revenue",
        "Borealis Motors | 41%",
        "Line one",
        "Line two",
    ]


def test_hidden_xbrl_scripts_and_comments_are_removed():
    text = html_to_text(PAGE)
    for gone in ("hidden facts", "var x", "a comment", "Table of Contents", "10-K"):
        assert gone not in text


def test_paragraph_breaks_are_kept_as_blank_lines():
    text = html_to_text("<p>First.</p><p>12</p><p>Second.</p>")
    assert text == "First.\n\nSecond."


def test_empty_input():
    assert html_to_text(b"   ") == ""
