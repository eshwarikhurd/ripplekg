"""Graph Explorer: extracted triples, the edges they form, rejections, and review marks.

Read-only with respect to extracted data: review marks are appended to data/reviews/reviews.jsonl
and nothing is ever deleted.
"""

import pandas as pd
import streamlit as st

from ripplekg import settings
from ripplekg.explore.data import (
    aggregate_edges,
    available_runs,
    chunk_text,
    entity_table,
    load_run,
    unresolved_names,
)
from ripplekg.explore.network import KIND_COLORS, neighbourhood, network_html
from ripplekg.explore.reviews import LABELS, ReviewStore
from ripplekg.ontology.schema import load_ontology

st.set_page_config(page_title="Graph Explorer · RippleKG", page_icon="🕸️", layout="wide")
REVIEWS = ReviewStore(settings.DATA_DIR / "reviews" / "reviews.jsonl")


@st.cache_resource
def ontology():
    return load_ontology()


@st.cache_data(show_spinner="Re-verifying triples under the current rules…")
def triples_for(run_id: str) -> pd.DataFrame:
    run = next(r for r in available_runs() if r.id == run_id)
    return load_run(run, ontology())


@st.cache_data
def context_for(chunk_id: str) -> str:
    return chunk_text(chunk_id)


ont = ontology()
runs = available_runs()
if not runs:
    st.error("No extraction runs found. Run `ripplekg pilot` or `ripplekg extract run` first.")
    st.stop()

# ---------------------------------------------------------------- sidebar filters
with st.sidebar:
    st.header("Data")
    run = st.radio("Extraction run", runs, format_func=lambda r: r.label)
    triples = triples_for(run.id)

    st.header("Filters")
    verdict = st.radio("Triples", ["passed", "failed", "all"], horizontal=True)
    companies = sorted(n.id for n in ont.nodes.values() if n.kind == "company")
    picked_companies = st.multiselect("Company (filer or endpoint)", companies)
    relations = sorted(triples["canonical_relation"].replace("", pd.NA).dropna().unique())
    picked_relations = st.multiselect("Relation", relations)
    families = sorted(triples["family"].unique())
    picked_families = st.multiselect("Relation family", families)
    statuses = sorted(s for s in triples["status"].unique() if s)
    picked_status = st.multiselect("Status", statuses)
    forms = sorted(triples["form"].dropna().unique())
    picked_forms = st.multiselect("Form", forms)
    years = sorted(triples["year"].dropna().unique())
    year_range = (
        st.select_slider("Filing year", years, value=(years[0], years[-1])) if len(years) > 1 else None
    )
    search = st.text_input("Text in quote or names")
    review_filter = st.radio("Review marks", ["any", "unreviewed", "Y", "N", "?"], horizontal=True)

marks = REVIEWS.latest()
mark_of = dict(zip(marks["triple_key"], marks["label"], strict=True)) if len(marks) else {}
note_of = dict(zip(marks["triple_key"], marks["note"], strict=True)) if len(marks) else {}

f = triples
if picked_companies:
    sel = set(picked_companies)
    f = f[f["filer"].isin(sel) | f["source_ref"].isin(sel) | f["target_ref"].isin(sel)]
if picked_relations:
    f = f[f["canonical_relation"].isin(picked_relations)]
if picked_families:
    f = f[f["family"].isin(picked_families)]
if picked_status:
    f = f[f["status"].isin(picked_status)]
if picked_forms:
    f = f[f["form"].isin(picked_forms)]
if year_range:
    f = f[(f["year"] >= year_range[0]) & (f["year"] <= year_range[1])]
if search:
    s = search.lower()
    hay = (f["quote"].fillna("") + " " + f["source"].fillna("") + " " + f["target"].fillna("")).str.lower()
    f = f[hay.str.contains(s, regex=False)]
f = f.assign(review=f["triple_key"].map(mark_of).fillna(""))
if review_filter == "unreviewed":
    f = f[f["review"] == ""]
elif review_filter in LABELS:
    f = f[f["review"] == review_filter]

shown = f if verdict == "all" else f[f["ok"] == (verdict == "passed")]
edges = aggregate_edges(f)

# ---------------------------------------------------------------- header metrics
st.title("🕸️ Graph Explorer")
st.caption(f"{run.label} · every triple re-checked under the current ontology and verification rules")
c = st.columns(5)
c[0].metric("Triples (filtered)", f"{len(f):,}")
c[1].metric("Passed", f"{int(f['ok'].sum()):,}")
c[2].metric("Graph edges", f"{len(edges):,}")
c[3].metric("Nodes", f"{len(set(edges['source_ref']) | set(edges['target_ref'])):,}")
c[4].metric("Reviewed", f"{int((f['review'] != '').sum()):,}")


def review_widget(row: pd.Series, where: str) -> None:
    """Y / N / ? buttons and a note for one triple; each save appends a mark."""
    key = row["triple_key"]
    current = mark_of.get(key, "")
    cols = st.columns([3, 5, 1])
    choice = cols[0].radio(
        "Mark",
        ["Y", "N", "?"],
        index=["Y", "N", "?"].index(current) if current else None,
        horizontal=True,
        key=f"{where}-label-{key}",
        format_func=lambda x: f"{x} · {LABELS[x]}",
    )
    note = cols[1].text_input("Note", value=note_of.get(key, ""), key=f"{where}-note-{key}")
    if cols[2].button("Save", key=f"{where}-save-{key}", disabled=choice is None):
        REVIEWS.add(key, run.id, choice, note)
        st.toast(f"Saved {choice} for this triple")
        st.rerun()


def triple_card(row: pd.Series, where: str) -> None:
    verdict_txt = "✅ passed" if row["ok"] else f"❌ {row['reason']}"
    head = f"**{row['source']}** —{row['relation']}→ **{row['target']}** · {row['status']} · {verdict_txt}"
    with st.container(border=True):
        st.markdown(head)
        if row["ok"]:
            st.caption(
                f"Graph edge: `{row['source_ref']}` —{row['canonical_relation']}→ `{row['target_ref']}`"
            )
        quote = row["quote_full"] or row["quote"]
        st.markdown(f"> {quote}")
        filed = str(row.get("filing_date") or "")[:10]
        meta = f"{row['filer']} · {row.get('form') or ''} filed {filed} · section {row.get('section') or '?'}"
        if pd.notna(row.get("strength")):
            meta += f" · strength {row['strength']:g} ({row['strength_basis']})"
        st.caption(meta + (f" · [filing]({row['url']})" if isinstance(row.get("url"), str) else ""))
        with st.expander("Chunk text and raw model output"):
            st.text(context_for(row["chunk_id"]))
            st.json({k: row[k] for k in ("source", "relation", "target", "status", "quote", "strength",
                                          "strength_basis", "material", "component", "family")})  # fmt: skip
        review_widget(row, where)


tab_edges, tab_entities, tab_network, tab_rejected, tab_reviews = st.tabs(
    ["Edges", "Entities", "Network", "Rejected", "Reviews"]
)

# ---------------------------------------------------------------- edges
with tab_edges:
    st.write("Passed triples merged into graph edges. Select a row to see every triple behind it.")
    event = st.dataframe(
        edges, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="edges"
    )
    if event.selection.rows:
        e = edges.iloc[event.selection.rows[0]]
        st.subheader(f"`{e.source_ref}` —{e.canonical_relation}→ `{e.target_ref}`")
        support = f[
            f["ok"]
            & (f["source_ref"] == e.source_ref)
            & (f["canonical_relation"] == e.canonical_relation)
            & (f["target_ref"] == e.target_ref)
        ].sort_values("published_at")
        st.caption(f"{len(support)} contributing triple(s) from {support['doc_id'].nunique()} filing(s)")
        for _, row in support.head(50).iterrows():
            triple_card(row, "edge")

# ---------------------------------------------------------------- entities
with tab_entities:
    ents = entity_table(f, ont)
    kinds = st.multiselect("Node kind", sorted(ents["kind"].unique()) if len(ents) else [])
    st.dataframe(ents[ents["kind"].isin(kinds)] if kinds else ents, hide_index=True, width="stretch")

# ---------------------------------------------------------------- network
with tab_network:
    nodes = sorted(set(edges["source_ref"]) | set(edges["target_ref"]))
    cols = st.columns([3, 1, 1])
    focus = cols[0].selectbox("Centre on", [""] + nodes, format_func=lambda x: x or "(all edges)")
    hops = cols[1].slider("Hops", 1, 3, 1, disabled=not focus)
    limit = cols[2].number_input("Max edges", 20, 1000, 250, step=50)
    sub = neighbourhood(edges, focus or None, hops).head(int(limit))
    st.caption(
        " · ".join(f"<span style='color:{c}'>●</span> {k}" for k, c in KIND_COLORS.items())
        + f" · showing {len(sub)} of {len(edges)} edges",
        unsafe_allow_html=True,
    )
    if sub.empty:
        st.info("No edges for these filters.")
    else:
        # Only curated node ids/names and relation counts go into this HTML, never raw model output.
        st.iframe(network_html(sub, ont, focus or None), height=680)

# ---------------------------------------------------------------- rejected
with tab_rejected:
    failed = f[~f["ok"]]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Why triples failed")
        reasons = failed["reason"].str.replace(r" \(.*\)", "", regex=True).value_counts()
        st.bar_chart(reasons, horizontal=True)
    with right:
        st.subheader("Unresolved names (curation queue)")
        st.dataframe(unresolved_names(f), hide_index=True, width="stretch", height=320)
    reason = st.selectbox("Show failed triples with reason", ["(all)"] + list(reasons.index))
    rows = failed if reason == "(all)" else failed[failed["reason"].str.startswith(reason)]
    table = rows[["source", "relation", "target", "status", "filer", "reason", "quote", "review"]]
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="rejected",
    )
    if event.selection.rows:
        triple_card(rows.iloc[event.selection.rows[0]], "rejected")

# ---------------------------------------------------------------- reviews
with tab_reviews:
    st.write(
        "Your review marks (latest mark per triple), stored in `data/reviews/reviews.jsonl`. "
        "Data is never deleted."
    )
    if marks.empty:
        st.info("No marks yet. Use the Y / N / ? buttons on any triple.")
    else:
        joined = marks.merge(
            triples[["triple_key", "source", "relation", "target", "ok", "reason", "quote"]],
            on="triple_key",
            how="left",
        )
        st.dataframe(joined[joined["run"] == run.id], hide_index=True, width="stretch")
        counts = joined[joined["run"] == run.id]["label"].value_counts()
        st.caption(" · ".join(f"{k}: {v}" for k, v in counts.items()))
