"""Curation: review proposed aliases, new nodes and skips before they change the node lists.

Decisions are appended to config/curation/<round>_review.jsonl; the draft itself is never edited
here, so rebuilding it keeps every decision. Nothing reaches the node lists until
`ripplekg curate apply` is run.
"""

import streamlit as st

from ripplekg import settings
from ripplekg.ontology.curation_review import (
    EDITABLE,
    ReviewStore,
    entries_table,
    entry_id,
    load_draft,
)
from ripplekg.ontology.schema import load_ontology

st.set_page_config(page_title="Curation · RippleKG", page_icon="🧹", layout="wide")
CURATION_DIR = settings.CONFIG_DIR / "curation"
SECTION_LABELS = {
    "aliases": "Alias of a node",
    "new_nodes": "New node",
    "skip": "Skip",
    "needs_review": "Undecided",
}


@st.cache_resource
def ontology():
    return load_ontology()


rounds = sorted(
    p.stem for p in CURATION_DIR.glob("*.yaml") if not p.stem.endswith(("_corrections", "_review"))
)
if not rounds:
    st.error("No curation draft found. Run `ripplekg curate draft` first.")
    st.stop()

with st.sidebar:
    st.header("Draft")
    round_name = st.selectbox("Round", rounds, index=len(rounds) - 1)

draft = load_draft(CURATION_DIR / f"{round_name}.yaml")
store = ReviewStore(CURATION_DIR / f"{round_name}_review.jsonl")
latest = store.latest()
table = entries_table(draft)
table["review"] = table["id"].map(lambda i: latest[i].decision if i in latest else "")
by_id = {entry_id(s, e["key"]): (s, e) for s in draft for e in draft[s]}
ont = ontology()
new_labels = sorted(e["label"] for e in draft["new_nodes"])
regions = sorted(
    {r for r, n in ont.nodes.items() if n.kind == "region"}
    | {e["label"] for e in draft["new_nodes"] if e["kind"] == "region"}
)
node_choices = sorted(set(ont.nodes) | set(new_labels))

with st.sidebar:
    st.header("Filters")
    sections = st.multiselect(
        "Section", list(SECTION_LABELS), default=list(SECTION_LABELS), format_func=SECTION_LABELS.get
    )
    kinds = st.multiselect("Kind (new nodes)", sorted(k for k in table["kind"].unique() if k))
    review_state = st.radio("Review", ["unreviewed", "all", "approve", "edit", "reject"], horizontal=True)
    flags = st.multiselect(
        "Show only", ["has issues", "corrected by reviewer pass", "pre-approved", "not pre-approved"]
    )
    min_seen = st.number_input("Seen at least", 0, 1000, 0)
    search = st.text_input("Search names")

f = table[table["section"].isin(sections) & (table["seen"] >= min_seen)]
if kinds:
    f = f[f["kind"].isin(kinds)]
if review_state == "unreviewed":
    f = f[f["review"] == ""]
elif review_state != "all":
    f = f[f["review"] == review_state]
if "has issues" in flags:
    f = f[f["issues"] > 0]
if "corrected by reviewer pass" in flags:
    f = f[f["corrected"]]
if "pre-approved" in flags:
    f = f[f["pre_approved"]]
if "not pre-approved" in flags:
    f = f[~f["pre_approved"]]
if search:
    s = search.lower()
    f = f[
        f["name"].str.lower().str.contains(s, regex=False)
        | f["target"].str.lower().str.contains(s, regex=False)
    ]
f = f.sort_values(["seen"], ascending=False)

st.title("🧹 Curation")
st.caption(
    f"Round `{round_name}`: proposed changes to the node lists. Decisions are saved to "
    f"`config/curation/{round_name}_review.jsonl`; nothing changes until `ripplekg curate apply`."
)
done = int((table["review"] != "").sum())
c = st.columns(5)
c[0].metric("Entries", f"{len(table):,}")
c[1].metric("Reviewed", f"{done:,}", f"{done / max(len(table), 1):.0%}")
c[2].metric("Approved", f"{int(table['review'].isin(['approve', 'edit']).sum()):,}")
c[3].metric("Rejected", f"{int((table['review'] == 'reject').sum()):,}")
c[4].metric("Shown", f"{len(f):,}")
st.progress(done / max(len(table), 1))

left, right = st.columns([5, 6])
with left:
    shown = f[
        [
            "section",
            "name",
            "target",
            "kind",
            "seen",
            "confidence",
            "pre_approved",
            "issues",
            "corrected",
            "review",
        ]
    ]
    event = st.dataframe(
        shown.assign(section=shown["section"].map(SECTION_LABELS)),
        hide_index=True,
        width="stretch",
        height=560,
        on_select="rerun",
        selection_mode="single-row",
        key=f"entries-{round_name}",
    )
    with st.expander(f"Bulk decision for all {len(f):,} shown entries"):
        bulk_note = st.text_input("Note for the bulk decision", key="bulk-note")
        sure = st.checkbox(f"I have checked these {len(f):,} entries")
        b1, b2 = st.columns(2)
        if b1.button("Approve all shown", disabled=not sure or f.empty):
            n = store.add_many(list(zip(f["section"], f["key"], strict=True)), "approve", bulk_note)
            st.toast(f"Approved {n} entries")
            st.rerun()
        if b2.button("Reject all shown", disabled=not sure or f.empty):
            n = store.add_many(list(zip(f["section"], f["key"], strict=True)), "reject", bulk_note)
            st.toast(f"Rejected {n} entries")
            st.rerun()

with right:
    if not event.selection.rows:
        st.info("Select an entry on the left to review it.")
        st.stop()
    row = f.iloc[event.selection.rows[0]]
    section, entry = by_id[row["id"]]
    prior = latest.get(row["id"])

    if section == "aliases":
        st.subheader(f"{entry['variants'][0] if entry.get('variants') else row['name']} → `{entry['node']}`")
        st.caption("Proposed: these names become aliases of an existing (or newly proposed) node.")
    elif section == "new_nodes":
        st.subheader(f"New {entry['kind']}: {entry['label']}")
    elif section == "skip":
        st.subheader(f"Skip: {entry.get('name', row['name'])}")
        st.caption("Proposed: stop reporting this name as unresolved.")
    else:
        st.subheader(f"Undecided: {entry.get('name', row['name'])}")

    if prior:
        st.success(f"Current decision: **{prior.decision}** {('· ' + prior.note) if prior.note else ''}")
    for issue in entry.get("issues") or []:
        st.warning(issue)
    if entry.get("review_note"):
        st.info(f"Reviewer pass: {entry['review_note']}")
    expected = ", ".join(entry.get("expected_kind") or []) or "-"
    mentioned = ", ".join(entry.get("mentioned_by") or []) or "-"
    st.markdown(
        f"**Seen** {entry.get('seen', 0)} times · **expected kind** {expected} · **mentioned by** {mentioned}"
    )
    if entry.get("variants"):
        st.markdown("**Spellings:** " + " · ".join(f"`{v}`" for v in entry["variants"]))
    if entry.get("reason"):
        conf = entry.get("confidence")
        st.markdown(f"**Model:** {entry['reason']}" + (f" (confidence {conf})" if conf is not None else ""))
    if entry.get("fuzzy_match"):
        st.markdown(f"**Close existing node:** {entry['fuzzy_match']}")
    for q in entry.get("quotes") or []:
        st.markdown(f"> {q}")

    st.divider()
    edits: dict = {}
    if section == "aliases":
        current = (prior.edits.get("node") if prior else None) or entry["node"]
        node = st.selectbox(
            "Alias of", node_choices, index=node_choices.index(current) if current in node_choices else 0
        )
        if node != entry["node"]:
            edits["node"] = node
    elif section == "new_nodes":
        prev = prior.edits if prior else {}
        a, b = st.columns(2)
        label = a.text_input("Label (at most 4 words)", prev.get("label", entry["label"]))
        kinds_all = ["company", "facility", "material", "component", "region", "authority", "program"]
        kind = b.selectbox("Kind", kinds_all, index=kinds_all.index(prev.get("kind", entry["kind"])))
        parent = a.text_input("Parent (node label, optional)", prev.get("parent", entry.get("parent", "")))
        region = ""
        if kind == "facility":
            cur_region = prev.get("region", entry.get("region", ""))
            region = b.selectbox(
                "Region",
                [""] + regions,
                index=([""] + regions).index(cur_region) if cur_region in regions else 0,
            )
        aliases = st.text_input(
            "Aliases (separated by ;)", "; ".join(prev.get("aliases", entry.get("aliases", [])))
        )
        for field_name, value, original in [
            ("label", label, entry["label"]),
            ("kind", kind, entry["kind"]),
            ("parent", parent, entry.get("parent", "")),
            ("region", region, entry.get("region", "")),
            ("aliases", [x.strip() for x in aliases.split(";") if x.strip()], entry.get("aliases", [])),
        ]:
            if (
                value != original
                and field_name in EDITABLE["new_nodes"]
                and not (field_name == "region" and kind != "facility")
            ):
                edits[field_name] = value
        if len(label.split()) > 4:
            st.error("Labels can be at most 4 words.")

    elif section == "needs_review":
        prev = prior.edits if prior else {}
        action = st.radio("Resolve as", ["alias", "skip"], horizontal=True,
                          index=["alias", "skip"].index(prev.get("action", "alias")))  # fmt: skip
        edits["action"] = action
        if action == "alias":
            current = prev.get("node", "")
            edits["node"] = st.selectbox(
                "Alias of", node_choices, index=node_choices.index(current) if current in node_choices else 0
            )

    note = st.text_input("Note (optional)", prior.note if prior else "")
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Approve" + (" with edits" if edits else ""), type="primary"):
        store.add(section, entry["key"], "edit" if edits else "approve", edits, note)
        st.toast("Saved")
        st.rerun()
    if b2.button("❌ Reject"):
        store.add(section, entry["key"], "reject", {}, note)
        st.toast("Saved")
        st.rerun()
    if b3.button("↩︎ Clear filters to see all"):
        st.session_state.clear()
        st.rerun()
