"""RippleKG app home. Run with: uv run streamlit run app/Home.py"""

import streamlit as st

st.set_page_config(page_title="RippleKG", page_icon="🔋", layout="wide")
st.title("RippleKG")
st.write(
    "Temporal knowledge graph of the EV-battery and critical-minerals supply chain, built from SEC filings."
)
st.markdown(
    """
**Pages**

- **Graph Explorer**: browse extracted triples and the edges they form, see the evidence behind each edge,
  inspect rejected triples and unresolved names, and mark triples correct or wrong.

Pages for questions, event replay, results and data quality come with later stages.
"""
)
