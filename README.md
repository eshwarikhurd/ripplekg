# RippleKG: Tracing Supply-Chain Shocks Through a Temporal Knowledge Graph

## Abstract

When a disruption hits one company, such as a mine closure, a factory fire or a port lockdown, the companies that suffer
most are often two or three steps away along the supply chain and are never mentioned in the news about the event.
Retrieval-augmented generation (RAG) that finds text similar to a question cannot see these chains, because the links
between companies are spread across many separate disclosures.

RippleKG builds a temporal, directed knowledge graph of the electric-vehicle battery and critical-minerals supply chain
(miners, refiners, cathode and cell makers, and automakers) from public SEC filings (10-K, 10-Q and 8-K), stored in
FalkorDB. Each relationship carries its source quote, the period in which it held, and, where filings disclose it, a
dependency strength such as the share of revenue that comes from a customer. Given a disruption at one company,
RippleKG traverses the graph as of the event date and scores how strongly the shock is likely to reach each other
company, weakening with every hop according to dependency strength and time.

Instead of grading answers with an LLM judge, RippleKG is evaluated against what actually happened. A benchmark of real,
dated supply-chain disruptions is paired with an event study of stock prices: a predicted bystander counts as correct
if it shows an abnormal return in the days after the event. Ripple scoring is compared with plain graph distance and
with vector-only RAG, and every relationship's quoted evidence is checked programmatically against the original filing.

## Status

Early stage: the repository currently holds the project description only.

## Author

Eshwari Khurd

## License

Licensed under the [Apache License 2.0](LICENSE). If you use or adapt this work, keep the [NOTICE](NOTICE) file and
please cite it using [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button).
