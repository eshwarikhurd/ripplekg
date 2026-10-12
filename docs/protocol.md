# Evaluation protocol

**Status: frozen.** This protocol and [config/events.yaml](../config/events.yaml) were committed before any
stock-price data was downloaded, any filing was parsed, and any graph was built. The git history is the
record of that order. Any later change is listed under [Deviations](#11-deviations) with a date and a
reason. Nothing is changed silently.

## 1. Question and hypotheses

> Given a dated disruption, can a temporal, dependency-weighted supply-chain graph built only from SEC
> filings published before the disruption predict which *other* companies' stocks react to it?

- **H1 (primary).** RippleKG ripple scoring achieves a higher mean average precision (AP) on the test
  events than **graph distance alone** (hop-count baseline on the same graph).
- **H2 (primary).** RippleKG achieves a higher mean AP on the test events than **vector RAG** over the same
  filings.
- **Secondary.** Comparisons with the zero-shot LLM, degree, same-tier and random baselines; the four
  ablations in section 6; precision@k, recall@k and rank correlation; and the sign of the reaction.

H1 tests whether the dependency weights, direction and time handling add anything beyond the graph's
connectivity. H2 tests whether the graph adds anything beyond retrieving similar text.

## 2. Events

### 2.1 Inclusion criteria

An event is included if all of the following hold:

1. It is a discrete supply- or demand-side shock to the EV-battery or critical-minerals chain: a
   production halt, recall, mine suspension, bankruptcy, export restriction, tariff, subsidy rule, or a
   comparable change to supply or demand.
2. Its first public date can be pinned to a single calendar day using a primary source (an SEC filing,
   an official notice) or, failing that, two independent reports.
3. Its origin is a company in [config/universe.yaml](../config/universe.yaml) or a material, region,
   authority or program node.
4. It falls between 2021-01-01 and 2025-12-31, so the graph has at least two years of filings before it.
5. Its event window [day 0, day 0 + 3] does not overlap another primary event's window. When two
   candidates overlap, the later one is dropped. `ripplekg events-check` enforces this.

### 2.2 The benchmark

23 events: **20 primary**, used for all metrics, and **3 macro**, reported separately.

| Group | Count | Events |
|---|---|---|
| Company origin | 12 | GM Bolt recall, Ford Lightning fire, Tesla Red Sea halt, Tesla arson, Rivian parts shortage, Li-Cycle CCAA, Ford three-row EV cancellation, Northvolt Chapter 11, MP–DoD partnership, Hyundai–LGES raid, Lithium Americas stake, GM EV charge |
| Material, region or policy origin | 8 | CATL Jianxiawo mine halt¹, Zimbabwe lithium-ore ban, Chile lithium strategy, China graphite controls, DRC cobalt suspension, US–China EV tariffs, IRA deal, FEOC guidance |
| Macro (separate) | 3 | Suez blockage, Shanghai lockdown, China rare-earth controls (April 2025, confounded by the tariff sell-off) |

¹ Its origin includes `material:lithium`, so it is stratified as non-company.

### 2.3 Candidates considered and excluded

| Candidate | Reason |
|---|---|
| Mexico cancels Ganfeng's Sonora lithium concessions (Aug 2023) | No source pins the day; criterion 2 |
| Ford pauses its Marshall, Michigan battery plant (Sep 2023) | Inside the UAW strike, which moved the same stocks |
| Freeport Grasberg force majeure (2025-09-24) | Same day as the Lithium Americas event; mainly a copper shock |
| Rivian commercial-van parts pause (Aug 2024) | Superseded by the larger, dated October 2024 disclosure |
| Arcadium–Rio Tinto acquisition (Oct 2024) | Corporate transaction, not a supply or demand shock |
| Li-Cycle pauses its Rochester hub (2023-10-23) | Window overlaps China's graphite controls (2023-10-20); the later event is dropped (criterion 5) |

### 2.4 Dev/test split

Primary events are split before any results exist, by `ripplekg.events.assign_split`:

- stratified by origin kind (12 company and 8 non-company events)
- within each stratum, ids are sorted and then shuffled with seed **20261011**
- the first **4 company** and **2 non-company** events go to **dev**, and the rest go to **test**

The result is **6 dev events and 14 test events**, as recorded in `events.yaml`. `ripplekg events-check`
and the unit tests recompute the split and fail if the file disagrees.

- Dev events are used for tuning and debugging, without limit.
- Test events are scored once per method, after tuning is frozen (section 7).

## 3. Day 0

- `announced` is the date the information first became public, using the first credible report even if a
  formal announcement came later (for example, the tariff reports on 2024-05-10 before the formal
  2024-05-14 announcement).
- **Day 0** is the first US trading day on which the market could react:
  - `before_open`, `during_market` or `unknown` on a trading day: day 0 = `announced`
  - `after_close`: the next trading day
  - `non_trading_day`: the next trading day
- Trading days come from the SPY price calendar. If a recorded `day0` is not a trading day in that
  calendar, the next trading day is used and the change is logged as a deviation.

## 4. Scored companies per event

For each event, a company is **scored** if all of the following hold:

1. It is an SEC filer in the universe with a US-listed ticker.
2. Daily prices exist for at least 200 of the 221 trading days in the estimation window, and for every day
   in the event window.
3. It is not an origin company, not in the event's `named` list, and not a predecessor or successor
   (`successor` field) of an origin company.
4. **No own-news confound.** It filed none of the following with SEC dated from day −1 to day +3:
   - an 8-K with item 1.01, 1.02, 1.03, 2.01, 2.02, 2.05 or 2.06 (material agreements, bankruptcy,
     acquisitions, results, restructuring, impairment)
   - for foreign private issuers, a 6-K, 20-F or 40-F whose EDGAR description matches
     `result|earning|quarter|annual report` (case-insensitive)

   This check is programmatic, from the submissions API.

Companies excluded by rules 2–4 are listed per event in the results.

## 5. Ground truth: event study

- **Prices.** Daily adjusted closes from `yfinance`, or Stooq if `yfinance` has no data for a ticker.
  Simple daily returns. Prices are cached with their download date.
- **Market model.** For each scored company, OLS of its returns on SPY returns over trading days
  [−250, −30] relative to day 0: r<sub>i,t</sub> = α<sub>i</sub> + β<sub>i</sub> r<sub>SPY,t</sub> + ε<sub>i,t</sub>.
- **Abnormal return.** AR<sub>i,t</sub> = r<sub>i,t</sub> − (α̂<sub>i</sub> + β̂<sub>i</sub> r<sub>SPY,t</sub>).
  **CAR** is summed over [0, +3] (primary window).
- **Test.** t<sub>i</sub> = CAR<sub>i</sub> / (σ̂<sub>i</sub> √L), where σ̂<sub>i</sub> is the residual
  standard deviation from the estimation window and L = 4 days.
- **Hit.** A scored company is a **hit** if |t<sub>i</sub>| > 1.96 (two-sided, p < 0.05). Direction is
  recorded but not required for a hit.
- **Robustness**, reported but not used for H1 or H2:
  - windows [0, +1] and [0, +5]
  - for company-origin events, a sector-adjusted model with the Global X Lithium & Battery Tech ETF (LIT)
    in place of SPY. This is not the primary model because, for material and policy shocks, a sector
    benchmark absorbs the very reaction being measured.

## 6. Methods

Every method outputs a score for **every** scored company of an event. A higher score means a stronger
predicted reaction. Companies a method cannot reach get score 0.

**Tie-breaking.** Ties are broken by a fixed random permutation seeded with the SHA-256 of the event id,
the same for every method, so no method benefits from alphabetical order.

| Method | Role | Description |
|---|---|---|
| **RippleKG** | proposed | Directional ripple scoring over the graph as of day 0, weighted by dependency strength, hop decay and recency |
| Hop distance | H1 baseline | 1 / (shortest-path hops) in the same as-of graph, ignoring direction and weights |
| Vector RAG | H2 baseline | Embed chunks from filings dated before day 0. Query with the event description and retrieve the top 10 chunks; the local LLM returns a ranked list of universe tickers, scored by reciprocal rank |
| Zero-shot LLM | secondary | The same local LLM and output schema, with no retrieval |
| Degree | secondary | As-of graph degree (a "hub" prior) |
| Same tier | secondary | Companies in an origin company's tier, ranked by average daily dollar volume over the estimation window; 0 for other tiers. Non-company origins use the tiers linked to the origin material |
| Random | secondary | Expected metric values computed analytically from the hit rate |
| Ablations | secondary | RippleKG without strength; without recency; undirected; with hop decay only |

**Information cut-off.** The graph and the RAG index use only filings accepted by EDGAR before 09:30 ET on
day 0. A unit test asserts that no as-of query returns an edge whose evidence was filed on or after
day 0.

LLM baselines are given the event title and date, never its outcome. The LLM's knowledge cut-off is
recorded, and results are also reported separately for events after that cut-off, since the model may
already know earlier outcomes.

## 7. Tuning and test discipline

- RippleKG settings are tuned **on the 6 dev events only**, by grid search maximising mean dev AP:
  - hop decay λ ∈ {0.3, 0.5, 0.7}
  - recency half-life τ ∈ {180, 365, 730} days
  - maximum hops H ∈ {2, 3}
  - default strength for edges without a disclosed percentage ∈ {0.1, 0.2, 0.3}
- Baselines with settings (vector RAG top-k, LLM prompt) are fixed on dev in the same way.
- The extraction model and prompt version are frozen on the extraction gold set before the full run.
- Each method is then run **once** on the 14 test events. If a bug forces a re-run, the bug, the fix and
  both results are logged under Deviations.

## 8. Metrics

- **Primary: average precision (AP)** per event over the scored companies, using the hits as relevant
  items. The headline number is mean AP over test events.
- Events with no hits have undefined AP. They are reported, but left out of mean AP.
- **Secondary:**
  - precision@k and recall@k, for k ∈ {3, 5, 10}
  - Spearman correlation between score and |CAR| across all scored companies
  - the hit rate per event (which sets the random baseline)
  - sign accuracy: the share of hits whose CAR sign matches the method's predicted sign (RippleKG only; uses the `COMPETES_WITH` and event `sign` fields)

## 9. Statistics

- **H1 and H2:** paired, two-sided Wilcoxon signed-rank tests on per-event AP differences across test
  events, with Holm correction across the two hypotheses at α = 0.05.
- **Effect size:** mean AP difference with a 95% bootstrap confidence interval (10,000 resamples of test
  events, seed 20261011).
- With 14 test events, the study is powered only for large effects. Effect sizes and CIs are reported
  alongside the p-values, and every per-event result goes in the appendix, including failures.

## 10. What is reported

- All methods on all test events: mean AP with CI, p-values for H1 and H2, and the secondary metrics.
- Per-event tables: scored companies, hits with CAR and t, each method's top 10, and exclusions with
  reasons.
- Dev results, labelled as dev.
- Macro-bucket results, labelled as macro.
- Robustness windows and the sector-adjusted model.
- The results before and after the LLM's knowledge cut-off.

## 11. Deviations

Each entry: date, what changed, why, and whether it was decided before or after seeing any test-event
result.

**D1 (2026-10-12): which LLM.** Section 6 said "the local LLM". The LLM is now chosen by an extraction
comparison on the hand-labelled gold set: Claude Haiku 5.5 (Anthropic API) against a local Ollama model
(Qwen 7–8B). The more accurate one is used, or Ollama if they are close. Whichever wins is used by
**every** LLM-based method (RippleKG extraction, vector RAG and zero-shot), and its name and knowledge
cut-off are recorded with the results. Reason: a local 7–8B model on the available 16 GB machine was
the main risk to extraction quality and run time. Decided **before** any price data was downloaded and
before any test-event result existed.

**D2 (2026-10-12): staged corpus.** The graph is first built from annual reports only (10-K, 20-F,
40-F: 10,431 pre-filtered chunks covering 45 companies). 10-Q and 8-K filings are added only if the dev
events show the graph is too sparse; that decision is made on dev events alone, and the corpus used for
the test run is fixed before it. Reason: cost and run time. Decided **before** any result.
