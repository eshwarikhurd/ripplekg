"""Command-line entry point: `ripplekg <command>`."""

from __future__ import annotations

import argparse
import sys

from ripplekg import settings
from ripplekg.download.client import EdgarClient
from ripplekg.ontology.universe import load_universe


def _client() -> EdgarClient:
    return EdgarClient(settings.sec_user_agent(), cache_dir=settings.HTTP_CACHE_DIR)


def cmd_universe_check(args: argparse.Namespace) -> int:
    from ripplekg.download.coverage import check_filers

    universe = load_universe()
    filers = check_filers(_client(), universe, args.start, args.end)
    out = settings.RESULTS_DIR / "coverage" / "filers.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    filers.to_csv(out, index=False)
    cols = ["id", "cik", "sec_name", "sec_tickers", "annual_form", "n_annual", "first_annual", "last_annual"]
    print(filers[cols].to_string(index=False))
    missing = filers.loc[filers["n_annual"] == 0, "id"].tolist()
    print(f"\n{len(filers)} filers checked; no annual report in window: {', '.join(missing) or 'none'}")
    print(f"wrote {out.relative_to(settings.ROOT)}")
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    from ripplekg.download import coverage

    universe = load_universe()
    client = _client()
    out_dir = settings.RESULTS_DIR / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("checking filers…", flush=True)
    filers = coverage.check_filers(client, universe, args.start, args.end)
    print("counting co-mentions…", flush=True)
    mentions = coverage.co_mentions(
        client, universe, args.start, args.end, progress=lambda m: print(m, flush=True)
    )
    print("counting supply phrases…", flush=True)
    phrases = coverage.phrase_counts(client, universe, args.start, args.end)

    pairs = coverage.pair_table(mentions)
    summary = coverage.summarise(universe, filers, pairs)
    for name, df in [("filers", filers), ("mentions", mentions), ("pairs", pairs), ("phrases", phrases)]:
        df.to_csv(out_dir / f"{name}.csv", index=False)
    report = settings.DOCS_DIR / "coverage.md"
    coverage.write_report(report, universe, filers, pairs, phrases, summary, args.start, args.end)

    print(
        f"\n{'GO' if summary.go else 'NO-GO'}: {summary.n_scored} scorable companies "
        f"(need {coverage.GO_MIN_SCORED}), {summary.n_undirected_pairs} co-mentioned pairs "
        f"(need {coverage.GO_MIN_PAIRS}); {client.network_calls} network requests"
    )
    print(f"wrote {report.relative_to(settings.ROOT)} and {out_dir.relative_to(settings.ROOT)}/*.csv")
    return 0


def cmd_events_check(args: argparse.Namespace) -> int:
    from ripplekg.evaluation.events import assign_split, load_events

    events = load_events()
    expected = assign_split(events)
    for e in sorted(events, key=lambda e: (e.bucket != "primary", e.split or "", e.day0)):
        print(f"{e.day0}  {e.bucket:7} {e.split or '-':4} {e.origin_kind:11} {e.id}")
    mismatched = [e.id for e in events if e.bucket == "primary" and e.split != expected[e.id]]
    if mismatched:
        print(f"\nsplit differs from the pre-registered assignment: {', '.join(mismatched)}", file=sys.stderr)
        return 1
    n_primary = sum(e.bucket == "primary" for e in events)
    print(f"\n{len(events)} events valid ({n_primary} primary); split matches pre-registration")
    return 0


def cmd_ontology_doc(args: argparse.Namespace) -> int:
    from ripplekg.ontology.doc import render
    from ripplekg.ontology.schema import example_errors, load_examples, load_ontology

    ont, examples = load_ontology(), load_examples()
    errors = example_errors(ont, examples)
    if errors:
        print("invalid relation examples:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1
    out = settings.DOCS_DIR / "ontology.md"
    out.write_text(render(ont, examples))
    n_nodes = sum(n.kind != "company" for n in ont.nodes.values())
    print(f"{len(ont.relations)} relations, {n_nodes} curated nodes, {len(examples)} examples")
    print(f"wrote {out.relative_to(settings.ROOT)}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from ripplekg.download import corpus

    df = corpus.build_index(
        _client(), load_universe(), args.start, args.end, progress=lambda m: print(m, flush=True)
    )
    out = settings.DATA_DIR / "index" / "filings.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    report = settings.DOCS_DIR / "corpus.md"
    corpus.write_report(report, df, args.start, args.end)
    print(f"\n{len(df)} filings indexed, {int(df['selected'].sum())} selected")
    print(f"wrote {out.relative_to(settings.ROOT)} and {report.relative_to(settings.ROOT)}")
    return 0


def _selected_filings(limit: int | None):
    import pandas as pd

    df = pd.read_parquet(settings.DATA_DIR / "index" / "filings.parquet")
    df = df[df["selected"]].sort_values(["filing_date", "accession"])
    return df.head(limit) if limit else df


def cmd_fetch(args: argparse.Namespace) -> int:
    from ripplekg.download.fetch import fetch_filings

    selected = _selected_filings(args.limit)
    manifest = fetch_filings(
        _client(),
        selected,
        settings.DATA_DIR / "raw",
        settings.DATA_DIR / "raw" / "manifest.parquet",
        progress=lambda m: print(m, flush=True),
    )
    mb = manifest["bytes"].sum() / 1e6
    print(
        f"\n{manifest['accession'].nunique()} filings, {len(manifest)} files, "
        f"{mb:,.0f} MB raw (stored gzipped)"
    )
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    from ripplekg.download.fetch import load_manifest
    from ripplekg.ingestion.documents import DocumentStore
    from ripplekg.ingestion.edgar_adapter import source_hash, to_document

    raw_dir = settings.DATA_DIR / "raw"
    manifest = load_manifest(raw_dir / "manifest.parquet")
    store = DocumentStore(settings.DATA_DIR / "documents")
    files = {acc: g.to_dict("records") for acc, g in manifest.groupby("accession")}
    built = skipped = empty = 0
    for filing in _selected_filings(args.limit).to_dict("records"):
        rows = files.get(filing["accession"])
        if not rows:
            continue
        existing = store.read(filing["company_id"], f"edgar:{filing['accession']}")
        if existing is not None and existing.source_sha256 == source_hash(rows) and not args.force:
            skipped += 1
            continue
        doc = to_document(filing, rows, raw_dir)
        store.write(doc)
        built += 1
        empty += doc.n_chars == 0
        if built % 100 == 0:
            print(f"parsed {built}", flush=True)
    print(f"\n{built} documents built, {skipped} unchanged, {empty} with no text")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    import pandas as pd

    from ripplekg.ingestion.chunk_report import write_report
    from ripplekg.ingestion.chunking import chunk_documents
    from ripplekg.ingestion.documents import DocumentStore
    from ripplekg.ingestion.prefilter import Prefilter
    from ripplekg.ontology.schema import load_ontology

    chunks, occurrences, stats = chunk_documents(DocumentStore(settings.DATA_DIR / "documents"))
    pf = Prefilter.from_ontology(load_ontology())
    df = pd.DataFrame([c.__dict__ for c in chunks])
    df["n_chars"] = df["text"].str.len()
    df["families"] = [pf.families(c.text, c.company_id) for c in chunks]
    df["mentions"] = [pf.other_companies(c.text, c.company_id) for c in chunks]

    out = settings.DATA_DIR / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "chunks.parquet", index=False)
    pd.DataFrame(occurrences).to_parquet(out / "occurrences.parquet", index=False)
    cost = write_report(settings.DOCS_DIR / "chunks.md", df, stats)
    n_sel = int((df["families"].map(len) > 0).sum())
    print(f"{len(df):,} chunks, {n_sel:,} pass the pre-filter, {cost.calls:,} LLM calls")
    print(f"estimated Haiku cost: ${cost.standard_usd:.2f} standard, ${cost.batch_usd:.2f} batch")
    print("wrote data/chunks/*.parquet and docs/chunks.md")
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    import pandas as pd

    from ripplekg.extraction.llm import ClaudeProvider, OllamaProvider, ResponseCache
    from ripplekg.extraction.runner import ANNUAL_FORMS, run_extraction, sample_chunks, write_pilot_report
    from ripplekg.ontology.schema import load_ontology

    chunks = pd.read_parquet(settings.DATA_DIR / "chunks" / "chunks.parquet")
    stage = chunks[chunks["form"].isin(ANNUAL_FORMS) & (chunks["families"].map(len) > 0)]
    sample = sample_chunks(stage, args.sample, args.seed)
    if args.provider == "ollama":
        model = args.model if args.model != "claude-haiku-5-5" else "qwen2.5:7b"
        provider = OllamaProvider(model=model)
        workers = 1  # one local model: parallel requests only queue
    else:
        provider = ClaudeProvider(model=args.model, effort=args.effort)
        workers = 4
    cache = ResponseCache(settings.DATA_DIR / "extractions" / "cache")
    res = run_extraction(
        sample,
        provider,
        cache,
        load_ontology(),
        load_universe(),
        workers=workers,
        progress=lambda m: print(m, flush=True),
    )
    out = settings.RESULTS_DIR / "pilot" / args.provider
    out.mkdir(parents=True, exist_ok=True)
    res.outcomes.to_parquet(out / "outcomes.parquet", index=False)
    res.edges.to_parquet(out / "edges.parquet", index=False)
    full_calls = int(stage["families"].map(len).sum())
    print(write_pilot_report(out / "pilot.md", res, len(sample), full_calls, provider.model))
    print(f"\nfull report: {(out / 'pilot.md').relative_to(settings.ROOT)}")
    return 0


def cmd_curate(args: argparse.Namespace) -> int:
    import json

    import yaml

    from ripplekg.explore.data import available_runs, load_run
    from ripplekg.extraction.llm import ClaudeProvider, ResponseCache
    from ripplekg.ontology import curation
    from ripplekg.ontology.schema import load_ontology

    ont = load_ontology()
    run = next(r for r in available_runs() if r.id == args.run)
    triples = load_run(run, ont)
    groups = curation.collect_groups(triples, ont, args.min_count)
    suggestions = curation.fuzzy_suggestions(groups, ont)
    generic = [g for g in groups if curation.is_generic(g.label)]
    to_triage = [g for g in groups if not curation.is_generic(g.label)]
    print(f"{len(groups)} name groups seen >= {args.min_count} times: {len(generic)} generic (rule), "
          f"{len(to_triage)} for LLM triage, {len(suggestions)} with a fuzzy match")  # fmt: skip
    provider = ClaudeProvider(model=args.model, effort="low")
    saved = settings.CONFIG_DIR / "curation" / f"{args.round}_decisions.json"
    known = {} if args.retriage or not saved.exists() else json.loads(saved.read_text())
    decisions, cost = curation.triage(
        to_triage, suggestions, ont, provider, ResponseCache(settings.DATA_DIR / "curation" / "cache"), known
    )
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(decisions, indent=1, sort_keys=True))
    print(f"reused {len(known)} saved decisions, asked the model about {len(decisions) - len(known)} groups")
    corrections_path = settings.CONFIG_DIR / "curation" / f"{args.round}_corrections.yaml"
    corrections = yaml.safe_load(corrections_path.read_text()) if corrections_path.exists() else {}
    if corrections:
        print(f"applying {len(corrections.get('corrections', []))} reviewer corrections and "
              f"{len(corrections.get('add_nodes', []))} added nodes")  # fmt: skip
    draft = curation.build_draft(
        groups,
        suggestions,
        curation.apply_corrections(decisions, corrections),
        ont,
        corrections.get("add_nodes"),
    )
    out = settings.CONFIG_DIR / "curation" / f"{args.round}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(draft, sort_keys=False, allow_unicode=True, width=110)
    out.write_text(curation.DRAFT_HEADER.format(round=args.round) + "\n" + body)
    counts = {k: len(v) for k, v in draft.items()}
    approved = sum(e.get("approve", False) for v in draft.values() for e in v)
    print(f"LLM cost: ${cost:.4f}; draft: {counts}; pre-approved (confidence >= 0.8): {approved}")
    print(f"review {out.relative_to(settings.ROOT)}, then run: ripplekg curate apply {args.round}")
    return 0


def cmd_gold(args: argparse.Namespace) -> int:
    import pandas as pd

    from ripplekg.extraction import gold

    folder = settings.RESULTS_DIR / "gold"
    if args.action == "sheet":
        pilot = settings.RESULTS_DIR / "pilot"
        edges = {d.name: pd.read_parquet(d / "edges.parquet") for d in sorted(pilot.iterdir()) if d.is_dir()}
        chunks = pd.read_parquet(settings.DATA_DIR / "chunks" / "chunks.parquet")
        sheet, key = gold.build_sheet(edges, chunks, args.per_model, args.seed)
        sheet_path, _ = gold.write_sheet(sheet, key, folder)
        print(f"{len(sheet)} edges from {', '.join(edges)} -> {sheet_path.relative_to(settings.ROOT)}")
        print("Fill the `label` column with Y, N or ?, then run: ripplekg gold score")
    else:
        labels = pd.read_csv(folder / "label_sheet.csv")
        print(gold.score(labels, pd.read_csv(folder / "key.csv")).to_string(index=False))
    return 0


STAGES = {"annual": {"10-K", "20-F", "40-F"}, "10-q": {"10-Q"}, "8-k": {"8-K"}}


def cmd_extract(args: argparse.Namespace) -> int:
    import pandas as pd

    from ripplekg.extraction import batch
    from ripplekg.extraction.llm import ClaudeProvider, ResponseCache
    from ripplekg.extraction.runner import run_extraction
    from ripplekg.ontology.schema import load_ontology

    chunks = pd.read_parquet(settings.DATA_DIR / "chunks" / "chunks.parquet")
    stage = chunks[chunks["form"].isin(STAGES[args.stage]) & (chunks["families"].map(len) > 0)]
    provider = ClaudeProvider(model=args.model, effort=args.effort)
    cache = ResponseCache(settings.DATA_DIR / "extractions" / "cache")
    state = settings.DATA_DIR / "extractions" / "batches.json"
    ont, universe = load_ontology(), load_universe()

    if args.action == "submit":
        jobs, n_cached = batch.prepare_jobs(stage, provider, cache, ont, universe)
        print(f"{len(stage):,} chunks: {n_cached:,} calls already cached, {len(jobs):,} to submit")
        batch.submit(provider.client, jobs, state)
    elif args.action == "status":
        for row in batch.status(provider.client, state):
            print(row)
    elif args.action == "collect":
        totals = batch.collect(provider.client, cache, state)
        print(
            f"collected {totals['succeeded']:,} results (${totals['cost_usd']:.2f}), "
            f"{totals['failed']} failed, {totals['batches_waiting']} batches still running"
        )
    elif args.action == "run":
        jobs, _ = batch.prepare_jobs(stage, provider, cache, ont, universe)
        if jobs and not args.allow_live:
            print(
                f"{len(jobs):,} calls are not cached yet: submit and collect a batch first, "
                "or pass --allow-live to pay full price for them now",
                file=sys.stderr,
            )
            return 1
        res = run_extraction(stage, provider, cache, ont, universe, progress=lambda m: print(m, flush=True))
        out = settings.DATA_DIR / "extractions" / args.stage
        out.mkdir(parents=True, exist_ok=True)
        res.outcomes.to_parquet(out / "outcomes.parquet", index=False)
        res.edges.to_parquet(out / "edges.parquet", index=False)
        e = res.edges
        print(f"{len(res.outcomes):,} calls, {len(e):,} edges, {int(e['ok'].sum()):,} verified")
        print(f"wrote {out.relative_to(settings.ROOT)}/edges.parquet")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ripplekg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def window(p: argparse.ArgumentParser) -> None:
        p.add_argument("--start", default=settings.CORPUS_START, help="first filing date (YYYY-MM-DD)")
        p.add_argument("--end", default=settings.CORPUS_END, help="last filing date (YYYY-MM-DD)")

    p = sub.add_parser("universe-check", help="validate universe CIKs against EDGAR and count filings")
    window(p)
    p.set_defaults(func=cmd_universe_check)

    p = sub.add_parser("coverage", help="week-1 go/no-go: co-mention density via EDGAR full-text search")
    window(p)
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("events-check", help="validate config/events.yaml and the pre-registered split")
    p.set_defaults(func=cmd_events_check)

    p = sub.add_parser("ontology-doc", help="validate relation examples and regenerate docs/ontology.md")
    p.set_defaults(func=cmd_ontology_doc)

    p = sub.add_parser("index", help="index the universe's filings and select the extraction corpus")
    window(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("fetch", help="download selected filings (resumable; skips files already on disk)")
    p.add_argument("--limit", type=int, help="only the first N selected filings, by date")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("parse", help="raw filings -> Documents with relationship-bearing sections")
    p.add_argument("--limit", type=int, help="only the first N selected filings, by date")
    p.add_argument("--force", action="store_true", help="rebuild documents even if unchanged")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("chunk", help="deduplicate, chunk and pre-filter documents; estimate LLM cost")
    p.set_defaults(func=cmd_chunk)

    p = sub.add_parser(
        "pilot", help="extract a small sample of annual-report chunks and report cost and quality"
    )
    p.add_argument("--sample", type=int, default=50, help="number of chunks")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--provider", default="claude", choices=["claude", "ollama"])
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("extract", help="bulk extraction through the Batch API: submit, status, collect, run")
    p.add_argument("action", choices=["submit", "status", "collect", "run"])
    p.add_argument("--stage", default="annual", choices=sorted(STAGES))
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--allow-live", action="store_true", help="run: pay full price for calls not yet cached")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("gold", help="blind labelling sheet for pilot edges, and precision per model")
    p.add_argument("action", choices=["sheet", "score"])
    p.add_argument("--per-model", type=int, default=60)
    p.add_argument("--seed", type=int, default=11)
    p.set_defaults(func=cmd_gold)

    p = sub.add_parser("curate", help="draft node curation from unresolved names (review before applying)")
    p.add_argument("action", choices=["draft"])
    p.add_argument("--round", default="round1")
    p.add_argument("--run", default="v1-annual")
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--retriage", action="store_true", help="ignore saved decisions and ask the model again")
    p.set_defaults(func=cmd_curate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except settings.SettingsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
