"""Command-line entry point: `ripplekg <command>`."""

from __future__ import annotations

import argparse
import sys

from ripplekg import settings
from ripplekg.edgar.client import EdgarClient
from ripplekg.universe import load_universe


def _client() -> EdgarClient:
    return EdgarClient(settings.sec_user_agent(), cache_dir=settings.HTTP_CACHE_DIR)


def cmd_universe_check(args: argparse.Namespace) -> int:
    from ripplekg.coverage import check_filers

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
    from ripplekg import coverage

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
    from ripplekg.events import assign_split, load_events

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
    from ripplekg.ontology import example_errors, load_examples, load_ontology
    from ripplekg.ontology_doc import render

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
    from ripplekg import corpus

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
