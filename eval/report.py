"""
Eval report loader and comparison tooling.

Loads eval result JSONs from eval/results/, prints comparison tables
with deltas between runs, and optionally exports Plotly charts.

Usage:
    python -m eval.report                          # show latest report
    python -m eval.report --all                    # compare all reports
    python -m eval.report --latest 3               # compare 3 most recent
    python -m eval.report --baseline baseline       # deltas against tag "baseline"
    python -m eval.report --tag bge-rerank         # filter to a specific tag
    python -m eval.report --plot                   # export Plotly bar charts
    python -m eval.report --by-topic               # include per-topic breakdown
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table


RESULTS_DIR = Path("eval/results")

# ── Aggregate metric keys (in display order) ──
# These map to keys in report["aggregate"]
RETRIEVAL_METRICS = [
    ("mean_precision_at_k", "P@K"),
    ("mean_recall_at_k", "R@K"),
    ("mean_mrr", "MRR"),
]
GENERATION_METRICS = [
    ("mean_bleu", "BLEU"),
    ("mean_rouge_l_f1", "ROUGE-L"),
    ("mean_token_f1", "Token F1"),
    ("mean_source_coverage", "Src Cov"),
]
ALL_METRICS = RETRIEVAL_METRICS + GENERATION_METRICS


# ── Loading ──

def load_report(path: Path) -> dict:
    """Load a single eval report JSON and attach the filepath."""
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    report["_filepath"] = str(path)
    return report


def load_all_reports(results_dir: Path = None) -> list[dict]:
    """
    Discover and load all eval_*.json reports from the results directory.
    Returns reports sorted by timestamp (oldest first).
    """
    results_dir = results_dir or RESULTS_DIR
    if not results_dir.exists():
        return []

    reports = []
    for path in sorted(results_dir.glob("eval_*.json")):
        try:
            reports.append(load_report(path))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [WARNING] Skipping {path.name}: {e}")

    return reports


def filter_reports(
    reports: list[dict],
    tag: str = None,
    latest: int = None,
) -> list[dict]:
    """Filter reports by tag and/or limit to N most recent."""
    if tag:
        reports = [r for r in reports if r.get("meta", {}).get("tag") == tag]
    if latest and latest > 0:
        reports = reports[-latest:]
    return reports


# ── Report label ──

def report_label(report: dict) -> str:
    """
    Build a short human-readable label for a report.
    Prefers the tag if present, otherwise uses timestamp.
    """
    meta = report.get("meta", {})
    tag = meta.get("tag")
    ts_raw = meta.get("timestamp", "")

    # Parse ISO timestamp into compact form
    ts_short = ""
    if ts_raw:
        try:
            dt = datetime.fromisoformat(ts_raw)
            ts_short = dt.strftime("%m/%d %H:%M")
        except ValueError:
            ts_short = ts_raw[:16]

    if tag:
        return f"{tag} ({ts_short})" if ts_short else tag
    return ts_short or Path(report.get("_filepath", "unknown")).stem


# ── Delta formatting ──

def _format_value(value: float, fmt: str = ".4f") -> str:
    """Format a metric value."""
    return f"{value:{fmt}}"


def _format_delta(current: float, baseline: float) -> str:
    """
    Format a delta value with direction indicator.
    Green ▲ for improvement, red ▼ for regression.
    """
    delta = current - baseline
    if abs(delta) < 0.0005:
        return "  ─"
    sign = "▲" if delta > 0 else "▼"
    color = "green" if delta > 0 else "red"
    return f"[{color}]{sign}{abs(delta):+.4f}"[:-1] + f"[/{color}]"
    # Simpler version:


def format_delta(current: float, baseline: float) -> str:
    """
    Format a delta with direction arrow + color for rich output.
    ▲ green = improvement, ▼ red = regression, ─ = no change.
    """
    delta = current - baseline
    if abs(delta) < 0.0005:
        return "[dim]─[/dim]"
    if delta > 0:
        return f"[green]▲{delta:.4f}[/green]"
    return f"[red]▼{delta:.4f}[/red]"


# ── Comparison table ──

def print_comparison_table(
    reports: list[dict],
    baseline: dict = None,
    console: Console = None,
) -> None:
    """
    Print a rich comparison table of aggregate metrics across runs.

    If baseline is provided, each non-baseline column shows deltas.
    """
    console = console or Console()

    if not reports:
        console.print("[yellow]No reports found.[/yellow]")
        return

    # Decide which metrics to show based on run modes
    has_generation = any(
        r.get("meta", {}).get("mode") != "retrieval-only"
        for r in reports
    )
    metrics = ALL_METRICS if has_generation else RETRIEVAL_METRICS

    # ── Build table ──
    table = Table(
        title="Eval Comparison",
        show_lines=True,
        title_style="bold",
        pad_edge=True,
    )

    table.add_column("Metric", style="bold", min_width=10)

    # One column per report
    for report in reports:
        label = report_label(report)
        is_baseline = (baseline is not None and report is baseline)
        style = "bold cyan" if is_baseline else ""
        header = f"{label}\n(baseline)" if is_baseline else label
        table.add_column(header, justify="right", style=style, min_width=12)

    # ── Rows ──
    for key, display_name in metrics:
        row = [display_name]
        for report in reports:
            value = report.get("aggregate", {}).get(key, 0.0)
            cell = _format_value(value)

            # Append delta if we have a baseline and this isn't the baseline
            if baseline is not None and report is not baseline:
                baseline_val = baseline.get("aggregate", {}).get(key, 0.0)
                cell += f"  {format_delta(value, baseline_val)}"

            row.append(cell)
        table.add_row(*row)

    # ── Extra info rows ──
    table.add_section()

    # Abstention rate
    row = ["Abstain %"]
    for report in reports:
        rate = report.get("aggregate", {}).get("abstention_rate", 0.0)
        row.append(f"{rate:.1%}")
    table.add_row(*row)

    # Questions
    row = ["Questions"]
    for report in reports:
        n = report.get("aggregate", {}).get("total_questions", "?")
        row.append(str(n))
    table.add_row(*row)

    # Time
    row = ["Time (s)"]
    for report in reports:
        t = report.get("meta", {}).get("total_elapsed_seconds", "?")
        row.append(str(t))
    table.add_row(*row)

    console.print()
    console.print(table)
    console.print()


def print_topic_breakdown(
    reports: list[dict],
    console: Console = None,
) -> None:
    """Print per-topic metrics for each report."""
    console = console or Console()

    for report in reports:
        label = report_label(report)
        by_topic = report.get("aggregate", {}).get("by_topic", {})

        if not by_topic:
            continue

        table = Table(
            title=f"Per-Topic: {label}",
            show_lines=True,
            title_style="bold",
        )
        table.add_column("Topic", style="bold", min_width=14)
        table.add_column("P@K", justify="right")
        table.add_column("R@K", justify="right")
        table.add_column("Token F1", justify="right")
        table.add_column("n", justify="right")

        for topic, stats in sorted(by_topic.items()):
            table.add_row(
                topic,
                f"{stats.get('precision_at_k', 0):.3f}",
                f"{stats.get('recall_at_k', 0):.3f}",
                f"{stats.get('token_f1', 0):.3f}",
                str(stats.get("count", "?")),
            )

        console.print(table)
        console.print()


# ── Config diff ──

def print_config_diff(
    reports: list[dict],
    console: Console = None,
) -> None:
    """
    Show only the config keys that differ between reports.
    Useful for identifying what changed between runs.
    """
    console = console or Console()

    if len(reports) < 2:
        return

    configs = [r.get("config", {}) for r in reports]
    all_keys = sorted(set().union(*(c.keys() for c in configs)))

    # Find keys that differ
    diff_keys = []
    for key in all_keys:
        values = [c.get(key) for c in configs]
        if len(set(str(v) for v in values)) > 1:
            diff_keys.append(key)

    if not diff_keys:
        console.print("[dim]All configs identical.[/dim]")
        return

    table = Table(title="Config Differences", show_lines=True, title_style="bold")
    table.add_column("Setting", style="bold")
    for report in reports:
        table.add_column(report_label(report), justify="right")

    for key in diff_keys:
        row = [key]
        for report in reports:
            val = report.get("config", {}).get(key, "—")
            row.append(str(val))
        table.add_row(*row)

    console.print(table)
    console.print()


# ── Plotly charts (optional) ──

def export_plotly_charts(
    reports: list[dict],
    output_dir: Path = None,
) -> list[Path]:
    """
    Generate bar charts comparing metrics across runs.
    Returns paths to saved HTML files.

    Requires plotly — skips gracefully if not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARNING] plotly not installed — skipping chart export.")
        print("  Install with: pip install plotly")
        return []

    output_dir = output_dir or RESULTS_DIR / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [report_label(r) for r in reports]

    has_generation = any(
        r.get("meta", {}).get("mode") != "retrieval-only"
        for r in reports
    )
    metrics = ALL_METRICS if has_generation else RETRIEVAL_METRICS

    saved = []

    # ── Single grouped bar chart ──
    fig = go.Figure()

    for key, display_name in metrics:
        values = [r.get("aggregate", {}).get(key, 0.0) for r in reports]
        fig.add_trace(go.Bar(name=display_name, x=labels, y=values))

    fig.update_layout(
        title="RagBench Eval Comparison",
        barmode="group",
        yaxis_title="Score",
        xaxis_title="Run",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    chart_path = output_dir / "comparison.html"
    fig.write_html(str(chart_path))
    saved.append(chart_path)
    print(f"  Chart saved: {chart_path}")

    return saved


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="RagBench Eval Report — compare evaluation runs",
    )
    parser.add_argument(
        "--dir", type=str, default=str(RESULTS_DIR),
        help="Path to eval results directory (default: eval/results)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show all reports (default: latest only)",
    )
    parser.add_argument(
        "--latest", type=int, default=None,
        help="Show N most recent reports",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Filter to reports with this tag",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Tag of the baseline run (deltas computed against this)",
    )
    parser.add_argument(
        "--by-topic", action="store_true",
        help="Show per-topic breakdown for each report",
    )
    parser.add_argument(
        "--diff", action="store_true",
        help="Show config differences between runs",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Export Plotly comparison chart to eval/results/charts/",
    )

    args = parser.parse_args()

    console = Console()
    results_dir = Path(args.dir)

    # Load reports
    all_reports = load_all_reports(results_dir)
    if not all_reports:
        console.print(f"[yellow]No eval reports found in {results_dir}[/yellow]")
        console.print("  Run an evaluation first: python -m eval.runner")
        return

    console.print(f"[dim]Found {len(all_reports)} report(s) in {results_dir}[/dim]")

    # Filter
    if args.tag:
        reports = filter_reports(all_reports, tag=args.tag)
    elif args.latest:
        reports = filter_reports(all_reports, latest=args.latest)
    elif args.all:
        reports = all_reports
    else:
        # Default: show only the latest report
        reports = all_reports[-1:]

    if not reports:
        console.print(f"[yellow]No reports match the filter.[/yellow]")
        return

    # Find baseline
    baseline_report = None
    if args.baseline:
        candidates = [r for r in all_reports if r.get("meta", {}).get("tag") == args.baseline]
        if candidates:
            # Use the most recent report with that tag
            baseline_report = candidates[-1]
            # Ensure baseline is in the display set
            if baseline_report not in reports:
                reports.insert(0, baseline_report)
        else:
            console.print(f"[yellow]No report found with tag '{args.baseline}'[/yellow]")

    # Display
    print_comparison_table(reports, baseline=baseline_report, console=console)

    if args.by_topic:
        print_topic_breakdown(reports, console=console)

    if args.diff and len(reports) > 1:
        print_config_diff(reports, console=console)

    if args.plot and len(reports) > 1:
        export_plotly_charts(reports)


if __name__ == "__main__":
    main()