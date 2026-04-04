"""
Lightweight test for eval/report.py — mocks rich so we can verify
core logic (loading, filtering, deltas) without the dependency.
"""

import json
import sys
import types
import tempfile
from pathlib import Path

# ── Mock rich before importing report ──
mock_rich = types.ModuleType("rich")
mock_console_mod = types.ModuleType("rich.console")
mock_table_mod = types.ModuleType("rich.table")


class MockConsole:
    def print(self, *args, **kwargs):
        # Strip rich markup for plain output
        text = " ".join(str(a) for a in args)
        import re
        text = re.sub(r'\[/?[^\]]*\]', '', text)
        print(text)


class MockTable:
    def __init__(self, **kwargs):
        self.title = kwargs.get("title", "")
        self.columns = []
        self.rows = []

    def add_column(self, name, **kwargs):
        self.columns.append(name)

    def add_row(self, *args):
        self.rows.append(args)

    def add_section(self):
        pass

    def __str__(self):
        return f"<Table '{self.title}' cols={len(self.columns)} rows={len(self.rows)}>"


mock_console_mod.Console = MockConsole
mock_table_mod.Table = MockTable
mock_rich.console = mock_console_mod
mock_rich.table = mock_table_mod

sys.modules["rich"] = mock_rich
sys.modules["rich.console"] = mock_console_mod
sys.modules["rich.table"] = mock_table_mod

# Now we can import
sys.path.insert(0, str(Path(__file__).parent))
from eval.report import (
    load_report,
    load_all_reports,
    filter_reports,
    report_label,
    format_delta,
    print_comparison_table,
    print_topic_breakdown,
    print_config_diff,
)


# ── Mock data ──

BASELINE = {
    "meta": {"timestamp": "2025-06-01T14:30:00", "tag": "baseline", "mode": "full",
             "dry_run": False, "total_elapsed_seconds": 42.5, "test_set_size": 10},
    "config": {"embedding_model": "BAAI/bge-base-en-v1.5", "top_k": 5,
               "chunk_strategy": "sentence", "chunk_size": 512},
    "aggregate": {
        "total_questions": 10, "abstention_rate": 0.0,
        "mean_precision_at_k": 0.140, "mean_recall_at_k": 0.150,
        "mean_mrr": 0.270, "mean_bleu": 0.035, "mean_rouge_l_f1": 0.190,
        "mean_token_f1": 0.261, "mean_source_coverage": 0.117,
        "by_topic": {
            "sacraments": {"precision_at_k": 0.133, "recall_at_k": 0.167, "token_f1": 0.280, "count": 3},
            "creed": {"precision_at_k": 0.133, "recall_at_k": 0.111, "token_f1": 0.250, "count": 3},
        },
    },
    "per_question": [],
}

IMPROVED = {
    "meta": {"timestamp": "2025-06-03T10:00:00", "tag": "bge-rerank", "mode": "full",
             "dry_run": False, "total_elapsed_seconds": 55.2, "test_set_size": 10},
    "config": {"embedding_model": "BAAI/bge-base-en-v1.5", "top_k": 10,
               "chunk_strategy": "sentence", "chunk_size": 512},
    "aggregate": {
        "total_questions": 10, "abstention_rate": 0.0,
        "mean_precision_at_k": 0.200, "mean_recall_at_k": 0.250,
        "mean_mrr": 0.350, "mean_bleu": 0.048, "mean_rouge_l_f1": 0.220,
        "mean_token_f1": 0.310, "mean_source_coverage": 0.180,
        "by_topic": {
            "sacraments": {"precision_at_k": 0.200, "recall_at_k": 0.250, "token_f1": 0.350, "count": 3},
            "creed": {"precision_at_k": 0.180, "recall_at_k": 0.200, "token_f1": 0.290, "count": 3},
        },
    },
    "per_question": [],
}


# ── Tests ──

def test_format_delta():
    # Improvement
    r = format_delta(0.200, 0.140)
    assert "▲" in r, f"Expected ▲ for improvement, got: {r}"
    # Regression
    r = format_delta(0.100, 0.140)
    assert "▼" in r, f"Expected ▼ for regression, got: {r}"
    # No change
    r = format_delta(0.140, 0.140)
    assert "─" in r, f"Expected ─ for no change, got: {r}"
    print("  format_delta")


def test_report_label():
    assert "baseline" in report_label(BASELINE)
    no_tag = {"meta": {"timestamp": "2025-06-01T14:30:00", "tag": None}}
    label = report_label(no_tag)
    assert "06/01" in label, f"Expected date, got: {label}"
    print("  report_label")


def test_load_and_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i, report in enumerate([BASELINE, IMPROVED]):
            path = tmpdir / f"eval_2025060{i+1}_000000.json"
            with open(path, "w") as f:
                json.dump(report, f)

        reports = load_all_reports(tmpdir)
        assert len(reports) == 2, f"Expected 2, got {len(reports)}"

        # _filepath should be attached
        assert "_filepath" in reports[0]

        # Filter by tag
        assert len(filter_reports(reports, tag="baseline")) == 1
        assert len(filter_reports(reports, tag="bge-rerank")) == 1
        assert len(filter_reports(reports, tag="nonexistent")) == 0

        # Filter latest
        assert len(filter_reports(reports, latest=1)) == 1

    print(" load_all_reports + filter_reports")


def test_comparison_table():
    console = MockConsole()
    # Should not raise
    print_comparison_table([BASELINE], console=console)
    print_comparison_table([BASELINE, IMPROVED], baseline=BASELINE, console=console)
    print("  print_comparison_table (no errors)")


def test_topic_breakdown():
    console = MockConsole()
    print_topic_breakdown([BASELINE], console=console)
    print(" print_topic_breakdown")


def test_config_diff():
    console = MockConsole()
    print_config_diff([BASELINE, IMPROVED], console=console)
    print(" print_config_diff (should detect top_k diff)")


if __name__ == "__main__":
    print("\n[eval/report.py smoke test]\n")
    test_format_delta()
    test_report_label()
    test_load_and_filter()
    test_comparison_table()
    test_topic_breakdown()
    test_config_diff()
    print("\nAll tests passed\n")