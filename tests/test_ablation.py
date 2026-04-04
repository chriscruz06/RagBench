"""
Tests for the ablation experiment runner.

These test the harness logic using --dry-run mode, which doesn't
require ChromaDB, the corpus, or Ollama.

Usage:
    python -m pytest tests/test_ablation.py -v
"""

import pytest
from experiments.ablation import run_ablation, ALL_STRATEGIES


class TestAblationDryRun:
    """Test ablation harness in dry-run mode (no pipeline calls)."""

    def test_dry_run_all_strategies(self):
        """Dry run produces one report per strategy."""
        reports = run_ablation(dry_run=True)
        assert len(reports) == len(ALL_STRATEGIES)

    def test_dry_run_tags_match_strategies(self):
        """Each report is tagged with chunk_<strategy>."""
        reports = run_ablation(dry_run=True)
        tags = [r["meta"]["tag"] for r in reports]
        expected = [f"chunk_{s}" for s in ALL_STRATEGIES]
        assert tags == expected

    def test_dry_run_subset(self):
        """Can run a subset of strategies."""
        reports = run_ablation(strategies=["fixed", "sentence"], dry_run=True)
        assert len(reports) == 2
        assert reports[0]["meta"]["tag"] == "chunk_fixed"
        assert reports[1]["meta"]["tag"] == "chunk_sentence"

    def test_dry_run_single_strategy(self):
        """Can run a single strategy."""
        reports = run_ablation(strategies=["semantic"], dry_run=True)
        assert len(reports) == 1
        assert reports[0]["meta"]["tag"] == "chunk_semantic"

    def test_report_has_aggregate(self):
        """Each report contains aggregate scores."""
        reports = run_ablation(strategies=["fixed"], dry_run=True)
        agg = reports[0]["aggregate"]
        assert "mean_precision_at_k" in agg
        assert "mean_recall_at_k" in agg
        assert "mean_mrr" in agg

    def test_report_config_reflects_strategy(self):
        """Config snapshot should show the strategy that was actually run."""
        reports = run_ablation(strategies=["sentence"], dry_run=True)
        assert reports[0]["config"]["chunk_strategy"] == "sentence"

    def test_dry_run_creates_result_files(self, tmp_path, monkeypatch):
        """Reports get written to eval/results/."""
        import experiments.ablation as ablation_mod
        from eval import runner as runner_mod

        # Redirect results to tmp_path
        monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)

        run_ablation(strategies=["fixed"], dry_run=True)

        result_files = list(tmp_path.glob("eval_*_chunk_fixed*.json"))
        assert len(result_files) == 1

# Christ, have mercy on us.
# Lord, have mercy on us.

# Pray for us, Saint Albert,
# That we may be made worthy of the promises of Christ.

# Let Us Pray.

# O God, Who richly adorned Saint Albert
# with Thy heavenly gifts
# and decorated him with all virtues,
# grant us, Thy servants,
# that we may follow in his footsteps,
# may persevere in Thy service until death
# and may securely obtain an everlasting reward,
# through Jesus Christ,
# Thy Son Our Lord.

# Amen.
