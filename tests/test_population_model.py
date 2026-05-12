from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qalbench.population import PopulationParams, run_population_benchmark  # noqa: E402


class PopulationBenchmarkTests(unittest.TestCase):
    def test_fixed_seed_population_run_is_deterministic(self) -> None:
        params = PopulationParams(seed=23, steps=12, initial_population=8, carrying_capacity=20)
        first = run_population_benchmark(params)
        second = run_population_benchmark(params)
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(first["trajectory"], second["trajectory"])

    def test_population_records_lineage_turnover_and_variation(self) -> None:
        params = PopulationParams(
            seed=31,
            steps=18,
            initial_population=10,
            carrying_capacity=28,
            mutation_probability=0.25,
        )
        result = run_population_benchmark(params)
        summary = result["summary"]
        self.assertGreater(summary["birth_count"], 0)
        self.assertGreater(summary["turnover_events"], 0)
        self.assertGreaterEqual(summary["max_lineage_depth"], 1)
        self.assertGreaterEqual(summary["mutation_event_rate"], 0.0)
        self.assertLessEqual(summary["final_population"], params.carrying_capacity)

    def test_resource_score_separates_quantum_from_dephased_population_control(self) -> None:
        quantum = run_population_benchmark(
            PopulationParams(
                model="quantum",
                scenario="resource_selection",
                seed=41,
                steps=10,
                initial_population=8,
                carrying_capacity=22,
            )
        )["summary"]
        dephased = run_population_benchmark(
            PopulationParams(
                model="dephased",
                scenario="resource_selection",
                seed=41,
                steps=10,
                initial_population=8,
                carrying_capacity=22,
            )
        )["summary"]
        self.assertGreater(quantum["mean_event_resource_score"], dephased["mean_event_resource_score"])
        self.assertEqual(dephased["mean_event_resource_score"], 0.0)

    def test_no_inheritance_control_reduces_theta_transmission(self) -> None:
        inherited = run_population_benchmark(
            PopulationParams(
                model="quantum",
                seed=57,
                steps=24,
                initial_population=12,
                carrying_capacity=34,
                mutation_probability=0.02,
            )
        )["summary"]
        no_inheritance = run_population_benchmark(
            PopulationParams(
                model="no_inheritance",
                seed=57,
                steps=24,
                initial_population=12,
                carrying_capacity=34,
                mutation_probability=0.02,
            )
        )["summary"]
        self.assertGreater(
            inherited["theta_parent_child_correlation"],
            no_inheritance["theta_parent_child_correlation"],
        )

    def test_no_inheritance_does_not_label_independent_draws_as_parent_mutations(self) -> None:
        result = run_population_benchmark(
            PopulationParams(
                model="no_inheritance",
                seed=61,
                steps=16,
                initial_population=10,
                carrying_capacity=26,
                mutation_probability=1.0,
            )
        )
        self.assertEqual(result["summary"]["mutation_event_rate"], 0.0)
        self.assertTrue(all(not event.mutation for event in result["birth_events"]))

    def test_attempt_records_include_failed_births_and_interaction_context(self) -> None:
        result = run_population_benchmark(
            PopulationParams(
                seed=67,
                steps=8,
                initial_population=8,
                carrying_capacity=18,
                base_birth_probability=0.05,
            )
        )
        attempts = result["attempts"]
        self.assertGreater(len(attempts), len(result["birth_events"]))
        self.assertTrue(any(not attempt.born for attempt in attempts))
        self.assertTrue(any(attempt.partner_id is not None for attempt in attempts))
        self.assertTrue(all(not attempt.capacity_blocked for attempt in attempts if attempt.born))

    def test_neutral_scenario_removes_interaction_selection(self) -> None:
        result = run_population_benchmark(
            PopulationParams(
                scenario="neutral",
                seed=71,
                steps=10,
                initial_population=8,
                carrying_capacity=18,
                interaction_strength=0.95,
            )
        )
        self.assertTrue(
            all(abs(attempt.birth_probability - 0.28) <= 1e-12 for attempt in result["attempts"])
        )

    def test_capacity_blocked_parent_opportunities_are_recorded(self) -> None:
        result = run_population_benchmark(
            PopulationParams(
                seed=73,
                steps=8,
                initial_population=10,
                carrying_capacity=10,
                base_death_probability=0.0,
                density_death_probability=0.0,
            )
        )
        summary = result["summary"]
        blocked = [attempt for attempt in result["attempts"] if attempt.capacity_blocked]
        self.assertGreater(len(blocked), 0)
        self.assertEqual(summary["capacity_blocked_opportunity_count"], len(blocked))
        self.assertEqual(summary["evaluated_birth_attempt_count"], 0)


if __name__ == "__main__":
    unittest.main()
