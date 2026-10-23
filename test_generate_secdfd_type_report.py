import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import mock_open, patch

from combine_secdfd_type_reports import combine_reports, load_report
from generate_secdfd_type_report import (
    CONCLUSIONS,
    DIMENSION_OUTCOMES,
    VALID_SECDFD_TYPES,
    additional_groundtruth_entries,
    assignment_outcome,
    build_groundtruth_index,
    build_secdfd_target_index,
    checked_node_report,
    evaluate_outcomes,
    generate_report,
    get_sabo_edges,
    groundtruth_availability,
    normalize_secdfd_type,
    summarize_node_outcomes,
    summarize_invalid_secdfd_types,
    summarize_outcomes,
    validate_evaluation_counts,
)


def node(node_id, types=None, primary=None, label="Operation"):
    return {"data": {"id": node_id, "labels": [label], "properties": {
        "secdfdTypes": types, "primarySecdfdType": primary,
    }}}


def mapping(node_id, label, element="example"):
    return {"sabo": node_id, "secdfd_type": label, "secdfd": element}


class SecdfdTypeReportTests(unittest.TestCase):
    def test_normalization_accepts_only_specified_aliases_and_case_variants(self):
        aliases = {
            "Asset": "Asset", "ASSET": "Asset", "asset": "Asset",
            "DataStore": "DataStore", "DATA STORE": "DataStore",
            "data_store": "DataStore", "Process": "Process", "FLOW": "Flow",
            "External Entity": "External Entity", "externalentity": "External Entity",
            "EXTERNAL_ENTITY": "External Entity", "undetermined": "Undetermined",
        }
        for label, expected in aliases.items():
            with self.subTest(label=label):
                self.assertEqual(normalize_secdfd_type(label), expected)
        for label in (None, 42, [], {}, "", " ", "Process ", "data-store", "Unknown"):
            with self.subTest(invalid=label):
                self.assertIsNone(normalize_secdfd_type(label))

    def test_assignment_undefined_cases(self):
        for primary in (None, ""):
            for types in (None, []):
                with self.subTest(primary=primary, types=types):
                    self.assertEqual(assignment_outcome(primary, types), "UNDEFINED")

    def test_assignment_consistency_requires_nonempty_string_and_exact_membership(self):
        cases = (
            ("Process", ["Asset", "Process"]),
            ("Unknown", ["Unknown"]),
            (" ", [" "]),
        )
        for primary, types in cases:
            with self.subTest(primary=primary):
                self.assertEqual(assignment_outcome(primary, types), "DEFINED_AND_CONSISTENT")

    def test_assignment_inconsistent_cases_include_wrong_field_types(self):
        cases = (
            (None, ["Process"]), ("", ["Process"]), ("Process", None),
            ("Process", []), ("Process", "Process"), ("Process", {}),
            (42, [42]), (False, [False]), ([], None), (None, {}), (None, ""),
            (None, 0), (" ", []), ("process", ["Process"]),
            ("Process", ["Process "]), ("DataStore", ["Data Store"]),
        )
        for primary, types in cases:
            with self.subTest(primary=primary, types=types):
                self.assertEqual(assignment_outcome(primary, types), "INCONSISTENT")

    def test_canonical_labels_are_valid_without_edges(self):
        for label in VALID_SECDFD_TYPES:
            with self.subTest(label=label):
                rows = checked_node_report([node("n", [label], label)], {}, {}, {})
                self.assertEqual(rows[0]["outcomes"]["validity"], "VALID")
                self.assertFalse(rows[0]["hasImplementsEdge"])

    def test_recognized_noncanonical_labels_are_almost_valid(self):
        for label in ("process", "ASSET", "Data Store", "Data_Store", "ExternalEntity"):
            with self.subTest(label=label):
                outcomes = evaluate_outcomes("DEFINED_AND_CONSISTENT", label, set(), set())
                self.assertEqual(outcomes["validity"], "ALMOST_VALID")
                self.assertEqual(outcomes["determination"], "DETERMINED")
                self.assertEqual(outcomes["mappingCorrectness"], "UNASSESSED")

    def test_invalid_values_have_no_determination_or_correctness(self):
        for label in ("Unknown", "data-store", "Process ", 42, None, {}, []):
            with self.subTest(label=label):
                outcomes = evaluate_outcomes("DEFINED_AND_CONSISTENT", label, {"Process"}, set())
                self.assertEqual(outcomes["validity"], "INVALID")
                self.assertIsNone(outcomes["determination"])
                self.assertIsNone(outcomes["mappingCorrectness"])

    def test_undefined_and_inconsistent_gate_all_type_dimensions(self):
        for assignment in ("UNDEFINED", "INCONSISTENT"):
            outcomes = evaluate_outcomes(assignment, "Process", {"Process"}, set())
            for dimension in ("validity", "determination", "mappingCorrectness"):
                self.assertIsNone(outcomes[dimension])
            self.assertEqual(outcomes["groundTruthAvailability"], "ORIGINAL_GROUND_TRUTH_AVAILABLE")

    def test_undetermined_including_almost_valid_variant_has_no_correctness(self):
        for label, validity in (("Undetermined", "VALID"), ("UNDETERMINED", "ALMOST_VALID")):
            outcomes = evaluate_outcomes("DEFINED_AND_CONSISTENT", label, {"Process"}, set())
            self.assertEqual(outcomes["validity"], validity)
            self.assertEqual(outcomes["determination"], "UNDETERMINED")
            self.assertIsNone(outcomes["mappingCorrectness"])

    def test_groundtruth_availability_original_takes_precedence(self):
        cases = (
            ({"Process"}, set(), "ORIGINAL_GROUND_TRUTH_AVAILABLE"),
            ({"Process"}, {"Asset"}, "ORIGINAL_GROUND_TRUTH_AVAILABLE"),
            (set(), {"Asset"}, "EXTENDED_GROUND_TRUTH_ONLY"),
            (set(), set(), "GROUND_TRUTH_UNAVAILABLE"),
        )
        for original, extended, expected in cases:
            self.assertEqual(groundtruth_availability(original, extended), expected)

    def test_correctness_checks_original_extended_union_and_unavailable(self):
        cases = (
            ({"Process"}, {"Process"}, "CORRECT"),
            ({"Process"}, {"Asset"}, "CORRECT"),
            ({"Asset"}, {"Process"}, "PROBABLY_CORRECT"),
            (set(), {"Process"}, "PROBABLY_CORRECT"),
            ({"Asset"}, {"Flow"}, "INCORRECT"),
            (set(), {"Asset"}, "INCORRECT"),
            (set(), set(), "UNASSESSED"),
        )
        for original, extended, expected in cases:
            with self.subTest(original=original, extended=extended):
                outcomes = evaluate_outcomes("DEFINED_AND_CONSISTENT", "process", original, extended)
                self.assertEqual(outcomes["mappingCorrectness"], expected)

    def test_each_label_is_evaluated_independently_with_raw_values_preserved(self):
        labels = ["Process", "Data Store", "Unknown", "Undetermined", None]
        rows = checked_node_report(
            [node("n", labels, "Process")],
            build_groundtruth_index([mapping("n", "process")]), {}, {},
        )
        self.assertEqual([row["secdfdType"] for row in rows], labels)
        self.assertEqual([row["secdfdTypeIndex"] for row in rows], list(range(5)))
        self.assertEqual(rows[0]["outcomes"]["mappingCorrectness"], "CORRECT")
        self.assertEqual(rows[1]["outcomes"]["mappingCorrectness"], "INCORRECT")
        self.assertEqual(rows[1]["almostValidSecdfdType"], "Data Store")
        self.assertEqual(rows[2]["invalidSecdfdType"], "Unknown")
        self.assertIsNone(rows[4]["invalidSecdfdType"])
        for row in rows:
            self.assertEqual(row["conclusion"], " and ".join(row["conclusions"]))
            self.assertEqual(row["conclusions"], [
                value for value in row["outcomes"].values() if value is not None
            ])

    def test_inconsistent_node_keeps_per_item_rows_but_does_not_assess_labels(self):
        rows = checked_node_report([node("n", ["Process", "Unknown"], "Asset")], {}, {}, {})
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["outcomes"]["assignmentAndConsistency"], "INCONSISTENT")
            self.assertIsNone(row["outcomes"]["validity"])
            self.assertNotIn("invalidSecdfdType", row)

    def test_missing_primary_with_populated_list_is_inconsistent(self):
        rows = checked_node_report([node("n", ["Process"])], {}, {}, {})
        self.assertEqual(rows[0]["outcomes"]["assignmentAndConsistency"], "INCONSISTENT")
        self.assertIsNone(rows[0]["outcomes"]["mappingCorrectness"])

    def test_nonlist_field_values_produce_one_diagnostic_entry(self):
        for types in ("Process", 1, False, {"type": "Process"}):
            rows = checked_node_report([node("n", types, "Process")], {}, {}, {})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["secdfdTypes"], types)
            self.assertEqual(rows[0]["outcomes"]["assignmentAndConsistency"], "INCONSISTENT")

    def test_duplicate_labels_still_produce_distinct_entry_indices(self):
        rows = checked_node_report([node("n", ["Process", "Process"], "Process")], {}, {}, {})
        self.assertEqual([row["secdfdTypeIndex"] for row in rows], [0, 1])

    def test_groundtruth_normalization_and_addition_deduplication(self):
        original = [mapping("n", "data_store")]
        copied = {**mapping("n", "DataStore"), "note": "copied"}
        added = mapping("n", "asset", "second")
        additions = additional_groundtruth_entries(original, [copied, added, added])
        self.assertEqual(additions, [added])
        rows = checked_node_report(
            [node("n", ["DataStore", "Asset"], "DataStore")],
            build_groundtruth_index(original), build_groundtruth_index(additions), {},
        )
        self.assertEqual(rows[0]["outcomes"]["mappingCorrectness"], "CORRECT")
        self.assertEqual(rows[1]["outcomes"]["mappingCorrectness"], "PROBABLY_CORRECT")
        self.assertEqual(rows[1]["outcomes"]["groundTruthAvailability"], "ORIGINAL_GROUND_TRUTH_AVAILABLE")

    def test_malformed_groundtruth_type_is_reported_as_input_error(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized ground-truth"):
            build_groundtruth_index([mapping("n", "Unknown")])

    def test_node_filter_and_direct_data_properties(self):
        nodes = [
            {"data": {"id": "n", "labels": ["operation"],
                      "secdfdTypes": ["Process"], "primarySecdfdType": "Process"}},
            node("t", label="type"), node("v", label="Variable"),
            node("ignore", ["Process"], "Process", label="Category"),
        ]
        rows = checked_node_report(nodes, {}, {}, {})
        self.assertEqual([row["id"] for row in rows], ["n", "t", "v"])
        self.assertEqual(rows[0]["outcomes"]["validity"], "VALID")

    def test_edge_metadata_requires_outgoing_edge_to_existing_category(self):
        nodes = [
            {"data": {"id": "secdfd:Process", "labels": ["Category"]}},
            {"data": {"id": "secdfd:Asset", "labels": ["Type"]}},
            {"data": {"id": "layer:Logic", "labels": ["Category"]}},
        ]
        edges = [
            {"data": {"source": "n", "target": target, "label": label}}
            for target, label in (
                ("secdfd:Process", "implements"), ("secdfd:Process", "implements"),
                ("secdfd:Asset", "implements"), ("layer:Logic", "implements"),
                ("secdfd:Flow", "implements"), ("secdfd:Process", "uses"),
            )
        ]
        edges.append({"data": {
            "source": "secdfd:Process", "target": "reverse", "label": "implements",
        }})
        self.assertEqual(build_secdfd_target_index(nodes, edges), {"n": ["secdfd:Process"]})
        self.assertEqual(get_sabo_edges({"elements": {"edges": edges}}), edges)
        self.assertEqual(get_sabo_edges({"edges": edges}), edges)
        self.assertEqual(get_sabo_edges({}), [])

    def test_summary_denominators_distinguish_nodes_entries_and_not_evaluated(self):
        rows = checked_node_report([
            node("multi", ["Process", "Undetermined", "Unknown"], "Process"),
            node("undefined"),
        ], {}, {}, {})
        summary = summarize_outcomes(rows)
        for dimension in DIMENSION_OUTCOMES:
            self.assertEqual(sum(summary[dimension].values()), 4)
        self.assertEqual(summary["mappingCorrectness"]["NOT_EVALUATED"], 3)
        node_summary = summarize_node_outcomes(rows)
        self.assertEqual(sum(node_summary["assignmentAndConsistency"].values()), 2)

    def make_report(self, extra_labels=None):
        original_path = Path("original.json")
        extended_path = Path("extended.json")
        sabo_path = Path("project.sabo.json")
        original = [mapping("n", "process"), mapping("NOT_FOUND", "asset")]
        extended = original + [mapping("n", "asset"), mapping("e", "asset")]
        documents = {
            original_path: original, extended_path: {"mappings": extended},
            sabo_path: {"elements": {
                "nodes": [
                    node("n", ["Process", "Data Store", "Asset", "Undetermined"]
                         + (extra_labels or []), "Process"),
                    node("e", ["Asset"], "Asset"), node("empty"),
                    node("u", ["Flow"], "Flow"),
                ], "edges": [],
            }},
        }
        with patch("generate_secdfd_type_report.load_json", side_effect=documents.__getitem__):
            return generate_report(original_path, sabo_path, extended_path)

    def test_full_report_uses_both_groundtruth_sources_and_retains_unmatched(self):
        report = self.make_report()
        summary = report["summary"]
        self.assertEqual(summary["totalCheckedNodes"], 4)
        self.assertEqual(summary["totalReportEntries"], 7)
        self.assertEqual(summary["totalOriginalGroundtruthEntries"], 2)
        self.assertEqual(summary["totalExtendedGroundtruthInputEntries"], 4)
        self.assertEqual(summary["totalExtendedGroundtruthEntries"], 2)
        self.assertEqual(summary["totalGroundtruthEntries"], 4)
        self.assertEqual(summary["unmatchedOriginalGroundtruthEntryCount"], 1)
        self.assertEqual(summary["unmatchedGroundtruthEntries"][0]["groundtruthSource"], "original")
        expected = {"CORRECT": 1, "PROBABLY_CORRECT": 2, "INCORRECT": 1, "UNASSESSED": 1}
        for label, count in expected.items():
            self.assertEqual(summary["conclusions"][label], count)
        self.assertEqual(sum(summary["conclusionCombinations"].values()), 7)
        self.assertEqual(set(summary["conclusions"]), set(CONCLUSIONS))

    def test_combined_report_reconciles_dimensions_and_counts_shared_ids_per_project(self):
        report = self.make_report()
        paths = [Path("first.json"), Path("second.json")]
        with patch("combine_secdfd_type_reports.load_report", return_value=report):
            combined = combine_reports(paths)
        summary = combined["summary"]
        self.assertEqual(summary["totalCheckedNodes"], 8)
        self.assertEqual(summary["totalReportEntries"], 14)
        self.assertEqual(summary["conclusions"]["PROBABLY_CORRECT"], 4)
        for dimension in DIMENSION_OUTCOMES:
            self.assertEqual(sum(summary["outcomes"][dimension].values()), 14)
        for dimension in summary["nodeOutcomes"]:
            self.assertEqual(sum(summary["nodeOutcomes"][dimension].values()), 8)
        self.assertEqual(summary["unmatchedGroundtruthEntryCount"], 2)
        self.assertEqual(summary["unmatchedGroundtruthEntries"][0]["sabo"], "NOT_FOUND")
        self.assertIn("sourceSabo", summary["unmatchedGroundtruthEntries"][0])

    def test_combiner_rejects_reports_from_previous_evaluation(self):
        old_report = {"summary": {}, "checkedNodes": []}
        with patch.object(Path, "open", mock_open(read_data=json.dumps(old_report))):
            with self.assertRaisesRegex(ValueError, "Regenerate report"):
                load_report(Path("old.json"))

    def test_count_checks_accept_empty_reports(self):
        validate_evaluation_counts([], summarize_outcomes([]))

    def test_count_checks_exclude_inconsistent_node_labels(self):
        rows = checked_node_report([
            node("inconsistent", ["Process", "Asset"], "Flow"),
            node("consistent", ["Process", "Undetermined", "Unknown"], "Process"),
        ], {}, {}, {})
        validate_evaluation_counts(rows, summarize_outcomes(rows))

    def test_count_checks_reject_wrong_validity_total(self):
        report = self.make_report()
        outcomes = deepcopy(report["summary"]["outcomes"])
        outcomes["validity"]["VALID"] += 1
        with self.assertRaisesRegex(ValueError, "Labels = Valid"):
            validate_evaluation_counts(report["checkedNodes"], outcomes)

    def test_count_checks_reject_wrong_determination_total(self):
        report = self.make_report()
        outcomes = deepcopy(report["summary"]["outcomes"])
        outcomes["determination"]["DETERMINED"] += 1
        with self.assertRaisesRegex(ValueError, "Determined \\+ Undetermined"):
            validate_evaluation_counts(report["checkedNodes"], outcomes)

    def test_count_checks_detect_missing_entry_from_multilabel_node(self):
        report = self.make_report()
        rows = report["checkedNodes"][1:]
        with self.assertRaisesRegex(ValueError, "Labels = Valid"):
            validate_evaluation_counts(rows, summarize_outcomes(rows))

    def test_combiner_rejects_source_report_with_wrong_counts(self):
        report = self.make_report()
        report["summary"]["outcomes"]["validity"]["ALMOST_VALID"] += 1
        with patch.object(Path, "open", mock_open(read_data=json.dumps(report))):
            with self.assertRaisesRegex(ValueError, "Invalid evaluation counts in corrupt.json"):
                load_report(Path("corrupt.json"))

    def test_invalid_type_summary_deduplicates_exact_json_values(self):
        values = [
            "Unknown", "Unknown", "UNKNOWN", None, None, True, 1,
            {"a": 1, "b": 2}, {"b": 2, "a": 1}, ["bad"], ["bad"],
        ]
        rows = [{"invalidSecdfdType": value} for value in values] + [{}]
        self.assertEqual(summarize_invalid_secdfd_types(rows), [
            "Unknown", "UNKNOWN", None, True, 1, {"a": 1, "b": 2}, ["bad"],
        ])

    def test_invalid_type_summary_excludes_nonevaluated_labels(self):
        rows = checked_node_report([
            node("inconsistent", ["Bad"], "Different"),
            node("consistent", ["Process", "Unknown", "Unknown"], "Process"),
            node("undefined"),
        ], {}, {}, {})
        self.assertEqual(summarize_invalid_secdfd_types(rows), ["Unknown"])
        self.assertEqual(summarize_invalid_secdfd_types([]), [])

    def test_report_summary_includes_distinct_invalid_types(self):
        report = self.make_report(["Unknown", "Command", "Unknown"])
        self.assertEqual(report["summary"]["invalidSecdfdTypes"], ["Unknown", "Command"])
        self.assertEqual(self.make_report()["summary"]["invalidSecdfdTypes"], [])

    def test_combined_summary_merges_invalid_types_across_projects(self):
        reports = [self.make_report(["Unknown", "Command"]),
                   self.make_report(["Command", "UNKNOWN"])]
        with patch("combine_secdfd_type_reports.load_report", side_effect=reports):
            combined = combine_reports([Path("first.json"), Path("second.json")])
        self.assertEqual(combined["summary"]["invalidSecdfdTypes"], [
            "Unknown", "Command", "UNKNOWN",
        ])
        self.assertEqual(combined["sourceReports"][0]["summary"]["invalidSecdfdTypes"], [
            "Unknown", "Command",
        ])


if __name__ == "__main__":
    unittest.main()
