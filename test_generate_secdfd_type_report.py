import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from combine_secdfd_type_reports import combine_reports
from generate_secdfd_type_report import (
    SECDFD_TARGET_TO_GROUNDTRUTH,
    VALID_SECDFD_TYPES,
    build_secdfd_target_index,
    checked_node_report,
    conclusion_for,
    generate_report,
    get_sabo_edges,
)


def node(node_id, types=None, primary=None, label="Operation"):
    return {"data": {"id": node_id, "labels": [label], "properties": {
        "secdfdTypes": types, "primarySecdfdType": primary,
    }}}


class SecdfdTypeReportTests(unittest.TestCase):
    def test_exact_edge_to_groundtruth_mappings(self):
        for target, expected in SECDFD_TARGET_TO_GROUNDTRUTH.items():
            label = target.removeprefix("secdfd:")
            with self.subTest(target=target):
                self.assertEqual(
                    conclusion_for(label, [target], [{"secdfd_type": expected}]),
                    ["SECDFD_TYPE_MATCHES"],
                )
                for incorrect in (expected.upper(), expected + " ", "different"):
                    self.assertEqual(
                        conclusion_for(label, [target], [{"secdfd_type": incorrect}]),
                        ["SECDFD_TYPE_DOES_NOT_MATCH"],
                    )

    def test_alias_validity_is_case_insensitive_but_edge_target_is_literal(self):
        for alias in VALID_SECDFD_TYPES:
            label = alias.upper()
            with self.subTest(label=label):
                self.assertEqual(
                    conclusion_for(label, [], []),
                    ["ALMOST_VALID_SECDFD_TYPE", "GROUND_TRUTH_NOT_FOUND"],
                )
        self.assertEqual(
            conclusion_for("process", ["secdfd:Process"], []),
            ["ALMOST_VALID_SECDFD_TYPE", "GROUND_TRUTH_NOT_FOUND"],
        )

    def test_undefined_requires_both_fields_empty(self):
        inputs = [node("n", [], primary) for primary in (None, "", " ")]
        inputs.append({"data": {"id": "n", "labels": ["Type"]}})
        for item in inputs:
            with self.subTest(node=item):
                rows = checked_node_report([item], {}, {"n": ["secdfd:Process"]})
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["conclusions"], [
                    "SECDFD_TYPE_UNDEFINED", "GROUND_TRUTH_NOT_FOUND",
                ])
                self.assertIsNone(rows[0]["secdfdTypeIndex"])

    def test_inconsistency_for_empty_types_and_any_defined_primary(self):
        for types in (None, [], ""):
            for primary in ("get", "Process", "Undetermined"):
                with self.subTest(types=types, primary=primary):
                    item = node("n", types, primary)
                    rows = checked_node_report([item], {}, {"n": ["secdfd:Process"]})
                    self.assertEqual(rows[0]["conclusions"], [
                        "INCONSISTENCY", "GROUND_TRUTH_NOT_FOUND",
                    ])
                    rows = checked_node_report([item], {"n": [{}]}, {})
                    self.assertEqual(rows[0]["conclusions"], ["INCONSISTENCY"])

    def test_each_type_uses_its_own_edge_and_groundtruth(self):
        rows = checked_node_report(
            [node("n", ["DataStore", "Process", "Asset"], "Flow")],
            {"n": [{"secdfd_type": "process"}]},
            {"n": ["secdfd:Process", "secdfd:Asset"]},
        )
        self.assertEqual([row["secdfdType"] for row in rows], [
            "DataStore", "Process", "Asset",
        ])
        self.assertEqual([row["conclusion"] for row in rows], [
            "ALMOST_VALID_SECDFD_TYPE", "SECDFD_TYPE_MATCHES",
            "SECDFD_TYPE_DOES_NOT_MATCH",
        ])
        self.assertEqual([row["secdfdTypeIndex"] for row in rows], [0, 1, 2])
        self.assertEqual(rows[0]["almostValidSecdfdType"], "DataStore")
        self.assertFalse(rows[0]["hasImplementsEdge"])
        self.assertTrue(rows[1]["hasImplementsEdge"])

    def test_populated_types_are_checked_without_primary(self):
        rows = checked_node_report(
            [node("n", ["Process", "Asset"])],
            {"n": [{"secdfd_type": "process"}]},
            {"n": ["secdfd:Process", "secdfd:Asset"]},
        )
        self.assertEqual([row["conclusion"] for row in rows], [
            "SECDFD_TYPE_MATCHES", "SECDFD_TYPE_DOES_NOT_MATCH",
        ])

    def test_invalid_type_values_are_preserved(self):
        invalid_values = ["Unknown", "data-store", "Process ", 42, None]
        rows = checked_node_report([node("n", invalid_values, "Process")], {}, {})
        self.assertEqual([row["invalidSecdfdType"] for row in rows], invalid_values)
        for row in rows:
            self.assertEqual(row["conclusions"], [
                "INVALID_SECDFD_TYPE", "GROUND_TRUTH_NOT_FOUND",
            ])

    def test_undetermined_is_checked_separately_from_other_labels(self):
        rows = checked_node_report(
            [node("n", ["Undetermined", "Process"])],
            {"n": [{"secdfd_type": "process"}]},
            {"n": ["secdfd:Undetermined", "secdfd:Process"]},
        )
        self.assertEqual([row["conclusion"] for row in rows], [
            "SECDFD_TYPE_UNDETERMINED", "SECDFD_TYPE_MATCHES",
        ])
        self.assertEqual(
            conclusion_for("Undetermined", ["secdfd:Undetermined"], []),
            ["SECDFD_TYPE_UNDETERMINED", "GROUND_TRUTH_NOT_FOUND"],
        )
        self.assertEqual(
            conclusion_for("Undetermined", [], []),
            ["ALMOST_VALID_SECDFD_TYPE", "GROUND_TRUTH_NOT_FOUND"],
        )

    def test_determined_edge_without_groundtruth(self):
        self.assertEqual(
            conclusion_for("Process", ["secdfd:Process"], []),
            ["GROUND_TRUTH_NOT_FOUND"],
        )

    def test_any_matching_groundtruth_entry_for_current_label(self):
        self.assertEqual(
            conclusion_for("Process", ["secdfd:Process"],
                           [{"secdfd_type": "flow"}, {"secdfd_type": "process"}]),
            ["SECDFD_TYPE_MATCHES"],
        )

    def test_edge_index_requires_outgoing_edge_to_existing_category(self):
        nodes = [
            {"data": {"id": "secdfd:Process", "labels": ["Category"]}},
            {"data": {"id": "secdfd:Asset", "labels": ["Type"]}},
            {"data": {"id": "layer:Logic", "labels": ["Category"]}},
        ]
        edges = [
            {"data": {"source": "n", "target": target, "label": label}}
            for target, label in (
                ("secdfd:Process", "implements"),
                ("secdfd:Process", "implements"),
                ("secdfd:Asset", "implements"),
                ("layer:Logic", "implements"),
                ("secdfd:Flow", "implements"),
                ("secdfd:Process", "uses"),
            )
        ]
        edges.append({"data": {
            "source": "secdfd:Process", "target": "reverse", "label": "implements",
        }})
        self.assertEqual(build_secdfd_target_index(nodes, edges), {"n": ["secdfd:Process"]})
        self.assertEqual(get_sabo_edges({"elements": {"edges": edges}}), edges)
        self.assertEqual(get_sabo_edges({"edges": edges}), edges)
        self.assertEqual(get_sabo_edges({"elements": {}}), [])

    def test_node_filter_and_direct_data_fields(self):
        nodes = [
            {"data": {"id": "n", "labels": ["operation"],
                      "secdfdTypes": ["Process"], "primarySecdfdType": None}},
            node("t", [], label="type"),
            node("v", [], label="Variable"),
            node("ignore", ["Process"], label="Category"),
        ]
        rows = checked_node_report(nodes, {}, {})
        self.assertEqual([row["id"] for row in rows], ["n", "t", "v"])
        self.assertEqual(rows[0]["almostValidSecdfdType"], "Process")

    def test_duplicate_type_items_keep_separate_indices(self):
        rows = checked_node_report([node("n", ["Process", "Process"])], {}, {})
        self.assertEqual([row["secdfdTypeIndex"] for row in rows], [0, 1])

    def test_generate_and_combine_separate_node_and_entry_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groundtruth = root / "groundtruth.json"
            sabo = root / "sabo.json"
            groundtruth.write_text(json.dumps([
                {"sabo": "n", "secdfd_type": "process"},
                {"sabo": "NOT_FOUND", "secdfd_type": "asset"},
            ]), encoding="utf-8")
            sabo.write_text(json.dumps({"elements": {
                "nodes": [
                    node("n", ["Process", "Asset", "Undetermined"]),
                    node("empty"),
                    {"data": {"id": "secdfd:Process", "labels": ["Category"]}},
                ],
                "edges": [{"data": {
                    "source": "n", "target": "secdfd:Process", "label": "implements",
                }}],
            }}), encoding="utf-8")
            report = generate_report(groundtruth, sabo)
            self.assertEqual(report["summary"]["totalCheckedNodes"], 2)
            self.assertEqual(report["summary"]["totalReportEntries"], 4)
            self.assertEqual(len(report["summary"]["unmatchedGroundtruthEntries"]), 1)
            counts = Counter(row["conclusion"] for row in report["checkedNodes"])
            for conclusion, count in counts.items():
                self.assertEqual(report["summary"]["conclusions"][conclusion], count)

            report_paths = [root / "first.json", root / "second.json"]
            for path in report_paths:
                path.write_text(json.dumps(report), encoding="utf-8")
            combined = combine_reports(report_paths)
            self.assertEqual(combined["summary"]["totalCheckedNodes"], 4)
            self.assertEqual(combined["summary"]["totalReportEntries"], 8)
            self.assertEqual(len(combined["checkedNodes"]), 8)
            self.assertEqual(combined["summary"]["conclusions"]["SECDFD_TYPE_MATCHES"], 2)
            for source in combined["sourceReports"]:
                self.assertEqual(source["totalCheckedNodes"], 2)
                self.assertEqual(source["totalReportEntries"], 4)


if __name__ == "__main__":
    unittest.main()
