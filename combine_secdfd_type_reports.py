from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from generate_secdfd_type_report import (
    EVALUATION_VERSION,
    summarize_conclusions,
    summarize_invalid_secdfd_types,
    summarize_node_outcomes,
    summarize_outcomes,
    validate_evaluation_counts,
)


TOTAL_FIELDS = (
    "totalGroundtruthEntries", "totalOriginalGroundtruthEntries",
    "totalExtendedGroundtruthEntries", "totalExtendedGroundtruthInputEntries",
    "groundtruthEntriesWithSabo", "totalSaboNodes", "totalCheckedNodes",
    "totalReportEntries", "unmatchedGroundtruthEntryCount",
    "unmatchedOriginalGroundtruthEntryCount", "unmatchedExtendedGroundtruthEntryCount",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine SecDFD type reports into one aggregate report."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory containing individual SecDFD type reports.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path. Defaults to combined_secdfd_type_report.json.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    if not isinstance(report, dict) or not isinstance(report.get("summary"), dict):
        raise ValueError(f"Invalid report structure: {path}")
    if not isinstance(report.get("checkedNodes"), list):
        raise ValueError(f"Report has no checkedNodes list: {path}")
    if report.get("evaluationVersion") != EVALUATION_VERSION:
        raise ValueError(f"Regenerate report with current evaluation criteria: {path}")
    try:
        validate_evaluation_counts(report["checkedNodes"], report["summary"]["outcomes"])
    except ValueError as error:
        raise ValueError(f"Invalid evaluation counts in {path}: {error}") from error
    return report


def combine_reports(report_paths: list[Path]) -> dict[str, Any]:
    source_reports = []
    checked_labels = set()
    checked_nodes = []
    unmatched_groundtruth_entries = []
    totals: Counter[str] = Counter()

    for path in report_paths:
        report = load_report(path)
        summary = report["summary"]
        groundtruth = report.get("groundtruth")
        sabo = report.get("sabo")
        checked_labels.update(report.get("checkedLabels", []))
        totals.update({field: summary[field] for field in TOTAL_FIELDS})

        source_reports.append(
            {
                "report": str(path),
                "groundtruth": groundtruth,
                "extendedGroundtruth": report.get("extendedGroundtruth"),
                "sabo": sabo,
                "summary": summary,
            }
        )

        source_metadata = {
            "sourceReport": path.name,
            "groundtruth": groundtruth,
            "extendedGroundtruth": report.get("extendedGroundtruth"),
            "sabo": sabo,
        }
        for node in report["checkedNodes"]:
            checked_nodes.append({**source_metadata, **node})

        for entry in summary.get("unmatchedGroundtruthEntries", []):
            unmatched_groundtruth_entries.append({
                "sourceReport": path.name,
                "sourceGroundtruth": groundtruth,
                "sourceExtendedGroundtruth": report.get("extendedGroundtruth"),
                "sourceSabo": sabo,
                **entry,
            })

    outcomes = summarize_outcomes(checked_nodes)
    validate_evaluation_counts(checked_nodes, outcomes)
    return {
        "evaluationVersion": EVALUATION_VERSION,
        "sourceReports": source_reports,
        "checkedLabels": sorted(checked_labels),
        "summary": {
            "reportCount": len(report_paths),
            **dict(totals),
            "totalReportEntries": len(checked_nodes),
            "outcomes": outcomes,
            "nodeOutcomes": summarize_node_outcomes(checked_nodes),
            "conclusions": summarize_conclusions(checked_nodes),
            "invalidSecdfdTypes": summarize_invalid_secdfd_types(checked_nodes),
            "conclusionCombinations": dict(Counter(row["conclusion"] for row in checked_nodes)),
            "unmatchedGroundtruthEntries": unmatched_groundtruth_entries,
        },
        "checkedNodes": checked_nodes,
    }


def main() -> None:
    args = parse_args()
    reports_dir = args.reports_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else reports_dir / "combined_secdfd_type_report.json"
    )
    report_paths = sorted(reports_dir.glob("*__secdfd_type_report.json"))
    if not report_paths:
        raise FileNotFoundError(f"No SecDFD type reports found in {reports_dir}")

    combined_report = combine_reports(report_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(combined_report, output_file, indent=2)
        output_file.write("\n")

    print(f"Wrote combined report: {output_path}")
    print(f"Source reports: {combined_report['summary']['reportCount']}")
    print(f"Checked nodes: {combined_report['summary']['totalCheckedNodes']}")
    print(f"Report entries: {combined_report['summary']['totalReportEntries']}")


if __name__ == "__main__":
    main()
