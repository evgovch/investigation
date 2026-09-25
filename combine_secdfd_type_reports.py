from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


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
    return report


def combine_reports(report_paths: list[Path]) -> dict[str, Any]:
    source_reports = []
    checked_labels = set()
    checked_nodes = []
    unmatched_groundtruth_entries = []
    conclusion_counts: Counter[str] = Counter()
    total_groundtruth_entries = 0
    groundtruth_entries_with_sabo = 0
    total_sabo_nodes = 0

    for path in report_paths:
        report = load_report(path)
        summary = report["summary"]
        groundtruth = report.get("groundtruth")
        sabo = report.get("sabo")
        conclusions = summary.get("conclusions", {})
        if not isinstance(conclusions, dict):
            raise ValueError(f"Report has invalid conclusion counts: {path}")

        checked_labels.update(report.get("checkedLabels", []))
        conclusion_counts.update(conclusions)
        total_groundtruth_entries += summary.get("totalGroundtruthEntries", 0)
        groundtruth_entries_with_sabo += summary.get(
            "groundtruthEntriesWithSabo", 0
        )
        total_sabo_nodes += summary.get("totalSaboNodes", 0)

        source_reports.append(
            {
                "report": str(path),
                "groundtruth": groundtruth,
                "sabo": sabo,
                "totalCheckedNodes": summary.get("totalCheckedNodes", 0),
                "conclusions": conclusions,
            }
        )

        source_metadata = {
            "sourceReport": path.name,
            "groundtruth": groundtruth,
            "sabo": sabo,
        }
        for node in report["checkedNodes"]:
            checked_nodes.append({**source_metadata, **node})

        for entry in summary.get("unmatchedGroundtruthEntries", []):
            unmatched_groundtruth_entries.append({**source_metadata, **entry})

    return {
        "sourceReports": source_reports,
        "checkedLabels": sorted(checked_labels),
        "summary": {
            "reportCount": len(report_paths),
            "totalGroundtruthEntries": total_groundtruth_entries,
            "groundtruthEntriesWithSabo": groundtruth_entries_with_sabo,
            "totalSaboNodes": total_sabo_nodes,
            "totalCheckedNodes": len(checked_nodes),
            "conclusions": dict(conclusion_counts),
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


if __name__ == "__main__":
    main()
