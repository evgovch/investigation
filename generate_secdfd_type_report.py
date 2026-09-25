from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CHECKED_LABELS = {"Variable", "Operation", "Type"}
VALID_SECDFD_TYPES = {
    "externalentity",
    "datastore",
    "asset",
    "process",
    "flow",
}
CONCLUSIONS = (
    "INVALID_SECDFD_TYPE",
    "SECDFD_TYPE_UNDETERMINED",
    "SECDFD_TYPE_UNDEFINED",
    "GROUND_TRUTH_NOT_FOUND",
    "INVALID_SECDFD_TYPE and GROUND_TRUTH_NOT_FOUND",
    "SECDFD_TYPE_UNDETERMINED and GROUND_TRUTH_NOT_FOUND",
    "SECDFD_TYPE_UNDEFINED and GROUND_TRUTH_NOT_FOUND",
    "SECDFD_TYPE_DOES_NOT_MATCH",
    "SECDFD_TYPE_MATCHES",
)
UNMATCHED_GROUNDTRUTH_CONCLUSION = "GROUND_TRUTH_ENTRIES_NOT_MATCHED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SABO node primarySecdfdType values with SecDFD types from "
            "a ground-truth SABO mapping file."
        )
    )
    parser.add_argument(
        "--groundtruth",
        "--groundtruth-json",
        "-g",
        required=True,
        type=Path,
        help="Path to the ground-truth JSON file.",
    )
    parser.add_argument(
        "--sabo",
        "-s",
        required=True,
        type=Path,
        help="Path to the SABO JSON file.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory where the generated report is written.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=(
            "Optional report path. If omitted, a filename is generated inside "
            "--reports-dir."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def ensure_mapping_entries(groundtruth_json: Any) -> list[dict[str, Any]]:
    if isinstance(groundtruth_json, list):
        entries = groundtruth_json
    elif isinstance(groundtruth_json, dict) and isinstance(
        groundtruth_json.get("mappings"), list
    ):
        entries = groundtruth_json["mappings"]
    else:
        raise ValueError(
            "Ground-truth JSON must be a list, or an object with a 'mappings' list."
        )

    return [entry for entry in entries if isinstance(entry, dict)]


def get_sabo_nodes(sabo_json: Any) -> list[dict[str, Any]]:
    if not isinstance(sabo_json, dict):
        raise ValueError("SABO JSON must be an object.")

    nodes = sabo_json.get("nodes")
    if nodes is None:
        elements = sabo_json.get("elements")
        if isinstance(elements, dict):
            nodes = elements.get("nodes")

    if not isinstance(nodes, list):
        raise ValueError("SABO JSON must contain a nodes list at 'elements.nodes'.")

    return [node for node in nodes if isinstance(node, dict)]


def normalize_secdfd_type(value: Any) -> str:
    """Normalize common style variants: DataStore, data_store, data store."""
    normalized = str(value).strip().casefold()
    return re.sub(r"[\s_-]+", "", normalized)


def is_defined(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or value.strip() != "")


def labels_from_data(data: dict[str, Any]) -> list[str]:
    labels = data.get("labels", [])
    if isinstance(labels, list):
        return [str(label) for label in labels]
    if labels is None:
        return []
    return [str(labels)]


def build_groundtruth_index(
    groundtruth_entries: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in groundtruth_entries:
        sabo_id = entry.get("sabo")
        if isinstance(sabo_id, str) and sabo_id.strip():
            index[sabo_id].append(entry)
    return dict(index)


def unique_defined_values(values: list[Any]) -> list[Any]:
    seen = set()
    unique = []
    for value in values:
        if not is_defined(value):
            continue
        key = str(value)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def conclusion_for(
    primary_secdfd_type: Any, groundtruth_entries: list[dict[str, Any]]
) -> str:
    if not is_defined(primary_secdfd_type):
        primary_conclusion = "SECDFD_TYPE_UNDEFINED"
    else:
        normalized_primary = normalize_secdfd_type(primary_secdfd_type)
        if normalized_primary == "undetermined":
            primary_conclusion = "SECDFD_TYPE_UNDETERMINED"
        elif str(primary_secdfd_type).strip().casefold() not in VALID_SECDFD_TYPES:
            primary_conclusion = "INVALID_SECDFD_TYPE"
        else:
            primary_conclusion = None

    if not groundtruth_entries:
        if primary_conclusion is not None:
            return f"{primary_conclusion} and GROUND_TRUTH_NOT_FOUND"
        return "GROUND_TRUTH_NOT_FOUND"

    if primary_conclusion is not None:
        return primary_conclusion

    normalized_groundtruth_types = {
        normalize_secdfd_type(entry.get("secdfd_type"))
        for entry in groundtruth_entries
        if is_defined(entry.get("secdfd_type"))
    }
    if normalized_primary in normalized_groundtruth_types:
        return "SECDFD_TYPE_MATCHES"
    return "SECDFD_TYPE_DOES_NOT_MATCH"


def checked_node_report(
    nodes: list[dict[str, Any]], groundtruth_index: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    report_rows = []
    for node in nodes:
        data = node.get("data")
        if not isinstance(data, dict):
            continue

        labels = labels_from_data(data)
        if not CHECKED_LABELS.intersection(labels):
            continue

        node_id = data.get("id")
        properties = data.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

        primary_secdfd_type = properties.get("primarySecdfdType")
        groundtruth_entries = groundtruth_index.get(node_id, [])
        conclusion = conclusion_for(primary_secdfd_type, groundtruth_entries)
        groundtruth_types = unique_defined_values(
            [entry.get("secdfd_type") for entry in groundtruth_entries]
        )

        report_rows.append(
            {
                "id": node_id,
                "labels": labels,
                "primarySecdfdType": primary_secdfd_type,
                "groundTruthSecdfdTypes": groundtruth_types,
                "groundTruthEntryCount": len(groundtruth_entries),
                "conclusion": conclusion,
            }
        )
    return report_rows


def find_unmatched_groundtruth_entries(
    groundtruth_entries: list[dict[str, Any]], nodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    sabo_node_ids = set()
    for node in nodes:
        data = node.get("data")
        if isinstance(data, dict) and is_defined(data.get("id")):
            sabo_node_ids.add(data["id"])

    return [
        entry for entry in groundtruth_entries if entry.get("sabo") not in sabo_node_ids
    ]


def safe_filename_part(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "report"


def default_output_path(groundtruth_path: Path, sabo_path: Path, reports_dir: Path) -> Path:
    filename = (
        f"{safe_filename_part(sabo_path)}__"
        f"{safe_filename_part(groundtruth_path)}__secdfd_type_report.json"
    )
    return reports_dir / filename


def generate_report(groundtruth_path: Path, sabo_path: Path) -> dict[str, Any]:
    groundtruth_entries = ensure_mapping_entries(load_json(groundtruth_path))
    sabo_nodes = get_sabo_nodes(load_json(sabo_path))
    groundtruth_index = build_groundtruth_index(groundtruth_entries)
    checked_nodes = checked_node_report(sabo_nodes, groundtruth_index)
    conclusion_counts = Counter(row["conclusion"] for row in checked_nodes)
    unmatched_groundtruth_entries = find_unmatched_groundtruth_entries(
        groundtruth_entries, sabo_nodes
    )

    return {
        "groundtruth": str(groundtruth_path),
        "sabo": str(sabo_path),
        "checkedLabels": sorted(CHECKED_LABELS),
        "summary": {
            "totalGroundtruthEntries": len(groundtruth_entries),
            "groundtruthEntriesWithSabo": sum(
                len(entries) for entries in groundtruth_index.values()
            ),
            "totalSaboNodes": len(sabo_nodes),
            "totalCheckedNodes": len(checked_nodes),
            "conclusions": {
                **{
                    conclusion: conclusion_counts.get(conclusion, 0)
                    for conclusion in CONCLUSIONS
                },
                UNMATCHED_GROUNDTRUTH_CONCLUSION: len(unmatched_groundtruth_entries),
            },
            "unmatchedGroundtruthEntries": unmatched_groundtruth_entries,
        },
        "checkedNodes": checked_nodes,
    }


def main() -> None:
    args = parse_args()
    groundtruth_path = args.groundtruth.resolve()
    sabo_path = args.sabo.resolve()

    if not groundtruth_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {groundtruth_path}")
    if not sabo_path.is_file():
        raise FileNotFoundError(f"SABO file not found: {sabo_path}")

    output_path = args.output
    if output_path is None:
        output_path = default_output_path(groundtruth_path, sabo_path, args.reports_dir)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = generate_report(groundtruth_path, sabo_path)
    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
        report_file.write("\n")

    print(f"Wrote report: {output_path}")
    print(f"Checked nodes: {report['summary']['totalCheckedNodes']}")
    for conclusion, count in report["summary"]["conclusions"].items():
        print(f"{conclusion}: {count}")


if __name__ == "__main__":
    main()
