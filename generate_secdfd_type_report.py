from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CHECKED_LABELS = {"Variable", "Operation", "Type"}
CHECKED_LABELS_CASEFOLDED = {label.casefold() for label in CHECKED_LABELS}
VALID_SECDFD_TYPES = {
    "external_entity",
    "externalentity",
    "external entity",
    "datastore",
    "data store",
    "data_store",
    "asset",
    "process",
    "flow",
    "undetermined",
}
SECDFD_TARGET_TO_GROUNDTRUTH = {
    "secdfd:Process": "process",
    "secdfd:Asset": "asset",
    "secdfd:DataStore": "data_store",
    "secdfd:Flow": "flow",
    "secdfd:External Entity": "external_entity",
}
CONCLUSIONS = (
    "INVALID_SECDFD_TYPE",
    "ALMOST_VALID_SECDFD_TYPE",
    "SECDFD_TYPE_UNDETERMINED",
    "SECDFD_TYPE_UNDEFINED",
    "INCONSISTENCY",
    "GROUND_TRUTH_NOT_FOUND",
    "INVALID_SECDFD_TYPE and GROUND_TRUTH_NOT_FOUND",
    "ALMOST_VALID_SECDFD_TYPE and GROUND_TRUTH_NOT_FOUND",
    "SECDFD_TYPE_UNDETERMINED and GROUND_TRUTH_NOT_FOUND",
    "SECDFD_TYPE_UNDEFINED and GROUND_TRUTH_NOT_FOUND",
    "INCONSISTENCY and GROUND_TRUTH_NOT_FOUND",
    "SECDFD_TYPE_DOES_NOT_MATCH",
    "SECDFD_TYPE_MATCHES",
)
UNMATCHED_GROUNDTRUTH_CONCLUSION = "GROUND_TRUTH_ENTRIES_NOT_MATCHED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SABO SecDFD implements edges with SecDFD types from "
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


def get_sabo_edges(sabo_json: dict[str, Any]) -> list[dict[str, Any]]:
    edges = sabo_json.get("edges")
    if edges is None:
        elements = sabo_json.get("elements", {})
        edges = elements.get("edges", []) if isinstance(elements, dict) else []
    if not isinstance(edges, list):
        raise ValueError("SABO edges must be a list.")
    return [edge for edge in edges if isinstance(edge, dict)]


def is_defined(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or value.strip() != "")


def labels_from_data(data: dict[str, Any]) -> list[str]:
    labels = data.get("labels", [])
    if isinstance(labels, list):
        return [str(label) for label in labels]
    if labels is None:
        return []
    return [str(labels)]


def build_secdfd_target_index(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, list[str]]:
    category_ids = set()
    for node in nodes:
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        node_id = data.get("id")
        if (
            isinstance(node_id, str)
            and node_id.startswith("secdfd:")
            and "category" in {label.casefold() for label in labels_from_data(data)}
        ):
            category_ids.add(node_id)

    index: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        data = edge.get("data")
        if not isinstance(data, dict) or data.get("label") != "implements":
            continue
        source, target = data.get("source"), data.get("target")
        if isinstance(source, str) and target in category_ids:
            if target not in index[source]:
                index[source].append(target)
    return dict(index)


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
    secdfd_type: Any,
    secdfd_targets: list[str],
    groundtruth_entries: list[dict[str, Any]],
) -> list[str]:
    conclusions = []
    if not isinstance(secdfd_type, str) or secdfd_type.casefold() not in VALID_SECDFD_TYPES:
        conclusions.append("INVALID_SECDFD_TYPE")
    elif f"secdfd:{secdfd_type}" not in secdfd_targets:
        conclusions.append("ALMOST_VALID_SECDFD_TYPE")
    elif secdfd_type.casefold() == "undetermined":
        conclusions.append("SECDFD_TYPE_UNDETERMINED")
    elif groundtruth_entries:
        target = f"secdfd:{secdfd_type}"
        matches = any(
            target in SECDFD_TARGET_TO_GROUNDTRUTH
            and entry.get("secdfd_type") == SECDFD_TARGET_TO_GROUNDTRUTH[target]
            for entry in groundtruth_entries
        )
        conclusions.append(
            "SECDFD_TYPE_MATCHES" if matches else "SECDFD_TYPE_DOES_NOT_MATCH"
        )

    if not groundtruth_entries:
        conclusions.append("GROUND_TRUTH_NOT_FOUND")
    return conclusions


def is_checked_node(data: dict[str, Any]) -> bool:
    return bool(CHECKED_LABELS_CASEFOLDED.intersection(
        label.casefold() for label in labels_from_data(data)
    ))


def secdfd_property(data: dict[str, Any], name: str) -> Any:
    if name in data:
        return data[name]
    properties = data.get("properties")
    if isinstance(properties, dict):
        return properties.get(name)
    return None


def checked_node_report(
    nodes: list[dict[str, Any]],
    groundtruth_index: dict[str, list[dict[str, Any]]],
    secdfd_target_index: dict[str, list[str]],
) -> list[dict[str, Any]]:
    report_rows = []
    for node in nodes:
        data = node.get("data")
        if not isinstance(data, dict):
            continue

        labels = labels_from_data(data)
        if not is_checked_node(data):
            continue

        node_id = data.get("id")
        primary_secdfd_type = secdfd_property(data, "primarySecdfdType")
        secdfd_types = secdfd_property(data, "secdfdTypes")
        secdfd_targets = secdfd_target_index.get(node_id, [])
        groundtruth_entries = groundtruth_index.get(node_id, [])
        groundtruth_types = unique_defined_values(
            [entry.get("secdfd_type") for entry in groundtruth_entries]
        )

        if secdfd_types and is_defined(secdfd_types):
            if not isinstance(secdfd_types, list):
                raise ValueError(f"secdfdTypes must be a list for node {node_id!r}")
            type_entries = list(enumerate(secdfd_types))
        else:
            type_entries = [(None, None)]

        has_inconsistency = is_defined(primary_secdfd_type) and (
            not isinstance(secdfd_types, list) or primary_secdfd_type not in secdfd_types
        )

        for type_index, secdfd_type in type_entries:
            if type_index is None:
                conclusions = [] if has_inconsistency else ["SECDFD_TYPE_UNDEFINED"]
                if not groundtruth_entries:
                    conclusions.append("GROUND_TRUTH_NOT_FOUND")
            else:
                conclusions = conclusion_for(
                    secdfd_type, secdfd_targets, groundtruth_entries
                )

            if has_inconsistency:
                conclusions.insert(0, "INCONSISTENCY")

            expected_target = (
                f"secdfd:{secdfd_type}"
                if type_index is not None and isinstance(secdfd_type, str)
                else None
            )
            row = {
                "id": node_id,
                "labels": labels,
                "primarySecdfdType": primary_secdfd_type,
                "secdfdTypes": secdfd_types,
                "secdfdType": secdfd_type,
                "secdfdTypeIndex": type_index,
                "implementsSecdfdTargets": secdfd_targets,
                "expectedSecdfdTarget": expected_target,
                "hasImplementsEdge": expected_target in secdfd_targets,
                "groundTruthSecdfdTypes": groundtruth_types,
                "groundTruthEntryCount": len(groundtruth_entries),
                "conclusion": " and ".join(conclusions),
                "conclusions": conclusions,
            }
            if "INVALID_SECDFD_TYPE" in conclusions:
                row["invalidSecdfdType"] = secdfd_type
            if "ALMOST_VALID_SECDFD_TYPE" in conclusions:
                row["almostValidSecdfdType"] = secdfd_type
            report_rows.append(row)
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
    sabo_json = load_json(sabo_path)
    sabo_nodes = get_sabo_nodes(sabo_json)
    secdfd_target_index = build_secdfd_target_index(
        sabo_nodes, get_sabo_edges(sabo_json)
    )
    groundtruth_index = build_groundtruth_index(groundtruth_entries)
    checked_nodes = checked_node_report(
        sabo_nodes, groundtruth_index, secdfd_target_index
    )
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
            "totalCheckedNodes": sum(
                is_checked_node(data)
                for node in sabo_nodes
                if isinstance((data := node.get("data")), dict)
            ),
            "totalReportEntries": len(checked_nodes),
            "conclusions": {
                **{
                    conclusion: conclusion_counts.get(conclusion, 0)
                    for conclusion in CONCLUSIONS
                },
                **{
                    conclusion: count
                    for conclusion, count in conclusion_counts.items()
                    if conclusion not in CONCLUSIONS
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
    print(f"Report entries: {report['summary']['totalReportEntries']}")
    for conclusion, count in report["summary"]["conclusions"].items():
        print(f"{conclusion}: {count}")


if __name__ == "__main__":
    main()
