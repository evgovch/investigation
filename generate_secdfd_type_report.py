from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EVALUATION_VERSION = 2
CHECKED_LABELS = {"Variable", "Operation", "Type"}
CHECKED_LABELS_CASEFOLDED = {label.casefold() for label in CHECKED_LABELS}
VALID_SECDFD_TYPES = {
    "Asset", "DataStore", "Process", "Flow", "External Entity", "Undetermined",
}
SECDFD_ALIASES = {label.casefold(): label for label in VALID_SECDFD_TYPES}
SECDFD_ALIASES.update({
    "data store": "DataStore", "data_store": "DataStore",
    "externalentity": "External Entity", "external_entity": "External Entity",
})
DIMENSION_OUTCOMES = {
    "assignmentAndConsistency": ("UNDEFINED", "DEFINED_AND_CONSISTENT", "INCONSISTENT"),
    "validity": ("VALID", "ALMOST_VALID", "INVALID"),
    "determination": ("DETERMINED", "UNDETERMINED"),
    "groundTruthAvailability": (
        "ORIGINAL_GROUND_TRUTH_AVAILABLE", "EXTENDED_GROUND_TRUTH_ONLY",
        "GROUND_TRUTH_UNAVAILABLE",
    ),
    "mappingCorrectness": ("CORRECT", "PROBABLY_CORRECT", "INCORRECT", "UNASSESSED"),
}
CONCLUSIONS = tuple(label for labels in DIMENSION_OUTCOMES.values() for label in labels)
NODE_DIMENSIONS = ("assignmentAndConsistency", "groundTruthAvailability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SABO SecDFD assignments across five evaluation dimensions."
    )
    parser.add_argument("--groundtruth", "--groundtruth-json", "-g", required=True,
                        type=Path, help="Original ground-truth JSON file.")
    extended = parser.add_mutually_exclusive_group()
    extended.add_argument(
        "--extended-groundtruth", "-e", type=Path,
        help="Extended mapping file; copied original mappings are excluded from additions.",
    )
    extended.add_argument(
        "--no-extended-groundtruth", action="store_true",
        help="Disable automatic discovery in the sibling groundtruth_extended directory.",
    )
    parser.add_argument("--sabo", "-s", required=True, type=Path, help="SABO JSON file.")
    parser.add_argument("--reports-dir", type=Path,
                        default=Path(__file__).resolve().parent / "reports")
    parser.add_argument("--output", "-o", type=Path, help="Optional report output path.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def ensure_mapping_entries(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        entries = document
    elif isinstance(document, dict) and isinstance(document.get("mappings"), list):
        entries = document["mappings"]
    else:
        raise ValueError("Ground truth must be a list or an object with a mappings list.")
    return [entry for entry in entries if isinstance(entry, dict)]


def get_sabo_nodes(document: Any) -> list[dict[str, Any]]:
    if not isinstance(document, dict):
        raise ValueError("SABO must be a JSON object.")
    nodes = document.get("nodes")
    if nodes is None and isinstance(document.get("elements"), dict):
        nodes = document["elements"].get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("SABO must contain a nodes list at nodes or elements.nodes.")
    return [node for node in nodes if isinstance(node, dict)]


def get_sabo_edges(document: dict[str, Any]) -> list[dict[str, Any]]:
    edges = document.get("edges")
    if edges is None:
        elements = document.get("elements", {})
        edges = elements.get("edges", []) if isinstance(elements, dict) else []
    if not isinstance(edges, list):
        raise ValueError("SABO edges must be a list.")
    return [edge for edge in edges if isinstance(edge, dict)]


def labels_from_data(data: dict[str, Any]) -> list[str]:
    labels = data.get("labels", [])
    if isinstance(labels, list):
        return [str(label) for label in labels]
    return [] if labels is None else [str(labels)]


def is_checked_node(data: dict[str, Any]) -> bool:
    return bool(CHECKED_LABELS_CASEFOLDED.intersection(
        label.casefold() for label in labels_from_data(data)
    ))


def secdfd_property(data: dict[str, Any], name: str) -> Any:
    if name in data:
        return data[name]
    properties = data.get("properties")
    return properties.get(name) if isinstance(properties, dict) else None


def normalize_secdfd_type(value: Any) -> str | None:
    return SECDFD_ALIASES.get(value.casefold()) if isinstance(value, str) else None


def groundtruth_type(entry: dict[str, Any]) -> str:
    normalized = normalize_secdfd_type(entry.get("secdfd_type"))
    if normalized is None:
        raise ValueError(f"Unrecognized ground-truth secdfd_type in mapping: {entry!r}")
    return normalized


def additional_groundtruth_entries(
    original: list[dict[str, Any]], extended: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # A copied mapping is still original, irrespective of auxiliary annotations.
    def key(entry: dict[str, Any]) -> str:
        return json.dumps([entry.get("sabo"), entry.get("secdfd"), groundtruth_type(entry)],
                          sort_keys=True)

    seen = {key(entry) for entry in original}
    additions = []
    for entry in extended:
        entry_key = key(entry)
        if entry_key not in seen:
            seen.add(entry_key)
            additions.append(entry)
    return additions


def build_groundtruth_index(
    entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groundtruth_type(entry)
        node_id = entry.get("sabo")
        if isinstance(node_id, str) and node_id:
            index[node_id].append(entry)
    return dict(index)


def build_secdfd_target_index(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
) -> dict[str, list[str]]:
    category_ids = set()
    for node in nodes:
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        node_id = data.get("id")
        if (isinstance(node_id, str) and node_id.startswith("secdfd:")
                and "category" in {label.casefold() for label in labels_from_data(data)}):
            category_ids.add(node_id)
    index: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        data = edge.get("data")
        if not isinstance(data, dict) or data.get("label") != "implements":
            continue
        source, target = data.get("source"), data.get("target")
        if (isinstance(source, str) and isinstance(target, str)
                and target in category_ids and target not in index[source]):
            index[source].append(target)
    return dict(index)


def assignment_outcome(primary: Any, types: Any) -> str:
    if (primary is None or primary == "") and (
        types is None or isinstance(types, list) and not types
    ):
        return "UNDEFINED"
    if (isinstance(primary, str) and primary != "" and isinstance(types, list)
            and types and primary in types):
        return "DEFINED_AND_CONSISTENT"
    return "INCONSISTENT"


def groundtruth_availability(original: set[str], extended: set[str]) -> str:
    if original:
        return "ORIGINAL_GROUND_TRUTH_AVAILABLE"
    return "EXTENDED_GROUND_TRUTH_ONLY" if extended else "GROUND_TRUTH_UNAVAILABLE"


def evaluate_outcomes(
    assignment: str, label: Any, original: set[str], extended: set[str],
) -> dict[str, str | None]:
    outcomes: dict[str, str | None] = {
        "assignmentAndConsistency": assignment,
        "validity": None,
        "determination": None,
        "groundTruthAvailability": groundtruth_availability(original, extended),
        "mappingCorrectness": None,
    }
    if assignment != "DEFINED_AND_CONSISTENT":
        return outcomes
    normalized = normalize_secdfd_type(label)
    if normalized is None:
        outcomes["validity"] = "INVALID"
        return outcomes
    outcomes["validity"] = "VALID" if label in VALID_SECDFD_TYPES else "ALMOST_VALID"
    if normalized == "Undetermined":
        outcomes["determination"] = "UNDETERMINED"
        return outcomes
    outcomes["determination"] = "DETERMINED"
    if normalized in original:
        correctness = "CORRECT"
    elif normalized in extended:
        correctness = "PROBABLY_CORRECT"
    elif original or extended:
        correctness = "INCORRECT"
    else:
        correctness = "UNASSESSED"
    outcomes["mappingCorrectness"] = correctness
    return outcomes


def checked_node_report(
    nodes: list[dict[str, Any]],
    original_index: dict[str, list[dict[str, Any]]],
    extended_index: dict[str, list[dict[str, Any]]],
    target_index: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows = []
    for node in nodes:
        data = node.get("data")
        if not isinstance(data, dict) or not is_checked_node(data):
            continue
        node_id = data.get("id")
        primary = secdfd_property(data, "primarySecdfdType")
        types = secdfd_property(data, "secdfdTypes")
        assignment = assignment_outcome(primary, types)
        original_entries = original_index.get(node_id, [])
        extended_entries = extended_index.get(node_id, [])
        original = {groundtruth_type(entry) for entry in original_entries}
        extended = {groundtruth_type(entry) for entry in extended_entries}
        targets = target_index.get(node_id, [])
        type_entries = enumerate(types) if isinstance(types, list) and types else [(None, None)]
        for type_index, label in type_entries:
            outcomes = evaluate_outcomes(assignment, label, original, extended)
            conclusions = [outcome for outcome in outcomes.values() if outcome is not None]
            expected_target = f"secdfd:{label}" if isinstance(label, str) else None
            row = {
                "id": node_id,
                "labels": labels_from_data(data),
                "primarySecdfdType": primary,
                "secdfdTypes": types,
                "secdfdType": label,
                "secdfdTypeIndex": type_index,
                "normalizedSecdfdType": (
                    normalize_secdfd_type(label)
                    if assignment == "DEFINED_AND_CONSISTENT" else None
                ),
                "implementsSecdfdTargets": targets,
                "expectedSecdfdTarget": expected_target,
                "hasImplementsEdge": expected_target in targets,
                "originalGroundTruthSecdfdTypes": sorted(original),
                "extendedGroundTruthSecdfdTypes": sorted(extended),
                "groundTruthSecdfdTypes": sorted(original | extended),
                "originalGroundTruthEntryCount": len(original_entries),
                "extendedGroundTruthEntryCount": len(extended_entries),
                "groundTruthEntryCount": len(original_entries) + len(extended_entries),
                "outcomes": outcomes,
                "conclusion": " and ".join(conclusions),
                "conclusions": conclusions,
            }
            if outcomes["validity"] == "INVALID":
                row["invalidSecdfdType"] = label
            elif outcomes["validity"] == "ALMOST_VALID":
                row["almostValidSecdfdType"] = label
            rows.append(row)
    return rows


def summarize_outcomes(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary = {}
    for dimension, labels in DIMENSION_OUTCOMES.items():
        counts = Counter(row["outcomes"][dimension] for row in rows)
        summary[dimension] = {label: counts[label] for label in labels}
        summary[dimension]["NOT_EVALUATED"] = counts[None]
    return summary


def validate_evaluation_counts(
    rows: list[dict[str, Any]], outcomes: dict[str, dict[str, int]],
) -> None:
    """Reconcile eligible label occurrences with validity and determination counts."""
    classified_nodes = {}
    for row in rows:
        if row["outcomes"]["assignmentAndConsistency"] == "DEFINED_AND_CONSISTENT":
            classified_nodes.setdefault((row.get("sourceReport"), row["id"]), row)
    labels = sum(len(row["secdfdTypes"]) for row in classified_nodes.values())
    validity = outcomes["validity"]
    valid, almost_valid, invalid = (
        validity["VALID"], validity["ALMOST_VALID"], validity["INVALID"]
    )
    if labels != valid + almost_valid + invalid:
        raise ValueError(
            "Evaluation count check failed: Labels = Valid + Almost valid + Invalid; "
            f"Labels={labels}, Valid={valid}, Almost valid={almost_valid}, Invalid={invalid}"
        )
    determination = outcomes["determination"]
    determined, undetermined = determination["DETERMINED"], determination["UNDETERMINED"]
    if determined + undetermined != valid + almost_valid:
        raise ValueError(
            "Evaluation count check failed: Determined + Undetermined = Valid + Almost valid; "
            f"Determined={determined}, Undetermined={undetermined}, "
            f"Valid={valid}, Almost valid={almost_valid}"
        )


def summarize_node_outcomes(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    nodes = {}
    for row in rows:
        nodes.setdefault((row.get("sourceReport"), row["id"]), row)
    summary = summarize_outcomes(list(nodes.values()))
    return {dimension: summary[dimension] for dimension in NODE_DIMENSIONS}


def summarize_conclusions(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(label for row in rows for label in row["conclusions"])
    return {label: counts[label] for label in CONCLUSIONS}


def find_unmatched_groundtruth_entries(
    entries: list[dict[str, Any]], nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    node_ids = {
        data["id"] for node in nodes
        if isinstance((data := node.get("data")), dict) and isinstance(data.get("id"), str)
    }
    return [entry for entry in entries
            if not isinstance(entry.get("sabo"), str) or entry["sabo"] not in node_ids]


def default_output_path(groundtruth_path: Path, sabo_path: Path, reports_dir: Path) -> Path:
    def safe_part(path: Path) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "report"
    return reports_dir / f"{safe_part(sabo_path)}__{safe_part(groundtruth_path)}__secdfd_type_report.json"


def generate_report(
    groundtruth_path: Path, sabo_path: Path, extended_groundtruth_path: Path | None = None,
) -> dict[str, Any]:
    original = ensure_mapping_entries(load_json(groundtruth_path))
    extended_input = (
        ensure_mapping_entries(load_json(extended_groundtruth_path))
        if extended_groundtruth_path is not None else []
    )
    extended = additional_groundtruth_entries(original, extended_input)
    sabo_json = load_json(sabo_path)
    nodes = get_sabo_nodes(sabo_json)
    original_index = build_groundtruth_index(original)
    extended_index = build_groundtruth_index(extended)
    rows = checked_node_report(
        nodes, original_index, extended_index,
        build_secdfd_target_index(nodes, get_sabo_edges(sabo_json)),
    )
    outcomes = summarize_outcomes(rows)
    validate_evaluation_counts(rows, outcomes)
    unmatched_original = find_unmatched_groundtruth_entries(original, nodes)
    unmatched_extended = find_unmatched_groundtruth_entries(extended, nodes)
    unmatched = [
        {**entry, "groundtruthSource": source}
        for source, entries in (("original", unmatched_original), ("extended", unmatched_extended))
        for entry in entries
    ]
    return {
        "evaluationVersion": EVALUATION_VERSION,
        "groundtruth": str(groundtruth_path),
        "extendedGroundtruth": str(extended_groundtruth_path) if extended_groundtruth_path else None,
        "sabo": str(sabo_path),
        "checkedLabels": sorted(CHECKED_LABELS),
        "summary": {
            "totalGroundtruthEntries": len(original) + len(extended),
            "totalOriginalGroundtruthEntries": len(original),
            "totalExtendedGroundtruthEntries": len(extended),
            "totalExtendedGroundtruthInputEntries": len(extended_input),
            "groundtruthEntriesWithSabo": sum(
                len(entries) for index in (original_index, extended_index) for entries in index.values()
            ),
            "totalSaboNodes": len(nodes),
            "totalCheckedNodes": sum(
                is_checked_node(data) for node in nodes
                if isinstance((data := node.get("data")), dict)
            ),
            "totalReportEntries": len(rows),
            "outcomes": outcomes,
            "nodeOutcomes": summarize_node_outcomes(rows),
            "conclusions": summarize_conclusions(rows),
            "conclusionCombinations": dict(Counter(row["conclusion"] for row in rows)),
            "unmatchedGroundtruthEntryCount": len(unmatched),
            "unmatchedOriginalGroundtruthEntryCount": len(unmatched_original),
            "unmatchedExtendedGroundtruthEntryCount": len(unmatched_extended),
            "unmatchedGroundtruthEntries": unmatched,
        },
        "checkedNodes": rows,
    }


def main() -> None:
    args = parse_args()
    groundtruth_path, sabo_path = args.groundtruth.resolve(), args.sabo.resolve()
    extended_path = args.extended_groundtruth
    if extended_path is None and not args.no_extended_groundtruth:
        candidate = groundtruth_path.parent.parent / "groundtruth_extended" / groundtruth_path.name
        if candidate.is_file():
            extended_path = candidate
    if extended_path is not None:
        extended_path = extended_path.resolve()
    report = generate_report(groundtruth_path, sabo_path, extended_path)
    output_path = (args.output or default_output_path(
        groundtruth_path, sabo_path, args.reports_dir,
    )).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
        output_file.write("\n")
    print(f"Wrote report: {output_path}")
    print(f"Checked nodes: {report['summary']['totalCheckedNodes']}")
    print(f"Report entries: {report['summary']['totalReportEntries']}")
    print(f"Additional extended mappings: {report['summary']['totalExtendedGroundtruthEntries']}")


if __name__ == "__main__":
    main()
