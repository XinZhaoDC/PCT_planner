#!/usr/bin/env python3
"""Compare refactored WHU outputs with current-code and activity baselines."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_txt(path):
    points = np.loadtxt(path, delimiter=",", dtype=np.float64)
    return np.atleast_2d(points)[:, :3]


def load_pcd(path):
    with Path(path).open("rb") as handle:
        header = {}
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PCD has no DATA entry: %s" % path)
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                data_offset = handle.tell()
                break

    if header["DATA"][0].lower() != "binary":
        raise ValueError("Expected binary PCD: %s" % path)
    fields = header["FIELDS"]
    sizes = [int(value) for value in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    type_map = {
        ("F", 4): "<f4", ("F", 8): "<f8",
        ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4",
        ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4",
    }
    names, formats, offsets = [], [], []
    byte_offset = 0
    for field, size, kind, count in zip(fields, sizes, types, counts):
        scalar = np.dtype(type_map[(kind, size)])
        names.append(field)
        formats.append(scalar if count == 1 else (scalar, (count,)))
        offsets.append(byte_offset)
        byte_offset += size * count
    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": byte_offset,
    })
    count = int(header.get("POINTS", header["WIDTH"])[0])
    records = np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(count,))
    return np.column_stack((records["x"], records["y"], records["z"])).astype(
        np.float64
    )


def max_abs(left, right):
    return np.max(np.abs(left - right), axis=0).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--activity-baseline", type=Path, required=True)
    parser.add_argument("--current-code-baseline", type=Path, required=True)
    args = parser.parse_args()

    report = {"routes": {}, "passed": True}
    expected_counts = [496, 541, 451, 379, 352]
    for route_id, expected_count in enumerate(expected_counts, start=1):
        stem = "waypoint_%d" % route_id
        generated_txt = load_txt(args.generated / (stem + ".txt"))
        generated_pcd = load_pcd(args.generated / (stem + ".pcd"))
        activity_txt = load_txt(args.activity_baseline / (stem + ".txt"))
        activity_pcd = load_pcd(args.activity_baseline / (stem + ".pcd"))
        current_txt = load_txt(args.current_code_baseline / (stem + ".txt"))
        current_pcd = load_pcd(args.current_code_baseline / (stem + ".pcd"))

        shapes_match = all(
            len(points) == expected_count
            for points in (
                generated_txt, generated_pcd, activity_txt, activity_pcd,
                current_txt, current_pcd,
            )
        )
        current_delta = max_abs(generated_txt, current_txt)
        activity_delta = generated_txt - activity_txt
        activity_abs_max = np.max(np.abs(activity_delta), axis=0).tolist()
        z_near_positive_06 = int(np.count_nonzero(np.isclose(
            activity_delta[:, 2], 0.6, atol=5e-6
        )))

        route_passed = (
            shapes_match
            and max(max_abs(generated_txt, generated_pcd)) <= 5e-6
            and max(max_abs(activity_txt, activity_pcd)) <= 5e-6
            and max(max_abs(current_txt, current_pcd)) <= 5e-6
            and max(current_delta) <= 1e-9
            and activity_abs_max[0] <= 1e-9
            and activity_abs_max[1] <= 1e-9
            and z_near_positive_06 == (468 if route_id == 1 else expected_count)
        )
        report["routes"][str(route_id)] = {
            "count": len(generated_txt),
            "expected_count": expected_count,
            "generated_txt_vs_pcd_max_abs_xyz": max_abs(generated_txt, generated_pcd),
            "generated_vs_current_code_max_abs_xyz": current_delta,
            "generated_vs_activity_max_abs_xyz": activity_abs_max,
            "activity_z_delta_near_positive_0_6_count": z_near_positive_06,
            "passed": route_passed,
        }
        report["passed"] = report["passed"] and route_passed

    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
