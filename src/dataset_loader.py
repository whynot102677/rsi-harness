"""Load the pinned Google Research MBPP Full data and reproduce the P0 split."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "f82046ba5aabbbb427dbfd38a254d26bff08b533"
SOURCE = f"https://raw.githubusercontent.com/google-research/google-research/{REVISION}/mbpp/mbpp.jsonl"
SHA256 = "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"
RAW = ROOT / "data/raw/mbpp.jsonl"
MANIFEST = ROOT / "data/splits/manifest.json"
SEED = 42
SPLITS = {
    "experience": (601, 974, 30),
    "validation": (511, 600, 10),
    "test": (11, 510, 40),
}


def load_tasks(path=RAW):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("MBPP SHA-256 differs from the pinned Full dataset")
    rows = [json.loads(line) for line in data.splitlines()]
    tasks = {row["task_id"]: row for row in rows}
    if len(rows) != 974 or set(tasks) != set(range(1, 975)):
        raise ValueError("MBPP must contain one row for each ID 1–974")
    for row in rows:
        if not row["text"].strip() or not isinstance(row["test_setup_code"], str):
            raise ValueError("MBPP task text or setup is invalid")
        if len(row["test_list"]) != 3 or not all(
            isinstance(test, str) and test.strip() for test in row["test_list"]
        ):
            raise ValueError("Each MBPP task must have three standard tests")
    return tasks


def load_selected(task_ids, path=RAW):
    """Verify the dataset and deserialize only explicitly authorized task rows."""
    wanted = set(task_ids)
    tasks = {}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            match = re.search(rb'(?:^\{|,)\s*"task_id"\s*:\s*(\d+)\s*[,}]', line)
            if match and (task_id := int(match[1])) in wanted:
                tasks[task_id] = json.loads(line)
    if digest.hexdigest() != SHA256 or tasks.keys() != wanted:
        raise ValueError("Pinned MBPP data or requested IDs failed verification")
    return tasks


def build_manifest(tasks):
    rng = random.Random(SEED)
    draws = {
        name: rng.sample([task_id for task_id in range(first, last + 1) if task_id in tasks], count)
        for name, (first, last, count) in SPLITS.items()
    }
    ids = [task_id for draw in draws.values() for task_id in draw]
    if len(set(ids)) != len(ids) or min(ids) < 11:
        raise ValueError("Overlapping split IDs or reserved IDs")
    return {
        "dataset": "Google Research MBPP Full",
        "source": SOURCE,
        "revision": REVISION,
        "sha256": SHA256,
        "task_count": len(tasks),
        "seed": SEED,
        "sampling": "random.Random.sample on ascending IDs, in experience/validation/test order",
        "splits": {name: sorted(draw) for name, draw in draws.items()},
        "smoke_task_ids": draws["experience"][:3],
        "integrity": {
            "sha256_verified": True,
            "unique_ids_1_to_974": True,
            "three_standard_tests_per_task": True,
            "no_split_overlap": True,
            "reserved_ids_excluded": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="fetch the pinned official file")
    args = parser.parse_args()
    if args.download:
        with urlopen(SOURCE, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            parser.exit(1, "Downloaded MBPP failed SHA-256 verification\n")
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_bytes(data)
    tasks = load_tasks()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(build_manifest(tasks), indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(tasks)} tasks and wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
