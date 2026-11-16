"""Finish experience-only optimization and freeze the formal evaluation protocol."""

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

import yaml

from .api_client import APIProblem, AttemptLog, DeepSeek, phase_usage
from .dataset_loader import MANIFEST, ROOT, build_manifest, load_selected
from .experiments import RESULTS, Traces, read_json, require_p0, validate, write_json
from .harness import ask, evaluate_fixed
from .sandbox_runner import Sandbox, SandboxProblem
from .smoke_test import atomic_write, extract_source, prompt_for

P25 = RESULTS / "p25"
FROZEN = ROOT / "configs/experiment_frozen.yaml"
HASH_FILE = FROZEN.with_suffix(".yaml.sha256")
PROTOCOL = {
    "candidate_attempts": 2,
    "candidate_selection_C": "first format-valid candidate in proposal order",
    "candidate_selection_D": "among gate-passing candidates maximize passes, then minimize tokens, then prefer earlier proposal",
    "validation_gate": {"minimum_gains": 1, "maximum_regressions": 0, "maximum_extra_tokens_percent": 25},
    "request_limit_p25_and_p3": 450,
    "cost_stop_cny": 20,
    "bootstrap": {"seed": 42, "resamples": 10000, "confidence": 0.95, "method": "paired percentile"},
    "evaluation": {
        "first_attempt": "all three standard MBPP assertions must pass",
        "final_attempt": "last scored attempt, with at most one model repair after failure",
        "denominator": "completed tasks for rates, with incomplete counts reported separately",
        "pairing": "Task ID, pairwise complete cases, final success",
        "reuse": "A/B share initial response; identical full Harness configurations share scores",
        "transport_retries": 1,
        "sandbox_retries": 1,
        "stopping": "finish all four task-method rows or stop on budget or persistent global infrastructure failure",
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    def render(stream):
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write(path, render)


def pilot_validation(ids):
    traces = read_json(RESULTS / "p2/traces.json")
    return {
        task_id: [
            traces[key] for stage in ("initial", "repair")
            if (key := f"p2/validation/H0/{stage}:{task_id}") in traces
        ]
        for task_id in ids
    }


def diagnostics(tasks, experience, sandbox):
    traces = Traces(P25 / "failure_trajectories.json")
    for task_id, steps in experience.items():
        task = tasks[task_id]
        name = re.search(r"Define a top-level function named (\w+)", prompt_for(task))[1]
        for number, step in enumerate(steps, 1):
            original = step["failure_type"]
            key = f"{task_id}:{number}"
            if not original or traces.get(key):
                continue
            detail = None
            if original in {"syntax_error", "code_extraction_failure"}:
                category = "syntax_or_format"
            elif original == "timeout":
                category = "runtime"
            else:
                detail = sandbox.evaluate(
                    extract_source(step["text"]), task["test_setup_code"], task["test_list"], diagnose_name=name,
                )
                if detail.execution_status == "blocked" or detail.failure_type == "sandbox_protocol_error":
                    raise SandboxProblem("Experience diagnostic sandbox failed")
                category = {
                    "signature": "signature", "runtime_error": "runtime", "boundary": "boundary",
                    "output": "output", "timeout": "runtime",
                }.get(detail.failure_type, "other")
                if original == "runtime_error" and detail.failure_type == "assertion_failure":
                    category = "other"
            traces.put(key, {
                "task_id": task_id, "step": number, "original_failure": original, "category": category,
                "original_tests_passed": step["tests_passed"],
                "diagnostic_failure": detail.failure_type if detail else None,
                "diagnostic_tests_passed": detail.tests_passed if detail else None,
                "source_record": step["purpose"], "attempt": step["attempt"],
            })
    counts = Counter(row["category"] for row in traces.rows.values())
    distinct = {
        category: {row["task_id"] for row in traces.rows.values() if row["category"] == category}
        for category in ("syntax_or_format", "signature", "runtime", "boundary", "output", "algorithm", "other")
    }
    write_csv(P25 / "failure_types.csv", [
        {"category": category, "failed_attempts": counts[category], "distinct_tasks": len(distinct[category])}
        for category in distinct
    ])
    return sorted(
        ((category, len(ids), counts[category]) for category, ids in distinct.items()
         if len(ids) >= 2 and category != "other"),
        key=lambda item: (-item[1], -item[2], item[0]),
    )


def experience_steps(tasks, manifest, h0, client, sandbox):
    prior = read_json(RESULTS / "p1/trials.json")["B"]
    steps = {int(key): rows for key, rows in prior.items()}
    for line in (RESULTS / "p0/smoke_results.jsonl").read_text().splitlines():
        row = json.loads(line)
        task_id = row["task_id"]
        source = RESULTS / f"p0/generated/task-{task_id}-attempt-{row['api_attempt']}.py"
        steps[task_id] = [{
            **row, "text": source.read_text(), "purpose": "p0/smoke", "attempt": row["api_attempt"],
        }]
    traces = Traces(P25 / "traces.json")
    for task_id in manifest["splits"]["experience"]:
        if task_id not in steps:
            steps[task_id] = evaluate_fixed(
                tasks[task_id], h0, "p25/experience/H0", client, sandbox, traces,
            )
            print(f"P2.5 experience {task_id}: {steps[task_id][-1]['execution_status']}", flush=True)
    write_json(P25 / "experience_trials.json", steps)
    return steps


def candidate_change(text, old_rule):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"rule"}:
        raise ValueError("Expected exactly one rule field")
    rule = value["rule"]
    if not isinstance(rule, str) or not re.fullmatch(r"[A-Za-z][A-Za-z ,'-]{9,139}\.", rule):
        raise ValueError("Rule outside the plain-text length or character limits")
    if ("function name" not in rule.lower() or "requested" not in rule.lower()
            or rule.casefold() == old_rule.casefold()):
        raise ValueError("Rule must preserve the task behavior and function-name requirements and make a change")
    return rule


def optimize(tasks, ids, h0, evidence, client, sandbox):
    traces = Traces(P25 / "traces.json")
    h0_scores = pilot_validation(ids)
    decisions = []
    seen = set()
    old_rule = h0["rules"][0]
    for index, (category, count, failed_attempts) in enumerate(evidence[:PROTOCOL["candidate_attempts"]], 1):
        version = f"R{index}"
        path = P25 / f"candidate_{index}.json"
        train = {"failure_category": category, "distinct_experience_tasks": count, "failed_attempts": failed_attempts}
        record = {"version": version, "parent": "H0", "training_evidence": train}
        prompt = (
            "Rewrite the first generic rule for a Python coding agent based on the aggregate failure evidence. "
            "Preserve the words requested and function name and their original meaning. Address only the "
            "reported category with a concrete generic check. Write one sentence under 140 characters, "
            "using only letters, spaces, commas, apostrophes, hyphens and a final period. "
            "Do not include tasks, identifiers, answers, examples or code. Return JSON with one field, rule.\n"
            f"Original rule: {old_rule}\nEvidence: {json.dumps(train, sort_keys=True)}"
        )
        completion = ask(client, prompt, purpose=f"p25/{version}/proposal")
        record["proposal_attempt"] = completion.attempt
        try:
            rule = candidate_change(completion.text, old_rule)
            if rule.casefold() in seen:
                raise ValueError("Duplicate proposed rule")
        except ValueError as error:
            record.update(state="format_rejected", reason=str(error))
            write_json(path, record)
            decisions.append(record)
            continue
        seen.add(rule.casefold())
        harness = {"rules": [rule, *h0["rules"][1:]], "skills": h0["skills"]}
        write_json(P25 / "versions" / f"{version}.json", harness)
        record.update(state="validating", harness=harness, diff={"rule_index": 0, "before": old_rule, "after": rule})
        write_json(path, record)
        scores = {}
        for task_id in ids:
            scores[task_id] = evaluate_fixed(
                tasks[task_id], harness, f"p25/validation/{version}", client, sandbox, traces,
            )
        decision = validate(h0_scores, scores, PROTOCOL["validation_gate"]["maximum_extra_tokens_percent"])
        reasons = []
        if not decision["gains"]:
            reasons.append("no_new_pass")
        if decision["regressions"]:
            reasons.append("regression")
        if decision["token_increase_percent"] > PROTOCOL["validation_gate"]["maximum_extra_tokens_percent"]:
            reasons.append("extra_tokens_above_limit")
        record.update(state="validated", decision=decision, reasons=reasons)
        write_json(path, record)
        decisions.append(record)
        print(f"P2.5 {version}: {decision['candidate']['final_passes']}/10, accepted={decision['accepted']}", flush=True)
    return decisions


def freeze_formal(manifest, model_config, image, model, h0, decisions):
    valid = [item for item in decisions if item["state"] == "validated"]
    accepted = [item for item in valid if item["decision"]["accepted"]]
    accepted.sort(key=lambda item: (
        -item["decision"]["candidate"]["final_passes"],
        item["decision"]["candidate"]["input_tokens"] + item["decision"]["candidate"]["output_tokens"],
        item["version"],
    ))
    naive = valid[0] if valid else {"version": "H0", "harness": h0}
    verified = accepted[0] if accepted else {"version": "H0", "harness": h0}
    frozen = {
        "protocol": PROTOCOL,
        "dataset": {"sha256": manifest["sha256"], "manifest_sha256": sha(MANIFEST), "seed": manifest["seed"],
                    "splits": manifest["splits"]},
        "inference": {key: value for key, value in model_config.items() if key != "api_attempt_limit"},
        "thinking": "disabled", "response_model": model, "docker_image_id": image,
        "groups": {
            "A": {"method": "Direct LLM", "version": "H0", "harness": h0, "max_repairs": 0},
            "B": {"method": "Fixed Harness", "version": "H0", "harness": h0, "max_repairs": 1},
            "C": {"method": "Naive RSI-Harness", "version": naive["version"], "harness": naive["harness"], "max_repairs": 1},
            "D": {"method": "Verified RSI-Harness", "version": verified["version"], "harness": verified["harness"], "max_repairs": 1},
        },
        "source_sha256": {path.name: sha(path) for path in sorted((ROOT / "src").glob("*.py"))},
    }
    atomic_write(FROZEN, lambda stream: stream.write(yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True)))
    atomic_write(HASH_FILE, lambda stream: stream.write(f"{sha(FROZEN)}  {FROZEN.name}\n"))
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("p25",))
    parser.parse_args()
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    manifest = read_json(MANIFEST)
    if build_manifest(range(1, 975)) != manifest:
        raise ValueError("Split manifest differs from the pinned MBPP allocation")
    model = require_p0(manifest)
    pilot = read_json(RESULTS / "experiment_config.json")
    config = pilot["inference_and_sandbox"]
    h0 = pilot["design"]["h0"]
    sandbox = Sandbox(config["docker_image"], config["execution_timeout"])
    image = sandbox.prepare()
    if image != pilot["docker_image_id"]:
        raise SandboxProblem("Docker image differs from the pilot")
    with (RESULTS / "experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if FROZEN.exists():
            raise ValueError("Formal configuration already frozen")
        P25.mkdir(parents=True, exist_ok=True)
        write_json(P25 / "protocol.json", PROTOCOL)
        ids = manifest["splits"]["validation"]
        tasks = load_selected(manifest["splits"]["experience"])
        log = AttemptLog(P25 / "api_attempts.jsonl", 450, phase="p25")
        client = DeepSeek(config, log, env_file=ROOT / ".env", expected_model=model)
        steps = experience_steps(tasks, manifest, h0, client, sandbox)
        evidence = diagnostics(tasks, steps, sandbox)
        tasks.update(load_selected(ids))
        decisions = optimize(tasks, ids, h0, evidence, client, sandbox)
        write_json(P25 / "decisions.json", decisions)
        # Freeze in a separate step after the formal runner is finalized and hashed.
        write_json(P25 / "freeze_inputs.json", {"image": image, "model": model, "decisions": decisions})
        write_json(P25 / "usage.json", phase_usage(RESULTS, formal=True)["p25"])


if __name__ == "__main__":
    main()
