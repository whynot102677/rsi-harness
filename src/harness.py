"""Fixed and versioned Harness execution using the same MBPP scoring path."""

import time
from dataclasses import asdict

from .api_client import APIProblem, prompt_hash
from .sandbox_runner import SandboxProblem
from .smoke_test import extract_source, prompt_for

MODEL_FAILURES = {"syntax_error", "runtime_error", "assertion_failure", "timeout", "code_extraction_failure"}
INFRA_FAILURES = {"sandbox_initialization_failure", "sandbox_protocol_error"}


def task_prompt(task, harness):
    rules = "\n".join(f"- {rule}" for rule in harness["rules"] + harness["skills"])
    return f"{prompt_for(task)}\n\nRules:\n{rules}"


def ask(client, prompt, *, purpose, task_id=None):
    replay = client.log.replay(purpose, task_id, prompt)
    if replay:
        return replay
    if getattr(client.log, "phase", None) != "p3":
        return client.complete(prompt, purpose=purpose, task_id=task_id)
    events = client.log.events()
    attempts = {
        event["attempt"] for event in events
        if event["event"] == "attempt" and event["purpose"] == purpose and event["task_id"] == task_id
    }
    terminal = any(
        event["event"] == "outcome" and event["attempt"] in attempts and (
            event["status"] == "invalid_response" or event.get("http_status") in (400, 401, 402, 403, 404, 422)
        ) for event in events
    )
    remaining = client.config["retry_limit"] + 1 - len(attempts)
    if remaining <= 0 or terminal:
        raise APIProblem("Recorded API attempts for this generation are exhausted")
    return client.complete(prompt, purpose=purpose, task_id=task_id, max_attempts=remaining)


def generate(task, prompt, purpose, client, sandbox, traces):
    trace_key = f"{purpose}:{task['task_id']}"
    existing = traces.get(trace_key)
    if existing and existing["prompt_sha256"] == prompt_hash(prompt) and existing["failure_type"] not in INFRA_FAILURES:
        return existing

    started = time.monotonic()
    completion = ask(client, prompt, purpose=purpose, task_id=task["task_id"])
    row = asdict(completion) | {
        "task_id": task["task_id"], "purpose": purpose, "prompt_sha256": prompt_hash(prompt),
        "tests_total": len(task["test_list"]), "docker_image_id": sandbox.image_id,
    }
    try:
        source = extract_source(completion.text)
    except ValueError:
        row.update(execution_status="failed", failure_type="code_extraction_failure",
                   tests_passed=None, execution_time=0)
    else:
        row.update(asdict(sandbox.evaluate(source, task["test_setup_code"], task["test_list"])))
    row["generation_and_execution_time"] = round(time.monotonic() - started, 3)
    traces.put(trace_key, row)
    if row["failure_type"] in INFRA_FAILURES:
        raise SandboxProblem(f"Docker evaluation stopped ({row['failure_type']})")
    return row


def repair(task, base_prompt, initial, purpose, client, sandbox, traces):
    if initial["failure_type"] not in MODEL_FAILURES:
        return []
    passed = initial["tests_passed"]
    test_count = f" {passed}/{initial['tests_total']} checks passed." if passed is not None else ""
    prompt = (
        f"{base_prompt}\n\nRevise your previous Python solution. "
        f"Execution feedback: {initial['failure_type']}.{test_count}\n"
        f"Previous solution:\n{initial['text']}"
    )
    return [generate(task, prompt, purpose, client, sandbox, traces)]


def evaluate_fixed(task, harness, purpose, client, sandbox, traces):
    prompt = task_prompt(task, harness)
    first = generate(task, prompt, f"{purpose}/initial", client, sandbox, traces)
    return [first, *repair(task, prompt, first, f"{purpose}/repair", client, sandbox, traces)]


def summarize(trials, expected):
    count = len(trials)
    first_passes = sum(steps[0]["execution_status"] == "passed" for steps in trials.values())
    final_passes = sum(steps[-1]["execution_status"] == "passed" for steps in trials.values())
    repairs = sum(len(steps) == 2 for steps in trials.values())
    repaired = sum(len(steps) == 2 and steps[-1]["execution_status"] == "passed" for steps in trials.values())
    steps = [step for trial in trials.values() for step in trial]
    inputs = sum(step["input_tokens"] for step in steps)
    outputs = sum(step["output_tokens"] for step in steps)
    elapsed = sum(step["api_time"] + step["execution_time"] for step in steps)
    return {
        "completed": count, "expected": expected,
        "first_passes": first_passes, "final_passes": final_passes,
        "first_pass_rate": first_passes / count if count else None,
        "final_pass_rate": final_passes / count if count else None,
        "repairs": repairs, "successful_repairs": repaired,
        "repair_success_rate": repaired / repairs if repairs else None,
        "model_calls": len(steps), "average_model_calls": len(steps) / count if count else None,
        "input_tokens": inputs, "output_tokens": outputs,
        "average_input_tokens": inputs / count if count else None,
        "average_output_tokens": outputs / count if count else None,
        "model_and_docker_seconds": round(elapsed, 3),
        "average_model_and_docker_seconds": round(elapsed / count, 3) if count else None,
    }
