"""Honest lexical-proxy reports and a separate blind grounding-review worksheet."""
import json
from pathlib import Path
import random

from .reporting import write_json, tier_requirement, tier_test_status, tier_retained_score


def write_blind_review(summary, examples, output, count=30):
    output = Path(output)
    artifacts = [output / name for name in ('blind-review.jsonl', 'blind-review-key.json',
                                           'blind-review-instructions.txt')]
    if any(path.exists() for path in artifacts):
        if not all(path.is_file() for path in artifacts):
            raise ValueError('Incomplete blind review artifacts; preserve existing ratings and repair explicitly')
        return  # Cached finalization must never erase a reviewer's ratings.
    rng = random.Random(summary["config"]["seed"] + 911)
    names = list(summary["final_test"])
    rng.shuffle(names)
    aliases = {f"response_{i + 1}": name for i, name in enumerate(names)}
    by_key = {name: {r["key"]: r for r in result["metrics"]["predictions"]}
              for name, result in summary["final_test"].items()}
    ordered = list(examples)
    rng.shuffle(ordered)
    seen, selected = set(), []
    for ex in ordered:
        group = ex.group_id
        if group in seen:
            continue
        seen.add(group)
        selected.append(ex)
        if len(selected) >= count:
            break
    rows = [{"key": ex.key, "document_group": ex.group_id, "prompt_with_support_context": ex.prompt,
             "reference_reply": ex.answer,
             "responses": {alias: by_key[name][ex.key]["prediction"] for alias, name in aliases.items()},
             "review": {alias: {"supported_by_context": None, "addresses_request": None,
                                 "fabricates_action_or_policy": None, "notes": ""} for alias in aliases}}
            for ex in selected]
    (output / "blind-review.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(output / "blind-review-key.json", aliases)
    (output / "blind-review-instructions.txt").write_text(
        "Rate the anonymized replies using the supplied documentation and conversation.\n"
        "For every reply record whether its claims are supported, whether it answers the request, "
        "and whether it fabricates a policy, fact, or completed action.\n"
        "Accept correct paraphrases even if they differ from the reference. Cite supporting or contradictory context in notes.\n"
        "Do not consult blind-review-key.json until ratings are frozen. These cases are already held-out results; "
        "review must not trigger checkpoint reselection. Ratings are blank until a reviewer completes them.\n")


def render_support_report(summary, timings=None):
    selection = summary["selection"]
    reference = selection["reference_id"]
    candidates = {c["id"]: c for c in selection["candidates"]}
    relative = any(tier.get("mode") == "relative_retention" for tier in selection["tiers"])
    requirement_note = ("Requirements are fractions of baseline token F1: 95% retention means baseline F1 multiplied by 0.95. "
                        "This is not 95% accuracy or a five-point F1 drop. " if relative else
                        "Tolerances are absolute token-F1 points on a 0–100 scale. ")
    counts = ", ".join(f"{name}: {v['count']}" for name, v in summary["dataset"].items())
    lines = ["# Document-grounded support response experiment", "",
             f"Model: `{summary['config']['model']}`. Examples: {counts}.", "",
             "Source: [Doc2Dial](https://doc2dial.github.io/data.html), crowdsourced public-service support dialogues. "
             "The provided supporting sections come from gold grounding annotations. This evaluates response writing with oracle evidence, not retrieval or end-to-end support resolution.", "",
             "**Token F1 and ROUGE-L measure reference-word overlap, not factual correctness, policy compliance, or helpfulness.** "
             "A correct paraphrase can score low; a contradiction can score high. No customer-support accuracy percentage is established. "
             "Blind review worksheets are provided separately; they have not been graded automatically.", "",
             f"Selection reference: **{reference}**. " + requirement_note +
             "Selection is frozen on development scores before final testing. All predeclared candidates are evaluated so equal-size allocation methods can be compared.", "",
             "| Model | Parameters removed | Development token F1 | Test token F1 | Test ROUGE-L | Test exact match | Test replies reaching token cap |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in summary["final_test"].items():
        record = candidates.get(name)
        dev = record["dev"] if record else selection["baseline"]["dev"]
        m = result["metrics"]
        reduction = record["actual_reduction"] if record else 0.0
        cap = f"{100 * m['generation_limit_fraction']:.2f}%" if 'generation_limit_fraction' in m else 'Unavailable'
        lines.append(f"| {name} | {100 * reduction:.2f}% | {100 * dev['token_f1']:.2f} | "
                     f"{100 * m['token_f1']:.2f} | {100 * m['rouge_l']:.2f} | {100 * m['exact_match']:.2f}% | {cap} |")
    lines += ["", "Reaching the generation token cap does not confirm truncation: EOS can occur on the final token.", "",
              "## Frozen compression tiers", "",
              "| Quality requirement | Checkpoint | Test F1 delta vs reference | Test F1 retained | Threshold met |",
              "| --- | --- | ---: | ---: | --- |"]
    for tier in summary["tiers"]:
        if tier["candidate_id"] is None:
            lines.append(f"| {tier_requirement(tier)} | No qualifying model | — | — | — |")
        else:
            lines.append(f"| {tier_requirement(tier)} | {tier['candidate_id']} | {tier['test']['delta_points']:+.2f} | "
                         f"{tier_retained_score(tier)} | {tier_test_status(tier)} |")
    lines += ["", "Repeated selections may refer to one checkpoint. Threshold success refers only to the lexical metric. "
              "Relative retention is checked against the reference's held-out test F1, not development F1; an observed pass is not a statistical guarantee. "
              "Selection finds the smallest qualifying tested candidate, not a global optimum. "
              "Intervals in summary.json resample document groups to account for correlated conversations; they are descriptive.", "",
              "## Checkpoints", ""]
    for c in selection["candidates"]:
        if c["status"] != "complete":
            lines.append(f"- {c['id']}: failed — {c.get('error', 'unknown error')}.")
        else:
            lines.append(f"- {c['id']}: `{c['checkpoint']}`; {c['parameters']:,} parameters, "
                         f"{c['checkpoint_bytes'] / 2**20:.1f} MiB on disk; selected recovery update {c['recovery']['best_step']}.")
    lines += ["", "Use the project loader for heterogeneous checkpoints. See data-provenance.json for source hashes, filters, "
              "document grouping and overlap checks. These are custom document-disjoint subsets, not official leaderboard scores. "
              "Short reference/section length filters limit the task to responses that fit the local model's configured context.", ""]
    if timings:
        lines += ["## Local timing", "", "| Model | Median mean response latency | Tokens per run |", "| --- | ---: | --- |"]
        for name, m in timings["models"].items():
            lines.append(f"| {name} | {m['median_mean_example_ms']:.2f} ms | {m['generated_tokens_per_run']} |")
        lines += ["", "Output lengths differ, so response latency is not an isolated compute-speed measurement. "
                  "The separate fixed-token benchmark can control output length. Memory counters are snapshots, not peaks.", ""]
    return "\n".join(lines)
