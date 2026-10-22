"""Small contract checks for recovery targets and probability comparisons."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .probe import make_tasks
from .recovery_distribution import score_distribution


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class RecoveryTest(unittest.TestCase):
    def test_original_tokens_and_clean_controls(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plant = {"token_index": 2, "orig_token_id": 10,
                     "orig_token_text": " original", "orig_prob": 0.2,
                     "new_token_id": 11, "new_token_text": " planted"}
            cases = root / "cases.jsonl"
            write_jsonl(cases, [
                {"id": "case-one", "group": "science", "mode": "greedy", "kind": "one",
                 "prompt": "Explain this", "answer": "a planted answer",
                 "answer_token_ids": [3, 4, 11], "plants": [plant],
                 "gold_tokens": [" planted"]},
                {"id": "case-clean", "group": "science", "mode": "greedy",
                 "kind": "clean", "prompt": "Explain this", "answer": "an original answer",
                 "answer_token_ids": [3, 4, 10], "plants": [], "gold_tokens": []},
                {"id": "status", "kind": "status"},
            ])
            dolly = root / "dolly.jsonl"
            records = []
            for label in ("modified", "clean"):
                records.append({"split": "test", "label": label,
                                "prompt": f"Dolly {label}",
                                "messages": [{"role": "user", "content": "Question"},
                                             {"role": "assistant", "content": "Answer"}],
                                "modification": plant if label == "modified" else None,
                                "clean_response": "original answer"})
            write_jsonl(dolly, records)
            tasks = make_tasks("recovery", cases, [str(dolly)], 2, 0)
            self.assertEqual(len(tasks), 4)
            self.assertEqual([t["gold_tokens"] for t in tasks[:2]],
                             [[" original"], []])
            self.assertEqual(sorted(tuple(t["gold_tokens"]) for t in tasks[2:]),
                             [(), (" original",)])
            self.assertTrue(all(t["variant"] == "recovery" for t in tasks))

            for name, answer in (("base", "none"), ("checkpoint", "original")):
                write_jsonl(root / f"{name}.jsonl",
                            [{"id": task["id"], "answer": answer} for task in tasks])
            dist = [{"id": tasks[0]["id"], "plant_index": 0,
                     "group": "science", "mode": "greedy", "kind": "one",
                     "original": "original", "replacement": "planted",
                     "stored_original_prob": 0.2, "original_prob": 0.2,
                     "original_rank": 2,
                     "top_token": {"id": 9, "text": "likely", "prob": 0.3},
                     "top_10": [{"id": 9, "text": "likely", "prob": 0.3},
                                {"id": 10, "text": "original", "prob": 0.2}],
                     "top_100": [{"id": 9, "text": "likely", "prob": 0.3},
                                 {"id": 10, "text": "original", "prob": 0.2}]}]
            summary = score_distribution(tasks, dist, ["base", "checkpoint"], root)
            self.assertEqual(summary["base/science/greedy"]["recovered_rate"], 0)
            self.assertEqual(summary["checkpoint/science/greedy"]["recovered_rate"], 1)
            self.assertEqual(summary["checkpoint/science/greedy"]["mean_base_token_probability"], 0.2)
            scored = [json.loads(line) for line in
                      (root / "distribution-scored.jsonl").read_text().splitlines()]
            self.assertEqual(scored[1]["guess_top_100_rank"], 2)


if __name__ == "__main__":
    unittest.main()
