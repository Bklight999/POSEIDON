import os
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from stemgen import package_tasks, reference_gen


class QualityClosureTests(unittest.TestCase):
    def _eval_dir(self, source):
        td = tempfile.TemporaryDirectory()
        with open(os.path.join(td.name, "eval_script.py"), "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(source))
        self.addCleanup(td.cleanup)
        return td.name

    def test_hard_negative_gate_rejects_always_true_evaluator(self):
        tdir = self._eval_dir("""
            def evaluate(pred, gold):
                return True, "always true"
        """)
        ok, msg = package_tasks._verify_eval_hard_negatives(
            tdir, 10, {"type": "count", "gold": 10})
        self.assertFalse(ok)
        self.assertIn("hard-negative accepted", msg)

    def test_hard_negative_gate_accepts_exact_numeric_evaluator(self):
        tdir = self._eval_dir("""
            import ast
            def _p(x):
                try: return ast.literal_eval(x) if isinstance(x, str) else x
                except Exception: return x
            def evaluate(pred, gold):
                pred, gold = _p(pred), _p(gold)
                try: return (float(pred) == float(gold), "exact")
                except Exception: return (False, "invalid")
        """)
        positive, _ = package_tasks._verify_eval_script(tdir, 10, 10, llm_stub=True)
        negative, msg = package_tasks._verify_eval_hard_negatives(
            tdir, 10, {"type": "count", "gold": 10})
        self.assertTrue(positive)
        self.assertTrue(negative, msg)

    def test_hard_negative_gate_catches_partial_composite(self):
        tdir = self._eval_dir("""
            import ast
            def _p(x):
                try: return ast.literal_eval(x) if isinstance(x, str) else x
                except Exception: return x
            def evaluate(pred, gold):
                pred, gold = _p(pred), _p(gold)
                # Bug: accepts any one matching field.
                if not isinstance(pred, dict) or not isinstance(gold, dict): return False, "bad"
                return any(k in pred and pred[k] == v for k, v in gold.items()), "partial"
        """)
        gold = {"a": 1, "b": 2}
        ok, msg = package_tasks._verify_eval_hard_negatives(
            tdir, gold, {"type": "composite", "gold": gold})
        self.assertFalse(ok)
        self.assertIn("composite_missing_field", msg)

    def test_lineage_records_hashes_and_diff(self):
        rec = reference_gen._lineage_record(
            "x = 0.05\ny = x + 1", "x = 0.05", "x = 0.05\noutputs={'z': x}",
            "x = 0.05\noutputs={'z': x * 2}")
        self.assertEqual(len(rec["source_program_sha256"]), 64)
        self.assertTrue(rec["used_source_code_ref"])
        self.assertGreater(rec["base_to_gold_diff"]["added_lines"], 0)

    def _bp(self):
        return SimpleNamespace(
            task_id="repair_test", prompt="compute x", target_quantity_key="x",
            answer_type="scalar", target_estimand={}, composite_parts=[],
            task_type="repair", given_step=0, source_code_ref="outputs={'x': 1}",
            load_bearing_point="normalization", wrong_methods=["skip normalization"],
            mutation={"category": "missing_step", "violated_assumption": "normalization required"},
            removed_step={})

    def test_repair_requires_runnable_separated_buggy_output(self):
        adapted = {"ok": True, "outputs": {"x": 1}, "code": "outputs={'x': 1}", "iters": 1}
        run_results = [
            {"ok": True, "outputs": {"x": 1}},  # determinism run
            {"ok": True, "outputs": {"x": 2}},  # buggy run
        ]
        with patch.object(reference_gen.llm_gen, "generate_with_repair", return_value=adapted), \
             patch.object(reference_gen.llm_gen, "generate", return_value="```python\noutputs={'x': 2}\n```"), \
             patch.object(reference_gen, "_run", side_effect=run_results):
            result = reference_gen.build_reference_single(
                SimpleNamespace(decision_points=[]), self._bp(), "outputs={'x': 1}",
                ".", [], base_program="outputs={'x': 1}")
        self.assertTrue(result["ok"])
        self.assertTrue(result["buggy_separation_verified"])
        self.assertEqual(result["buggy_answer"], 2)
        self.assertIn("lineage", result)

    def test_repair_composite_requires_any_part_to_change(self):
        bp = self._bp()
        bp.answer_type = "composite"
        bp.composite_parts = [{"key": "a"}, {"key": "b"}]
        bp.target_quantity_key = "a"
        adapted = {"ok": True, "outputs": {"a": 1, "b": 2},
                   "code": "outputs={'a': 1, 'b': 2}", "iters": 1}
        run_results = [
            {"ok": True, "outputs": {"a": 1, "b": 2}},
            {"ok": True, "outputs": {"a": 1, "b": 3}},
        ]
        with patch.object(reference_gen.llm_gen, "generate_with_repair", return_value=adapted), \
             patch.object(reference_gen.llm_gen, "generate", return_value="```python\noutputs={'a': 1, 'b': 3}\n```"), \
             patch.object(reference_gen, "_run", side_effect=run_results):
            result = reference_gen.build_reference_single(
                SimpleNamespace(decision_points=[]), bp, "outputs={'a': 1, 'b': 2}",
                ".", [], base_program="outputs={'a': 1, 'b': 2}")
        self.assertTrue(result["ok"])
        self.assertEqual(result["buggy_answer"], {"a": 1, "b": 3})

    def test_repair_rejects_noop_buggy_output(self):
        adapted = {"ok": True, "outputs": {"x": 1}, "code": "outputs={'x': 1}", "iters": 1}
        run_results = [
            {"ok": True, "outputs": {"x": 1}},
            {"ok": True, "outputs": {"x": 1}},
        ]
        with patch.object(reference_gen.llm_gen, "generate_with_repair", return_value=adapted), \
             patch.object(reference_gen.llm_gen, "generate", return_value="```python\noutputs={'x': 1}\n```"), \
             patch.object(reference_gen, "_run", side_effect=run_results):
            result = reference_gen.build_reference_single(
                SimpleNamespace(decision_points=[]), self._bp(), "outputs={'x': 1}",
                ".", [], base_program="outputs={'x': 1}")
        self.assertFalse(result["ok"])
        self.assertIn("buggy_not_separated", result["error"])


if __name__ == "__main__":
    unittest.main()

class SensitiveEgressTests(unittest.TestCase):
    def test_sensitive_text_is_sanitized_and_clean(self):
        from stemgen.safe_egress import has_sensitive, sanitize_and_assert
        cleaned = sanitize_and_assert(
            os.path.join(os.sep, "private_org", "private_user", "data.csv"), tag="unit")
        self.assertNotIn("private_org", cleaned.lower())
        self.assertNotIn("private_user", cleaned.lower())
        self.assertFalse(has_sensitive(cleaned))

    def test_llm_gen_sanitizes_before_post(self):
        from stemgen import llm_gen
        captured = {}
        def fake_post(messages, *args, **kwargs):
            captured["text"] = messages[0]["content"]
            return "ok"
        with patch.object(llm_gen, "_post", side_effect=fake_post):
            out = llm_gen.generate("path=" + os.path.join(os.sep, "private_org", "private_user", "a.csv"), retries=1, tag="unit")
        self.assertEqual(out, "ok")
        self.assertNotIn("private_org", captured["text"].lower())
        self.assertNotIn("private_user", captured["text"].lower())

    def test_safe_egress_fails_closed_without_desensitizer(self):
        from stemgen import safe_egress
        with patch.object(safe_egress, "_HAS_DESENS", False):
            with self.assertRaises(safe_egress.SensitiveEgressError):
                safe_egress.sanitize_and_assert("ordinary text", tag="unit")

    def test_all_llm_urlopen_files_reference_egress_guard(self):
        """Static regression: discover first-party LLM HTTP exits and require the hard guard."""
        root = os.path.dirname(os.path.dirname(__file__))
        expected = {
            "stemgen/llm_gen.py": "sanitize_messages",
            "stemgen/package_tasks.py": "sanitize_and_assert",
            "qc/run_qc_split.py": "sanitize_and_assert",
        }
        discovered = set()
        for base, dirs, files in os.walk(root):
            relbase = os.path.relpath(base, root).replace("\\", "/")
            dirs[:] = [d for d in dirs if d not in {
                "tasks_out", "workspace", "stage1_smoke", ".git", "__pycache__", "tests"}]
            if relbase.startswith("stem_web_deploy"):
                continue  # deployment HTTP API, not an LLM/model egress
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(base, fn)
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if "urlopen(" in text and any(x in text for x in
                    ("chat/completions", "OPENAI")):
                    discovered.add(os.path.relpath(path, root).replace("\\", "/"))
        self.assertEqual(discovered, set(expected),
                         f"Update guard coverage for changed LLM exits: {discovered ^ set(expected)}")
        for rel, guard in expected.items():
            with open(os.path.join(root, rel), encoding="utf-8") as f:
                text = f.read()
            self.assertIn(guard, text, rel)

    def test_indirect_gpt_agent_solver_has_hard_guard(self):
        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, "stemgen/agent_solver.py"), encoding="utf-8") as f:
            text = f.read()
        self.assertIn("sanitize_and_assert", text)
        self.assertIn("if gpt_mode", text)


class LocalRedactorTests(unittest.TestCase):
    def test_configured_terms_and_nested_metadata(self):
        from stemgen.safe_egress import sanitize_messages
        with patch.dict(os.environ, {"STEM_REDACT_TERMS": '["Example Person"]',
                                    "STEM_GW_KEY": "synthetic-secret-value"}):
            result = sanitize_messages([{"role": "user", "content": "Example Person",
                                        "metadata": {"key": "synthetic-secret-value"}}])
        self.assertEqual(result[0]["content"], "[REDACTED]")
        self.assertEqual(result[0]["metadata"]["key"], "[REDACTED]")

    def test_invalid_configuration_blocks_egress(self):
        from stemgen.safe_egress import sanitize_and_assert, SensitiveEgressError
        with patch.dict(os.environ, {"STEM_REDACT_TERMS": "not-json"}):
            with self.assertRaises(SensitiveEgressError):
                sanitize_and_assert("ordinary content")

    def test_failed_replacement_blocks_egress(self):
        from stemgen import safe_egress
        with patch.dict(os.environ, {"STEM_REDACT_TERMS": '["Example Person"]'}), \
             patch.object(safe_egress, "desensitize", side_effect=lambda value: value):
            with self.assertRaises(safe_egress.SensitiveEgressError):
                safe_egress.sanitize_and_assert("Example Person")


    def test_nested_quoted_windows_paths_preserve_json_structure(self):
        import json
        from stemgen.safe_egress import sanitize_messages, sanitize_and_assert
        windows_path = "C:" + chr(92) + "private" + chr(92) + "report.csv"
        quoted = 'Read "' + windows_path + '" and keep "quoted text".'
        messages = [{"role": "user", "content": quoted,
                     "metadata": {windows_path: [quoted, True, None, 3, 1.5]}}]
        expected_text = 'Read "[REDACTED]" and keep "quoted text".'
        expected = [{"role": "user", "content": expected_text,
                     "metadata": {"[REDACTED]": [expected_text, True, None, 3, 1.5]}}]
        self.assertEqual(sanitize_messages(messages), expected)
        self.assertEqual(json.loads(sanitize_and_assert(messages)), expected)
        self.assertEqual(messages[0]["content"], quoted)

    def test_redacted_key_collisions_are_rejected(self):
        from stemgen.safe_egress import sanitize_messages, SensitiveEgressError
        with patch.dict(os.environ, {"STEM_REDACT_TERMS": '["Person One", "Person Two"]'}):
            with self.assertRaises(SensitiveEgressError):
                sanitize_messages([{"metadata": {"Person One": 1, "Person Two": 2}}])
