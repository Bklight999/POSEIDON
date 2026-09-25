"""POSEIDON scientific-task generation package. Modules cover workflow grouping, environment preparation, executable seed adaptation, structured task specifications, task derivation, executed references, evaluators, controlled subprocess execution, optional agent difficulty grading, orchestration, packaging, LLM requests, and outbound-content checks. Modules can be imported and tested independently."""
__all__ = [
    "seed_grouping", "env_setup", "adapt", "structure", "taskspec", "derive",
    "reference_gen", "eval_gen", "sandbox",
    "difficulty_gate", "agent_solver", "difficulty_pass",
    "llm_gen", "safe_egress", "orchestrator", "package_tasks",
]
