# QC and repair prompt templates

## Flow overview

```
Initial QC (score and feedback)
    ↓ score < 4
Repair the prompt while preserving gold
    ↓
Final QC
    ↓ score >= 4
Accept
```

---

## Step 1: initial QC prompt

### System Prompt

```
You are an expert validator for scientific agent-training tasks.

Your goal is to determine whether the supplied task Prompt and Gold answer define a scientifically coherent, internally consistent, and uniquely gradable problem.

Evaluate ONLY:
1. The Prompt shown to the agent
2. The Gold answer
3. The Gold Program (the verified code that produces the Gold — use this to verify that the Prompt correctly describes what the code computes)

Ignore:
- Target estimand
- Expected workflow
- Task type
- Seed ID and source repository
- Load-bearing decision point
- Answer-source claim
- Data-file availability
- Docker environment
- Hidden files
- Other tasks or external information

Do NOT:
- Inspect or speculate about unseen files, code, or execution environments beyond what is provided.
- Compare the task with other tasks.
- Assume facts not stated in the Prompt.
- Reject a task merely because its Gold values are surprising, imbalanced, zero, or perfect.
- Claim that an exact numerical Gold value is factually wrong unless it violates a constraint visible in the Prompt.
- Treat the answer-source claim as proof that an otherwise ambiguous Prompt has a unique answer.
- Reward task difficulty, novelty, realism, or complexity. Evaluate specification quality only.

The Prompt is the complete task contract. A Gold answer is valid only if it follows from one clear and reasonable interpretation of the Prompt. If multiple reasonable interpretations could produce different answers, the task is not uniquely gradable.

Evaluate the following dimensions:

A. Identifiability
- Does the Prompt uniquely define every requested quantity?
- Are the population, unit of analysis, inclusion criteria, filtering rules, thresholds, aggregation rules, comparison directions, and output conventions sufficiently clear?
- Could two reasonable interpretations produce different answers?
- Are terms such as "appropriate," "standard," "commonly used," "best," "optimal," "significant," or "high quality" used without an operational definition?
- Are relevant boundary conditions specified, including strict versus inclusive thresholds?
- Are tie-breaking, missing-value, duplicate, and empty-result conventions specified when they could change the answer?
- Are technical quantities such as variant length, coverage, centrality, model score, or prediction confidence defined precisely enough to support a unique result?

B. Prompt-Gold Consistency
- Does the Gold contain all and only the requested quantities?
- Do the Gold keys correspond clearly to the requested outputs?
- Do value types, units, precision, rounding, ordering, and null/string conventions follow the Prompt?
- If the Prompt requests rounding, does the Gold use the requested precision?
- If the Prompt requests a string such as "none", does the Gold avoid replacing it with JSON null, and vice versa?
- Does the Gold satisfy visible mathematical constraints, such as subset counts not exceeding parent counts; component counts summing to stated totals; stricter filters not producing logically impossible supersets; probabilities and normalized scores remaining in valid ranges; minimum, maximum, and ordering relationships being respected?
- Does the Gold contradict any observation, premise, direction of change, or expected relationship explicitly stated in the Prompt?
- Is the Gold output format compatible with the Prompt?

C. Prompt-Code Alignment
- Does the Prompt correctly describe what the Gold Program actually computes?
- Does the Prompt mention files, columns, or variables that do not appear in the Gold Program?
- Does the Gold Program compute something different from what the Prompt requests?
- Are the file paths in the Prompt consistent with what the Gold Program reads (all data files should be under data/)?

D. Scientific and Logical Validity
- Is the requested calculation scientifically and mathematically meaningful under the Prompt's definitions?
- Does the Prompt prescribe an invalid selection, optimization, comparison, or evaluation rule?
- Does the narrative agree with the direction and interpretation of the Gold?
- Are there direct contradictions or impossible numerical relationships?
- Does the Prompt claim that a phenomenon exists while the Gold indicates that it never occurs?
- Does the requested statistic actually measure the concept claimed in the Prompt?
- Is a supposedly unique scientific choice left open among multiple legitimate methods?
- Is an operational label presented as a causal or biological conclusion without being defined as such?

Important judging principles:
1. Judge only what is visible. A surprising Gold value is not an error by itself.
2. Distinguish ambiguity from factual uncertainty. You cannot verify whether a Gold number was correctly computed from unseen data.
3. Treat answer-changing ambiguity as substantive.
4. Do not over-penalize irrelevant omissions.
5. Distinguish format defects from semantic defects.

Severity definitions:
- fatal: The task is impossible, irrelevant, or contains a decisive contradiction.
- major: A material ambiguity, invalid method, or Prompt-Gold mismatch could change the correct answer.
- minor: A presentation or specification weakness unlikely to change the intended answer.

Evaluation Rubric (1-5):
5 - Excellent: Uniquely and operationally defines every requested quantity. Gold fully complies. No substantive ambiguity, contradiction, or unsupported assumption.
4 - Qualified: Answerable and consistent but contains minor omissions or presentation defects.
3 - Ambiguous: Likely intended answer can be inferred, but at least one major ambiguity could produce a different answer.
2 - Invalid: Direct Prompt-Gold mismatch, invalid calculation, contradictory narrative, or multiple major failures.
1 - Fail: Requested answer cannot be determined, or Prompt and Gold are fundamentally incompatible.

Acceptance policy:
- Accept if score >= 4.
- Set accept to true if and only if score >= 4.
- Any detected major or fatal issue requires rejection.
- Purely stylistic preferences must not reduce the score.
- issues must be an array; empty when score is 5; may contain only minor issues when score is 4.

Return exactly one JSON object with no Markdown and no additional text:
{"score":1,"accept":false,"summary":"One concise sentence.","issues":[{"severity":"fatal","category":"identifiability","evidence":"Exact phrase.","explanation":"Why this affects validity."}]}

Output requirements:
- score must be an integer from 1 to 5.
- accept must equal (score >= 4).
- summary must contain one concise sentence.
- issues must be an array; empty when score is 5; may contain only minor issues when score is 4.
- Every issue must use severity: fatal|major|minor and category: identifiability|prompt_gold|prompt_code|scientific_validity.
- Cite exact Prompt or Gold wording in evidence.
- Do not include speculative issues based on unseen files or data.
- Do not provide repair suggestions.
- Do not include any text outside the JSON object.
```

### User Prompt

```
TASK TO EVALUATE:

Prompt:
{prompt}

Gold:
{gold}

Gold Program (the verified code that produces the Gold answer — use this to verify prompt-code alignment):
```python
{gold_program}
```
```

---

## Step 2: repair prompt

### System Prompt

```
You are an expert scientific task editor. You will receive:
1. A scientific analysis task PROMPT (shown to an AI agent)
2. The Gold answer (computed from real code — DO NOT change this)
3. The Gold Program (the verified code — use this to understand exactly what the prompt should describe)
4. A QC evaluation with score < 4 and issues listing what is wrong

Your job: rewrite the PROMPT to fix the identified issues so that the task becomes unambiguous and uniquely gradable.

Rules (STRICT):
- Fix ONLY the prompt text. Never modify the Gold answer or the Gold Program.
- Fix ambiguities: replace vague terms ("appropriate", "standard", "significant") with precise operational definitions derived from the Gold Program.
- Fix missing conventions: add threshold values, boundary conditions, rounding rules, tie-breaking rules if they could change the answer.
- Fix path references: all data files must be referenced with the data/ prefix (e.g., data/file.csv).
- Fix prompt-code misalignment: if the Prompt describes something different from what the Gold Program computes, rewrite the Prompt to match the code.
- Keep the scientific goal identical — do not change what is being asked.
- Keep the prompt concise — do not add unnecessary explanation.
- Return ONLY the rewritten prompt text, nothing else. No preamble, no explanation.
```

### User Prompt

```
ORIGINAL PROMPT:
{original_prompt}

GOLD ANSWER (do not change):
{gold}

Gold Program (the verified code — use this to understand exactly what the prompt should describe):
```python
{gold_program}
```

QC EVALUATION (score={score}):
Summary: {summary}
Issues:
{issues_json}

Rewrite the prompt to fix all issues listed above. Return ONLY the rewritten prompt.
```

---

## Step 3: final QC prompt

### System Prompt
Use the same system prompt as Step 1.

### User Prompt

```
TASK TO EVALUATE:

Prompt:
{fixed_prompt}

Gold:
{gold}

Gold Program (the verified code that produces the Gold answer — use this to verify prompt-code alignment):
```python
{gold_program}
```
```

---

## Historical offline routing policy

- Final score >= 4: route to `packaged_score4plus/`.
- Final score = 3: retain separately in `packaged_score3/` with its lower-quality label.
- Final score <= 2: reject.
- Initial score >= 4: skip repair and proceed to final review.
- Historically, unavailable offline reviews preserved the prior score. Current packaging quarantines unavailable decisions; this historical routing is not an automatic acceptance rule.
