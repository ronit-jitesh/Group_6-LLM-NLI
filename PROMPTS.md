# NLI Classification -- Prompt Engineering Document

University of Edinburgh | MSc Business Analytics | LLM-based NLP
Seed: 42 | Temperature: 0.0 | Models: GPT-4o, Claude Sonnet 4.5, GPT-5 (o3-mini)

---

## Overview

This document specifies all prompt strategies used in the project, their design rationale, and
expected behaviour. Five strategies were designed and evaluated (P1-P5), ranging from minimal
zero-shot to 32-shot in-context learning.

All prompts enforce single-word output to allow deterministic parsing. Prompts were developed
iteratively on the 200-sample dev set before being locked for test evaluation.

---

## Prompt Design Principles

1. Minimality first: Start with the simplest possible instruction (P1) and add complexity only
   if accuracy improves on dev set.
2. Output constraint: Every prompt ends with an explicit instruction to respond with exactly one
   word (entailment, contradiction, or neutral) to prevent verbose outputs.
3. Balanced examples: Few-shot examples (P3, P4, P5) always include exactly one instance of
   each label to avoid class-frequency bias.
4. Reasoning before label (P4 only): CoT prompts instruct the model to reason step-by-step
   before emitting the label, following Wei et al. (2022).

---

## P1 -- Zero-Shot Simple

Strategy: Minimal instruction. No definitions, no examples. Forces the model to rely entirely
on pre-training knowledge of NLI.

Design rationale: Establishes the zero-shot baseline. Deliberately avoids priming the model
with task-specific vocabulary to measure raw NLI capability.

Prompt template:

    Classify the logical relationship between the premise and hypothesis.
    Premise: {premise}
    Hypothesis: {hypothesis}
    Respond with exactly one word: entailment, contradiction, or neutral.

Results: GPT-4o 84.0% | Claude 87.4%

---

## P2 -- Zero-Shot + Definitions

Strategy: Same structure as P1 but adds explicit definitions of all three labels before the
classification instruction.

Design rationale: Addresses potential ambiguity in the neutral/entailment boundary by making
the logical distinction explicit.

Prompt template:

    Definitions:
    - entailment: the hypothesis is necessarily TRUE given the premise
    - neutral: the hypothesis may or may not be true given the premise
    - contradiction: the hypothesis is necessarily FALSE given the premise

    Classify the logical relationship between the premise and hypothesis.
    Premise: {premise}
    Hypothesis: {hypothesis}
    Respond with exactly one word: entailment, contradiction, or neutral.

Observed result: Decreased accuracy on GPT-4o (84.0% -> 82.9%). The "necessarily TRUE"
criterion over-constrains GPT-4o, causing it to downgrade legitimate entailments to neutral.
Claude showed the opposite pattern (+1.0pp).

Results: GPT-4o 82.9% | Claude 88.4%

---

## P3 -- Few-Shot (3 Examples)

Strategy: Three balanced in-context examples (one per class) before the target pair. Examples
drawn from the 200-sample dev set, selected to cover different genre styles.

Design rationale: Few-shot examples provide a schema for the output format and implicitly
demonstrate the neutral/entailment distinction.

Prompt template:

    Examples:

    Premise: The actor was born in New York City.
    Hypothesis: The actor is American.
    Label: neutral

    Premise: The cat is sleeping on the mat.
    Hypothesis: The cat is resting.
    Label: entailment

    Premise: She said she would never eat meat again.
    Hypothesis: She eats steak every week.
    Label: contradiction

    Now classify:
    Premise: {premise}
    Hypothesis: {hypothesis}
    Respond with exactly one word: entailment, contradiction, or neutral.

Results: GPT-4o 84.8% | Claude 88.5%

---

## P4 -- Few-Shot + Chain-of-Thought (CoT)

Strategy: Same 3 examples as P3, but each example includes a brief step-by-step reasoning
trace before the label.

Design rationale: Wei et al. (2022) demonstrated CoT improves multi-step reasoning tasks.
For NLI, explicit reasoning about semantic entailment should reduce errors on complex cases.

Prompt template:

    Examples:

    Premise: The actor was born in New York City.
    Hypothesis: The actor is American.
    Step-by-step: Being born in NYC means the actor was born in the US. However,
    the hypothesis claims nationality (American), which requires citizenship,
    not just birthplace. This cannot be confirmed from the premise alone.
    Label: neutral

    Premise: The cat is sleeping on the mat.
    Hypothesis: The cat is resting.
    Step-by-step: Sleeping is a form of resting. If the cat is sleeping,
    it must be resting. The hypothesis necessarily follows.
    Label: entailment

    Premise: She said she would never eat meat again.
    Hypothesis: She eats steak every week.
    Step-by-step: She declared she would never eat meat. Steak is meat.
    These two claims are mutually exclusive.
    Label: contradiction

    Now classify step-by-step:
    Premise: {premise}
    Hypothesis: {hypothesis}
    Step-by-step:

Observed result: Best matched accuracy for GPT-4o (85.5%) but worse mismatched performance
than P1 (90.0% vs 90.5%). CoT templates calibrated on matched genres degrade on Verbatim
and face-to-face transcripts.

Results: GPT-4o 85.5% | Claude 80.0% (CoT introduced deliberation noise)

---

## P5 -- 32-Shot (Exploratory)

Strategy: 32 carefully curated examples from the dev set covering all 5 genres, balanced
across labels. No CoT reasoning traces.

Design rationale: Tests whether a larger example bank improves accuracy beyond the 3-example
sweet spot identified in P3.

Observed result: No accuracy gain over P1 on matched (84.0% = 84.0%). Confirms diminishing
returns beyond ~3 examples for GPT-4o on NLI. Cost: $3.52/1k vs $0.20/1k for P1 -- a 17x
cost premium for identical accuracy.

Results: GPT-4o 84.0% (exploratory only, run on dev set)

---

## Prompt Comparison Summary

| Prompt | Avg Tokens | GPT-4o Matched | Claude Matched | Cost/1k (GPT-4o) |
|--------|-----------|----------------|----------------|-----------------|
| P1 Zero-shot | 75 | 84.0% | 87.4% | $0.20 |
| P2 Zero-shot + Defs | 102 | 82.9% | 88.4% | $0.27 |
| P3 Few-shot (3) | 142 | 84.8% | 88.5% | $0.37 |
| P4 CoT | 156 | 85.5% | 80.0% | $0.41 |
| P5 32-shot | ~1401 | 84.0% | -- | $3.52 |

---

## Key Findings

Zero-shot vs few-shot: P1 (zero-shot) achieves 84.0% on GPT-4o and 87.4% on Claude -- strong
results for no task-specific examples. Moving to P3 adds +0.8pp for GPT-4o and +1.1pp for
Claude. The more important finding is on mismatched genres: GPT-4o P1 achieves 90.5%
mismatched -- higher than P4 CoT at 90.0%. Zero-shot allows the model to adapt its reasoning
style to each genre freely; CoT templates become a liability on out-of-distribution text.

Why CoT is weaker than fine-tuned encoders on matched: RoBERTa-base achieves 88.6% matched
vs GPT-4o P4 CoT at 85.5%. RoBERTa is fine-tuned directly on MultiNLI -- it has seen hundreds
of thousands of premise-hypothesis pairs from the same distribution. GPT-4o infers NLI from
general pre-training with no direct exposure to MultiNLI's label conventions. CoT also
introduces intermediate inference steps that can compound errors.

This supports the conclusion that fine-tuned encoders and LLMs solve different sub-problems:
encoders are better at distribution-specific pattern matching; LLMs are better at
out-of-distribution generalisation. The hybrid gatekeeper exploits this complementarity.

---

References: Wei et al. (2022) Chain-of-Thought Prompting. Ye and Durrett (2022) Unreliability
of Explanations in Few-Shot Prompting for Textual Reasoning.
