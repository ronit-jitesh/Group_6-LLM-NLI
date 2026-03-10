# NLI Classification — Prompt Engineering Document

**University of Edinburgh | MSc Business Analytics | LLM-based NLP**  
**Seed: 42 | Temperature: 0.0 | Models: GPT-4o, Claude Sonnet 4.5, GPT-5 (o3-mini)**

---

## Overview

This document specifies all prompt strategies used in the project, their design rationale, and expected behaviour. Five strategies were designed and evaluated (P1–P5), ranging from minimal zero-shot to 32-shot in-context learning.

All prompts enforce single-word output to allow deterministic parsing. Prompts were developed iteratively on the 200-sample dev set before being locked for test evaluation.

---

## Prompt Design Principles

1. **Minimality first**: Start with the simplest possible instruction (P1) and add complexity only if accuracy improves on dev set.
2. **Output constraint**: Every prompt ends with an explicit instruction to respond with exactly one word (`entailment`, `contradiction`, or `neutral`) to prevent verbose outputs that require post-hoc parsing.
3. **Balanced examples**: Few-shot examples (P3, P4, P5) always include exactly one instance of each label to avoid class-frequency bias.
4. **Reasoning before label** (P4 only): CoT prompts instruct the model to reason step-by-step *before* emitting the label, following Wei et al. (2022).

---

## P1 — Zero-Shot Simple

**Strategy**: Minimal instruction. No definitions, no examples. Forces the model to rely entirely on pre-training knowledge of NLI.

**Design rationale**: Establishes the zero-shot baseline. Deliberately avoids priming the model with task-specific vocabulary to measure raw NLI capability.

```
Classify the logical relationship between the premise and hypothesis.
Premise: {premise}
Hypothesis: {hypothesis}
Respond with exactly one word: entailment, contradiction, or neutral.
```

**Expected behaviour**: Fast, low-cost. Strong models (Claude Sonnet, GPT-4o) should handle this well. Weaker models may struggle with the neutral/entailment boundary.

**Dev set accuracy**: GPT-4o 84.0% | Claude 87.4%

---

## P2 — Zero-Shot + Definitions

**Strategy**: Same structure as P1 but adds explicit definitions of all three labels before the classification instruction.

**Design rationale**: Addresses potential ambiguity in the neutral/entailment boundary by making the logical distinction explicit. Hypothesis: definitions will reduce false entailment predictions.

```
Definitions:
- entailment: the hypothesis is necessarily TRUE given the premise
- neutral: the hypothesis may or may not be true given the premise  
- contradiction: the hypothesis is necessarily FALSE given the premise

Classify the logical relationship between the premise and hypothesis.
Premise: {premise}
Hypothesis: {hypothesis}
Respond with exactly one word: entailment, contradiction, or neutral.
```

**Expected behaviour**: Should improve neutral precision by anchoring the model to the "necessarily true/false" distinction.

**Observed result**: *Decreased* accuracy on GPT-4o (84.0% → 82.9%). The "necessarily TRUE" criterion appears to over-constrain GPT-4o, causing it to downgrade legitimate entailments to neutral. Claude showed the opposite pattern (+1.0pp), suggesting instruction-following models benefit more from explicit definitions.

**Dev set accuracy**: GPT-4o 82.9% | Claude 88.4%

---

## P3 — Few-Shot (3 Examples)

**Strategy**: Three balanced in-context examples (one per class) before the target pair. Examples drawn from the 200-sample dev set, selected to cover different genre styles (formal, conversational, factual).

**Design rationale**: Few-shot examples provide a schema for the output format and implicitly demonstrate the neutral/entailment distinction without constraining the reasoning path.

```
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
```

**Expected behaviour**: Improved accuracy over P1/P2, especially on Contradiction (examples demonstrate what "necessarily false" looks like in practice).

**Dev set accuracy**: GPT-4o 84.8% | Claude 88.5%

---

## P4 — Few-Shot + Chain-of-Thought (CoT)

**Strategy**: Same 3 examples as P3, but each example includes a brief step-by-step reasoning trace before the label. Target query also prompts for step-by-step reasoning.

**Design rationale**: Wei et al. (2022) demonstrated CoT improves multi-step reasoning tasks. For NLI, the hypothesis is that explicit reasoning about semantic entailment reduces errors on complex cases involving negation, attribution, and pragmatic inference.

```
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
```

**Expected behaviour**: Best matched accuracy due to explicit reasoning. *However*, the rigid step-by-step template introduces format bias on out-of-distribution genres.

**Observed result**: Best matched accuracy for GPT-4o (85.5%) but *worse* mismatched performance than P1 (90.0% vs 90.5%). CoT templates calibrated on matched genres degrade on Verbatim and face-to-face transcripts — consistent with Ye & Durrett (2022).

**Note on Claude P4**: Encountered `max_tokens` saturation at 1000 tokens (CoT traces are verbose). All 800 rows returned errors. Fix applied (`max_tokens=100`, `timeout=30`) but P4 was excluded from final evaluation given the cost/accuracy trade-off observed in GPT-4o P4 vs P3 (+0.7pp at +10% cost).

**Dev set accuracy**: GPT-4o 85.5% | Claude — (failed)

---

## P5 — 32-Shot

**Strategy**: 32 carefully curated examples from the dev set (covering all 5 genres, balanced across labels) provided as in-context demonstrations. No CoT reasoning traces.

**Design rationale**: Tests whether a larger example bank improves accuracy beyond the 3-example sweet spot identified in P3. Curated to represent genre diversity.

**Expected behaviour**: Marginal accuracy gain over P3 at significantly higher token cost.

**Observed result**: No accuracy gain over P1 on matched (84.0% = 84.0%). Confirms diminishing returns beyond ~3 examples for GPT-4o on NLI. Cost: $3.52/1k vs $0.20/1k for P1 — a 17× cost premium for identical accuracy.

**Dev set accuracy**: GPT-4o 84.0%

---

## Prompt Comparison Summary

| Prompt | Avg Tokens | GPT-4o Matched | Claude Matched | Cost/1k (GPT-4o) |
|--------|-----------|----------------|----------------|-----------------|
| P1 Zero-shot | 75 | 84.0% | 87.4% | $0.20 |
| P2 Zero-shot + Defs | 102 | 82.9% | 88.4% | $0.27 |
| P3 Few-shot (3) | 142 | 84.8% | **88.5%** | $0.37 |
| P4 CoT | 156 | **85.5%** | — | $0.41 |
| P5 32-shot | ~1,401 | 84.0% | — | $3.52 |

---

## Zero-Shot vs Few-Shot: Written Comparison

### What the results show

P1 (zero-shot) achieves 84.0% on GPT-4o and 87.4% on Claude Sonnet — strong results for a model receiving no task-specific examples. Moving to P3 (3 examples) adds +0.8pp for GPT-4o and +1.1pp for Claude. This small gain reflects a key property of large frontier models: they already have a strong prior for NLI from pre-training, and a handful of examples adds schema reinforcement rather than new capability.

The more important zero-shot vs few-shot finding is on **mismatched genres**. GPT-4o P1 (zero-shot) achieves **90.5% mismatched** — *higher* than P4 CoT at 90.0% and higher than P3 at 89.0%. This reversal occurs because few-shot and CoT examples are implicitly drawn from matched-genre styles (formal, written text). When the model encounters mismatched genres (conversational transcripts, Verbatim text), the rigid template of its in-context examples becomes a liability rather than an asset. Zero-shot allows the model to adapt its reasoning style to each genre freely.

### Why CoT is weaker than RoBERTa on matched

RoBERTa-base achieves 88.6% matched — higher than GPT-4o P4 CoT at 85.5%. This seems counter-intuitive given GPT-4o's far larger parameter count and broader pre-training, but has a clear explanation:

1. **Task specialisation**: RoBERTa is fine-tuned *directly on MultiNLI* — it has seen hundreds of thousands of premise-hypothesis pairs from the same distribution. GPT-4o is a general-purpose model inferring NLI from its pre-training, with no direct exposure to MultiNLI's label conventions.
2. **CoT overhead**: The step-by-step reasoning template in P4 introduces intermediate inference steps that can compound errors. If step 1 of the reasoning is slightly wrong, the label derived from it inherits the error. A fine-tuned encoder makes a direct mapping with no intermediate steps to go wrong.
3. **Annotation convention alignment**: MultiNLI has specific conventions about what counts as "neutral" vs "entailment" — especially around attribution ("he argued that X" vs "X is true"). Fine-tuned encoders learn these conventions implicitly; CoT prompts may apply a more general logical standard that diverges from the annotation guidelines.

This finding supports the conclusion that **fine-tuned encoders and LLMs solve fundamentally different sub-problems**: encoders are better at distribution-specific pattern matching; LLMs are better at out-of-distribution generalisation. The hybrid gatekeeper exploits exactly this complementarity.

---

## Why All Models Were Run

Running all models (BERT, RoBERTa, DeBERTa ×3, GPT-4o, Claude, GPT-5, Llama) rather than taking the single best from literature serves three purposes:

1. **Empirical validation over literature trust**: Literature results are typically reported on different dataset splits, different fine-tuning regimes, and often cherry-picked configurations. Running all models on the *same* test set with the *same* evaluation protocol produces directly comparable results.

2. **The counter-intuitive findings only emerge from comparison**: The finding that GPT-5 (o3-mini) performs *worse* than BERT-base ($14.83/1k vs $0/1k, 75.9% vs 83.6%) cannot be known in advance from literature — reasoning-optimised models are often assumed to be universally better. Similarly, CoT underperforming on mismatched is a finding that requires running CoT to discover.

3. **Rubric requirement**: The assignment explicitly requires comparison across "LLM families" and "prompt strategies." Running a single model and citing literature would not satisfy this criterion — it would be a literature review, not an empirical evaluation.

---

*References: Wei et al. (2022) — Chain-of-Thought Prompting. Ye & Durrett (2022) — Unreliability of Explanations in Few-Shot Prompting.*
