# NLI Classification - LLM-Based NLP
University of Edinburgh | MSc Business Analytics | 2025-26

---

## Project Overview

Comprehensive evaluation of Natural Language Inference (NLI) systems on MultiNLI, comparing five
encoder architectures, four LLM families across four prompt strategies, and five hybrid gatekeeper
architectures.

Best overall: Hybrid v5 (3-DeBERTa Ensemble Gate + GPT-4o) - 91.0% matched, 92.5% mismatched

Best cost-efficiency: Hybrid v4 (DeBERTa-v3-large + GPT-4o) - 90.62% matched, $0.007/1k queries

Key finding: Ensemble disagreement gating reveals that 87.5% of MultiNLI samples are unanimously
solvable at 95.0% accuracy, while the remaining 12.5% represent genuinely label-ambiguous cases
that even GPT-4o resolves at only 63% - pointing to an annotation ceiling in MultiNLI rather than
a model problem.

---

## Directory Structure

```
LLM Final/
|
|-- data/
|   |-- nli_dev_200.csv              Prompt tuning set (200 samples, matched)
|   |-- nli_test_800.csv             Primary test set (800 samples, matched)
|   |-- nli_test_mm_400.csv          Generalisation test set (400 samples, mismatched)
|
|-- src/                             Main pipeline (run in numbered order)
|   |-- 01_data_preparation.py       Stratified sampling from MultiNLI JSONL
|   |-- 02_encoder_baselines.py      5 encoder models: BERT, RoBERTa, DeBERTa x3
|   |-- 03_gpt4o_prompting.py        GPT-4o P1-P5 on matched + mismatched
|   |-- 04_other_llms.py             Claude Sonnet 4.5, GPT-5 (o3-mini), Llama 3.3
|   |-- 05a_hybrid_v1_v2_gatekeeper.py    Hybrid v1 (GPT-4o) + v2 (Claude), theta in {0.85,0.90,0.95}
|   |-- 05b_hybrid_v3_deberta_gpt4o_32shot.py  Hybrid v3: 32-shot fallback
|   |-- 05c_hybrid_v4_deberta_large_gpt4o.py   Hybrid v4: DeBERTa-large gate [BEST COST]
|   |-- 05d_hybrid_v5_ensemble_gate.py          Hybrid v5: 3-DeBERTa ensemble gate [BEST OVERALL]
|   |-- 05e_hybrid_v5b_tiered.py               Hybrid v5b: tiered fallback (no API calls)
|   |-- 05f_hybrid_v5c_ensemble_claude.py      Hybrid v5c: ensemble gate + Claude Sonnet
|   |-- 06_cost_analysis.py          Token usage and cost aggregation
|   |-- 07a_figures_main.py          Figures 1-12 (publication quality)
|   |-- 07b_figure2_pareto.py        Figure 2: Cost-accuracy Pareto frontier
|   |-- 08_error_analysis.py         Error type distribution + linguistic case studies
|   |-- 09_genre_label_analysis.py   Per-class P/R/F1 and genre breakdown (Figs 13-15)
|   |-- 10_significance_tests.py     McNemar's test for all key pairwise comparisons
|
|-- utils/
|   |-- evaluate.py                  Typed evaluation utilities: metrics, McNemar, ECE, cost
|   |-- generate_tables.py           Typed table generators: encoders, prompts, hybrids, Pareto
|
|-- results/                         Auto-generated CSVs (produced by src/ scripts)
|-- figures/                         Auto-generated PNGs (produced by src/07*.py)
|
|-- README.md                        This file
|-- AI_USE_DECLARATION.md            AI use declaration (required for submission)
|-- PROMPTS.md                       All prompt templates used in experiments
|-- TRACEABILITY.md                  Number-to-source traceability map
|-- requirements.txt                 Python dependencies
|-- .env.example                     API key template (copy to .env and fill in)
```

---

## Execution Order

```bash
# Environment setup
pip install -r requirements.txt
cp .env.example .env      # fill in your API keys

# Step 1: Data (one-time -- requires MultiNLI JSONL files)
python src/01_data_preparation.py

# Step 2: Encoder baselines (~30 min on MPS/GPU)
python src/02_encoder_baselines.py

# Step 3: GPT-4o prompting (~$3 total)
python src/03_gpt4o_prompting.py

# Step 4: Other LLMs
python src/04_other_llms.py

# Step 5: Hybrid systems (run in order)
python src/05a_hybrid_v1_v2_gatekeeper.py
python src/05b_hybrid_v3_deberta_gpt4o_32shot.py
python src/05c_hybrid_v4_deberta_large_gpt4o.py
python src/05d_hybrid_v5_ensemble_gate.py
python src/05e_hybrid_v5b_tiered.py

# Step 6: Analysis and figures
python src/06_cost_analysis.py
python src/07a_figures_main.py
python src/07b_figure2_pareto.py
python src/08_error_analysis.py
python src/09_genre_label_analysis.py

# Step 7: Statistical significance tests
python src/10_significance_tests.py
```

---

## Key Results

| System | Matched F1 | Mismatched F1 | API % | Cost/1k |
|--------|------------|---------------|-------|---------|
| DeBERTa-v3-base | 0.901 | 0.908 | 0% | $0.000 |
| DeBERTa-v3-large | 0.901 | 0.895 | 0% | $0.000 |
| GPT-4o P4 (pure) | 0.857 | 0.901 | 100% | $0.407 |
| Claude Sonnet P2 (pure) | 0.883 | -- | 100% | $0.399 |
| Hybrid v1 theta=0.90 | 0.901 | 0.913 | 3.8% | $0.013 |
| Hybrid v4 theta=0.90 [BEST COST] | 0.907 | 0.905 | 2.0% | $0.007 |
| Hybrid v5 Ensemble [BEST OVERALL] | 0.910 | 0.925 | 12.0% | $0.258 |

---

## Statistical Significance (McNemar's test, N=800)

| Comparison | p-value | Result |
|------------|---------|--------|
| DeBERTa-base vs Hybrid v1 | 1.0000 | ns |
| DeBERTa-large vs Hybrid v4 | 0.2266 | ns |
| Hybrid v4 vs Hybrid v5 | 0.8774 | ns |
| GPT-4o P1 vs P4 | 0.1624 | ns |
| GPT-4o P4 vs Hybrid v4 | 0.0002 | *** p<0.01 |
| DeBERTa-base vs Hybrid v5 | 0.3239 | ns |

ns = not significant at p<0.05. The GPT-4o vs Hybrid v4 comparison is strongly significant (p<0.01),
confirming the 5.25pp hybrid advantage is not attributable to sampling variation.

---

## Important Notes

Encoder pre-training: The DeBERTa and encoder baselines are fine-tuned on MultiNLI training data.
Their accuracy reflects in-distribution performance, not zero-shot generalisation. GPT-4o and other
LLMs are genuinely zero-shot. This asymmetry is intentional.

Hybrid v5 patch: 30 GPT-4o output labels in hybrid_v5_results.csv were resolved post-hoc
(19 via API retry, 10 via DeBERTa majority vote, 1 via fallback). All reported metrics reflect the
fully resolved 800/400 dataset.

Seed: 42 throughout. All experiments reproducible with provided scripts.

---

## References

Full methodology, results, and analysis: see the submitted report (LLM_Proyect_16_03.docx).
Code-to-result mapping: TRACEABILITY.md
Prompt templates: PROMPTS.md
