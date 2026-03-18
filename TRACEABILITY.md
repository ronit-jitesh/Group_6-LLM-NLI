# Code to Result Traceability Map

Every figure, table, and number in the submitted report (LLM_Proyect_16_03.docx) traces back
to a specific script and output file. This document is for markers to verify that all results
are reproducible from the code.

---

## How to Run Everything (in order)

```bash
cd "LLM Final"

# Data preparation
python src/01_data_preparation.py

# Encoder baselines
python src/02_encoder_baselines.py

# LLM API evaluations
python src/03_gpt4o_prompting.py
python src/04_other_llms.py

# Hybrid gatekeeper systems
python src/05a_hybrid_v1_v2_gatekeeper.py
python src/05b_hybrid_v3_deberta_gpt4o_32shot.py
python src/05c_hybrid_v4_deberta_large_gpt4o.py
python src/05d_hybrid_v5_ensemble_gate.py
python src/05e_hybrid_v5b_tiered.py
python src/05f_hybrid_v5c_ensemble_claude.py

# Analysis and figures
python src/06_cost_analysis.py
python src/07a_figures_main.py
python src/07b_figure2_pareto.py
python src/08_error_analysis.py
python src/09_genre_label_analysis.py

# Statistical significance tests
python src/10_significance_tests.py
```

---

## Script to Output File Map

| Script | Produces | Used in Report |
|--------|----------|----------------|
| 01_data_preparation.py | data/nli_dev_200.csv, data/nli_test_800.csv, data/nli_test_mm_400.csv | Section 2: dataset construction and label distributions |
| 02_encoder_baselines.py | results/encoder_predictions_matched.csv, results/encoder_predictions_mm.csv | Section 6.2: encoder baseline results, Tables 12 and 13 |
| 03_gpt4o_prompting.py | results/api_results_gpt4o.csv, results/api_results_gpt4o_mm.csv | Section 6.3: GPT-4o prompt strategy results, Table 14 |
| 04_other_llms.py | results/api_results_claude.csv, results/api_results_gpt5.csv, results/api_results_llama.csv | Section 6.3: Claude, GPT-5, Llama results, Tables 14 and 15 |
| 05a_hybrid_v1_v2_gatekeeper.py | results/hybrid_v1_results.csv, results/hybrid_v2_results.csv | Section 6.5: Hybrid v1 and v2 results, Table 18 |
| 05b_hybrid_v3_deberta_gpt4o_32shot.py | results/hybrid_v3_results.csv | Section 6.5: Hybrid v3 results, Table 18 and Appendix 1 |
| 05c_hybrid_v4_deberta_large_gpt4o.py | results/hybrid_v4_results.csv | Section 6.5: Hybrid v4 results, Tables 18 and 19 |
| 05d_hybrid_v5_ensemble_gate.py | results/hybrid_v5_results.csv | Section 6.5: Hybrid v5 results, Tables 18 and 19 |
| 05e_hybrid_v5b_tiered.py | results/hybrid_v5b_results.csv (internal comparison only) | Not in primary results; used in development |
| 05f_hybrid_v5c_ensemble_claude.py | results/hybrid_v5c_results.csv (if run) | Not in primary results; Claude fallback variant |
| 06_cost_analysis.py | results/cost_summary.csv | Section 5.3: cost per 1,000 queries, Table 8, Figure 11 |
| 07a_figures_main.py | figures/fig1 through figures/fig12 | All figures in Sections 6 and 7 |
| 07b_figure2_pareto.py | figures/fig2_cost_accuracy_frontier.png | Figure 11: cost-accuracy Pareto frontier |
| 08_error_analysis.py | results/error_analysis.csv | Section 7: error taxonomy, Table 23 |
| 09_genre_label_analysis.py | results/classification_reports.csv, results/genre_label_breakdown.csv, figures/fig13-15 | Sections 6.7 and 6.8: per-class P/R/F1 and genre analysis |
| 10_significance_tests.py | results/significance_tests.csv | Section 6.9: McNemar's test results |

---

## Key Claim to Source Mapping

| Report Claim | Value | Source Script | Source CSV |
|-------------|-------|---------------|------------|
| DeBERTa-v3-base Macro-F1 matched | 0.9010 | 02_encoder_baselines.py | encoder_predictions_matched.csv |
| DeBERTa-v3-base Macro-F1 mismatched | 0.9078 | 02_encoder_baselines.py | encoder_predictions_mm.csv |
| GPT-4o P4 Macro-F1 matched | 0.8568 | 03_gpt4o_prompting.py | api_results_gpt4o.csv |
| GPT-4o P1 Macro-F1 mismatched | 0.9058 | 03_gpt4o_prompting.py | api_results_gpt4o_mm.csv |
| Claude Sonnet P2 Macro-F1 | 0.8834 | 04_other_llms.py | api_results_claude.csv |
| Hybrid v1 theta=0.90 matched F1 | 0.9007 | 05a_hybrid_v1_v2_gatekeeper.py | hybrid_v1_results.csv |
| Hybrid v1 theta=0.90 mismatched F1 | 0.9126 | 05a_hybrid_v1_v2_gatekeeper.py | hybrid_v1_results.csv |
| Hybrid v4 theta=0.90 matched F1 | 0.9068 | 05c_hybrid_v4_deberta_large_gpt4o.py | hybrid_v4_results.csv |
| Hybrid v4 cost per 1k | $0.007 | 06_cost_analysis.py | cost_summary.csv |
| Hybrid v5 matched F1 | 0.9101 | 05d_hybrid_v5_ensemble_gate.py | hybrid_v5_results.csv |
| Hybrid v5 mismatched F1 | 0.9253 | 05d_hybrid_v5_ensemble_gate.py | hybrid_v5_results.csv |
| Hybrid v5 unanimous gate split (87.5%) | 700/800 rows unanimous | 05d_hybrid_v5_ensemble_gate.py | hybrid_v5_results.csv |
| Error counts (DeBERTa 79, GPT-4o 116) | See Table 23 | 08_error_analysis.py | error_analysis.csv |
| Entailment to Neutral errors (DeBERTa) | 30 | 08_error_analysis.py | error_analysis.csv |
| GPT-4o P4 vs Hybrid v4 p-value | 0.0002 | 10_significance_tests.py | significance_tests.csv |
| n=1500 validation F1 delta (max 0.013) | See Table 10 | 02_encoder_baselines.py | encoder_predictions_matched.csv |

---

## Figure to Script Map

| Figure in Report | File | Generated by |
|-----------------|------|--------------|
| Figure 2 (dashboard) | Embedded dashboard image | 02_encoder_baselines.py |
| Figure 3 (genre difficulty radar) | fig9_genre_heatmap.png | 07a_figures_main.py |
| Figure 11 (cost per 1k) | fig2_cost_accuracy_frontier.png | 07b_figure2_pareto.py |
| Figure 12 (n=800 vs n=1500 validation) | Separate validation run | 02_encoder_baselines.py |
| Figure 13 (system ranking bar) | fig1_strategy_accuracy_bar.png | 07a_figures_main.py |
| Figure 14 (encoder per-class F1) | fig13_classification_report_heatmap.png | 09_genre_label_analysis.py |
| Figure 15 (prompt strategy progression) | fig3_matched_vs_mismatched.png | 07a_figures_main.py |
| Figure 16 (matched vs mismatched) | fig3_matched_vs_mismatched.png | 07a_figures_main.py |
| Figure 17 (per-class F1 heatmap) | fig4_per_class_f1.png | 07a_figures_main.py |
| Figure 18 (genre heatmap) | fig9_genre_heatmap.png | 07a_figures_main.py |

---

## Results Files Present in repo

| File | Produced by | Status |
|------|-------------|--------|
| encoder_predictions_matched.csv | 02_encoder_baselines.py | Present |
| encoder_predictions_mm.csv | 02_encoder_baselines.py | Present |
| api_results_gpt4o.csv | 03_gpt4o_prompting.py | Present |
| api_results_gpt4o_mm.csv | 03_gpt4o_prompting.py | Present |
| api_results_claude.csv | 04_other_llms.py | Present |
| api_results_gpt5.csv | 04_other_llms.py | Present |
| api_results_llama.csv | 04_other_llms.py | Present |
| hybrid_v1_results.csv | 05a_hybrid_v1_v2_gatekeeper.py | Present |
| hybrid_v2_results.csv | 05a_hybrid_v1_v2_gatekeeper.py | Present |
| hybrid_v3_results.csv | 05b_hybrid_v3_deberta_gpt4o_32shot.py | Present |
| hybrid_v4_results.csv | 05c_hybrid_v4_deberta_large_gpt4o.py | Present |
| hybrid_v5_results.csv | 05d_hybrid_v5_ensemble_gate.py | Present |
| cost_summary.csv | 06_cost_analysis.py | Present |
| error_analysis.csv | 08_error_analysis.py | Present |
| classification_reports.csv | 09_genre_label_analysis.py | Present |
| genre_label_breakdown.csv | 09_genre_label_analysis.py | Present |
| significance_tests.csv | 10_significance_tests.py | Present |

---

## Notes

All CSV files in results/ are the raw prediction outputs. Every number in the report can be
verified by loading these files and running sklearn.metrics.f1_score or accuracy_score.

The data/ CSVs are the test sets. Seed 42 is used throughout all scripts and API calls.

PROMPTS.md contains the exact text of all five prompt strategies (P1-P5) used in
03_gpt4o_prompting.py and 04_other_llms.py.

API keys are required in .env to re-run scripts 03-05f. See .env.example for the template.
Estimated re-run cost: approximately $8-10 total.

Hybrid v5b_results.csv is not present in results/ as 05e produces it only when the tiered
variant is run post-hoc. The primary results use hybrid_v5_results.csv from 05d.
