# AI Use Declaration
**NLI Classification Group Project**
**University of Edinburgh | MSc Business Analytics | 2025–26**

---

## Declaration

This project made use of AI coding assistants — specifically **Claude (Anthropic)** and **ChatGPT (OpenAI)** — to support the development of code across all pipeline notebooks. This document declares the nature and scope of that use, and clarifies the boundary between AI-assisted code generation and the intellectual contribution of the project team.

All experimental design decisions, prompt engineering choices, result interpretation, and analysis in the report are the original work of the project team. AI tools were used as coding assistants in the same way one might use Stack Overflow, GitHub Copilot, or documentation lookups — to accelerate implementation, not to replace understanding.

---

## 1. What AI Was Used For

### 1.1 Code Generation (All Notebooks)

Each of the ten pipeline notebooks was developed iteratively using Claude and ChatGPT as coding assistants. The process was as follows:

- The team specified the problem, inputs, outputs, and constraints in natural language
- The AI assistant generated a draft implementation
- The team reviewed, tested, debugged, and corrected the output
- The process was repeated ("vibe coding") until the notebook produced correct results verified against manually checked outputs

The AI assistants did **not** design the experiments, choose the models, decide on thresholds, or interpret results. Those decisions were made by the team and then translated into code with AI assistance.

---

## 2. Representative Prompts Used During Development

The following are representative examples of the prompts used to generate each major component. These are reconstructed from the development process to give an accurate picture of the type of instructions given. Some prompts were refined across multiple turns before producing the final implementation.

---

### Notebook 01 — Data Preparation

> *"Write a Python script that loads the MultiNLI matched and mismatched dev sets from JSONL files. It should remove rows where gold_label is '-' (ambiguous), then do stratified sampling to create three non-overlapping sets: 200-sample dev set from matched genres, 800-sample test set from matched genres, and 400-sample test set from mismatched genres. Use random_seed=42 throughout. Also verify there's zero pair-level overlap between all three sets and print the label and genre distributions for each. Save outputs to data/ as CSV files."*

Follow-up prompt used:

> *"The data directory path is hardcoded — replace it with an environment variable MULTINLI_DIR with a sensible fallback default so other team members can run it without editing the file."*

---

### Notebook 02 — Encoder Baselines

> *"Write a PyTorch script that fine-tunes and evaluates four encoder models on NLI: bert-base-uncased, roberta-base, microsoft/deberta-v3-small, and microsoft/deberta-v3-base. Load the 800-sample matched test set and 400-sample mismatched test set. For each model, fine-tune on MultiNLI training data (using HuggingFace datasets), evaluate on both test sets, and save per-row predictions to a CSV with columns for each model's predicted label and softmax confidence score. Use seed=42, eval mode for inference, torch.no_grad(). I need the confidence scores from the softmax output saved per row — these will be used as gating signals in a hybrid system later."*

Follow-up:

> *"Also save per-class precision, recall, and F1 scores for each model and print a summary table at the end."*

---

### Notebook 03 — GPT-4o Prompting

> *"Write a script that calls the OpenAI GPT-4o API (model: gpt-4o) to classify NLI pairs from a 800-row CSV. I need to test 4 prompt strategies: P1 zero-shot (minimal instruction), P2 zero-shot with definitions, P3 few-shot with 3 balanced examples, P4 few-shot CoT with step-by-step reasoning. Use temperature=0.0 and seed=42 for all calls. Parse the output to extract exactly one of: entailment, neutral, contradiction. Track token counts and compute cost per query. Save everything to results/api_results_gpt4o.csv with columns: premise, hypothesis, label_true, predicted_label, prompt, tokens, cost_usd."*

Follow-up:

> *"Add exponential backoff retry logic (3 retries) for rate limit and API errors. Also add a label parser that handles edge cases where the model returns the label mid-sentence or with punctuation around it."*

Follow-up:

> *"Run the same script on the mismatched test set (400 rows) and save to api_results_gpt4o_mm.csv."*

---

### Notebook 04 — Other LLMs (Claude, Llama, GPT-5)

> *"Extend the LLM evaluation to three more models: Claude Sonnet (via Anthropic API), Llama 3.3 70B (via Groq API), and GPT-5 o3-mini (via OpenAI API). For each model, run all four prompt strategies (P1–P4) on the 800-sample matched test set. Use the same label parser, retry logic, and output format as notebook 03. Save separate CSVs for each model: api_results_claude.csv, api_results_llama.csv, api_results_gpt5.csv."*

Follow-up:

> *"Claude P4 is timing out because the CoT traces are very long. Add a timeout of 15 seconds per call and a max_tokens cap of 100 for Claude. Log any failed rows as 'unknown' rather than crashing the whole run."*

Follow-up:

> *"Make sure the main() function calls all three model evaluation functions, not just the first one."*

---

### Notebook 05 — Hybrid Gatekeeper v1 & v2

> *"Write a hybrid gatekeeper system that routes NLI queries through a local encoder first. If the DeBERTa softmax confidence is above a threshold theta, use the encoder prediction (free). If below, escalate to GPT-4o using the P3 few-shot prompt. Test thresholds 0.85, 0.90, 0.95. Run on both matched (800) and mismatched (400) test sets. Load the encoder confidence scores from the saved CSV (encoder_predictions_matched.csv). Track which rows go to encoder vs API, total API cost, and save full per-row results to hybrid_v1_results.csv."*

Follow-up:

> *"Create a v2 variant that uses Claude Sonnet with CoT as the fallback instead of GPT-4o. Parameterise the API caller so both v1 and v2 use the same run_hybrid() function with a different call_api_fn argument."*

---

### Notebook 05c — Hybrid v4 (DeBERTa-large gate)

> *"Create a v4 variant of the hybrid system that uses DeBERTa-v3-large as the gating encoder instead of DeBERTa-v3-base. The hypothesis is that the large model has better confidence calibration even if its aggregate accuracy is the same. Load the large model's confidence scores from the encoder CSV (deberta_v3_large_conf column). Keep GPT-4o P3 as the fallback. Run at thresholds 0.85, 0.90, 0.95 on matched, and θ=0.90 on mismatched. Save to hybrid_v4_results.csv."*

---

### Notebook 05d — Hybrid v5 (Ensemble Gate)

> *"Implement a novel ensemble gating strategy. Instead of using confidence threshold, use disagreement between three DeBERTa models (bert, deberta-small, deberta-base) as the gating signal. If all three models agree on the label (unanimous), accept that label. If any model disagrees, escalate to GPT-4o P4 CoT. The hypothesis is that disagreement identifies genuinely ambiguous samples, not just uncertain ones. Save gate decision (unanimous vs escalated), per-group accuracy, and full results to hybrid_v5_results.csv."*

---

### Notebook 06 — Cost Analysis

> *"Write a script that reads all results CSVs and computes cost per 1,000 queries for every system. For API-based systems, sum the cost_usd column and normalise to per-1000. For encoder-only systems, cost is $0. For hybrid systems, only the API-routed rows have a cost. Output a summary CSV (cost_summary.csv) with columns: system, strategy, cost_per_1k, matched_accuracy. Also compute a cost-efficiency metric (accuracy per dollar)."*

---

### Notebook 07 — Figures

> *"Write a matplotlib/seaborn script that generates 12 publication-quality figures from the results CSVs: (1) accuracy bar chart across all systems, (2) cost-accuracy Pareto frontier scatter, (3) matched vs mismatched comparison bar, (4) per-class F1 grouped bars, (5-8) confusion matrices for DeBERTa, GPT-4o, Claude, Hybrid v2, (9) genre accuracy heatmap, (10) hybrid threshold trade-off dual-axis line chart, (11) ensemble gate breakdown stacked bar, (12) confidence vs ensemble gating comparison. Save all to figures/ at 300 DPI. Use seaborn-v0_8-whitegrid style, colorblind-safe palette."*

Multiple follow-up fix prompts during the audit session (separate from initial generation):

> *"Fig 1 has duplicate Claude bars — there are two bars at 87.4% and two at 88.5%. The problem is the CSV loop creates 'Claude Sonnet P1/P3' keys and the verified_points dict creates 'Claude Sonnet 4.5 P*' keys — both get plotted. Fix it by removing the CSV loop for Claude entirely and relying only on the verified_points dict."*

> *"Fig 10 title says 'Hybrid Architecture Trade-off' but the data comes from hybrid_v2_results.csv. Change the title to 'Hybrid v2 Architecture Trade-off'."*

> *"Fig 3 only shows DeBERTa, GPT-4o, and Hybrid v2 in the matched vs mismatched comparison. Add Hybrid v1 (which has the best mismatched accuracy of 91.3%) and Hybrid v4 (best matched). Label them 'Hybrid v1 (best MM)' and 'Hybrid v4 (best M)'."*

> *"Fig 4 only shows DeBERTa and Hybrid v2 per-class F1 bars. Add Hybrid v4 as a third bar group. Adjust bar width to 0.25 since there are now 3 systems."*

---

### Notebook 08 — Error Analysis

> *"Write a script that analyses classification errors by genre and error type. For each error, record the true label, predicted label, genre, and the specific confusion type (e.g. entailment→neutral). Produce a summary table of error rates by genre, a table of dominant error types, and 4 detailed linguistic case studies — one per major error type — with the actual premise/hypothesis text. Save to results/error_analysis.csv."*

---

### fix_fig2.py — Publication Figure 2

> *"Create a separate publication-quality version of Figure 2 (cost-accuracy frontier) with an inset zoomed panel showing the crowded hybrid + encoder region (costs < $0.1, accuracy > 0.88). Use mpl_toolkits.axes_grid1.inset_locator. Plot all 10 system families with different marker styles. Use log scale for the x-axis. Annotate key systems (Hybrid v4 BEST, Claude P3, GPT-5). The main plot should show the full range including GPT-5 at $14.83. Hardcode the data points from the verified numbers rather than loading from CSV."*

Follow-up fix:

> *"The cost values for some systems are wrong. Fix: Hybrid v1 θ=0.90 should be $0.013 (not $0.011), Claude P2 should be $0.399 (not $0.410), Hybrid v1 θ=0.95 should be $0.023 (not $0.018). These come from cost_summary.csv."*

---

### Report Writing Assistance

> *"Given these results tables [pasted tables], write a Section 5.6 explaining the Hybrid v5 ensemble gating finding. The key insight is that the 100 escalated samples (those where the 3 DeBERTa models disagree) have only 51% accuracy even with GPT-4o — barely above random. This means ensemble disagreement gating identifies annotation ambiguity, not model uncertainty. Contrast this with confidence-threshold gating where GPT-4o scores ~85% on the escalated rows."*

> *"Write Section 9.2 contrasting confidence gating vs ensemble gating. Confidence gating escalates uncertain samples (tractable for LLMs). Ensemble gating escalates samples where models actively conflict (annotation-ambiguous, LLMs cannot help). The practical implication: confidence gating for accuracy, ensemble gating for data quality auditing."*

> *"Write the cost-accuracy Pareto frontier section (§7.2) identifying the three systems that lie on the frontier: DeBERTa-v3-base (free, 90.12%), Hybrid v4 (near-free, 90.62%), Hybrid v1 (best mismatched, 91.3%). Explain why all pure LLM systems fall below the frontier."*

Note: All report sections generated with AI assistance were reviewed, edited, and validated against the CSV results by the team before inclusion. Numbers were not altered from their verified values.

---

## 3. What Was NOT AI-Generated

The following contributions are entirely the team's own work:

- **Experimental design**: The four-tier architecture (encoders → GPT-4o → other LLMs → hybrid gatekeeper) was the team's original design. The hybrid gatekeeper concept, the choice of DeBERTa as the gate, and the ensemble disagreement variant (v5) were the team's ideas.
- **The annotation ceiling finding**: The discovery that GPT-4o scores only 51% on ensemble-disagreement-gated rows — and the interpretation of this as annotation ambiguity rather than model failure — emerged from running the experiments and was not suggested by an AI tool.
- **Prompt design (P1–P5)**: The five NLI prompt strategies were designed by the team through iterative dev-set evaluation. The documented prompts in `PROMPTS.md` are the team's own.
- **Result verification**: All numbers in the report were manually cross-checked against the source CSVs in `results/`. The bug-fix audit session (fixing 5 figure errors) was performed by the team.
- **Interpretation and analysis**: All discussion sections, the error case studies (§8.3), and the business implications (§9.3) reflect the team's own reasoning about the results.

---

## 4. AI Tools Used

| Tool | Provider | Primary Use |
|------|----------|-------------|
| Claude (claude-sonnet-4-5, claude-sonnet-4-6) | Anthropic | Code generation, notebook drafting, figure debugging, report section drafting |
| ChatGPT (GPT-4o) | OpenAI | Code generation, alternative implementations, debugging |

Both tools were accessed via their standard web interfaces (claude.ai and chat.openai.com). No API-based automation was used for code generation — all prompts were entered manually.

---

## 5. Summary Statement

AI coding assistants were used extensively to accelerate code development across all ten pipeline notebooks and the report. The nature of use is best described as **collaborative vibe coding**: the team specified what was needed, the AI generated a working draft, and the team iterated, corrected, and validated the outputs. This is consistent with modern software development practice.

The experimental design, intellectual contributions, novel findings, and result interpretation are the original work of the project team. The use of AI tools did not substitute for understanding — it accelerated execution.

---

*Declared by: Ronit Jitesh, Luis David, Muskan*
*Date: March 2026*
*Module: LLM-based NLP | MSc Business Analytics | University of Edinburgh*
