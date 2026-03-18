# AI Use Declaration
NLI Classification Group Project
University of Edinburgh | MSc Business Analytics | 2025-26

---

## Declaration

This project made use of AI coding assistants specifically Claude (Anthropic) and ChatGPT (OpenAI), to support the development of code across all pipeline scripts. This document declares the nature and scope of that use, and clarifies the boundary between AI-assisted code generation and the intellectual contribution of the project team.

All experimental design decisions, prompt engineering choices, result interpretation, and analysis in the report are the original work of the project team. AI tools were used as coding assistants in the same way one might use Stack Overflow, GitHub Copilot, or documentation lookups to accelerate implementation, not to replace understanding.

---

## 1. What AI Was Used For

### 1.1 Code Generation

Each of the pipeline scripts (01 through 10) was developed iteratively using Claude and ChatGPT as coding assistants. The process was as follows:

- The team specified the problem, inputs, outputs, and constraints in natural language
- The AI assistant generated a draft implementation
- The team reviewed, tested, debugged, and corrected the output
- The process was repeated until the script produced correct results verified against manually checked outputs

The AI assistants did NOT design the experiments, choose the models, decide on thresholds, or interpret results. Those decisions were made by the team and then translated into code with AI assistance.

### 1.2 Report Drafting Assistance

AI tools were used to draft initial versions of selected report sections. All AI-generated draft text was reviewed, edited, and validated against the CSV results by the team before inclusion. Numbers were not altered from their verified values. The experimental findings, interpretation, and conclusions are the team's own work.

---

## 2. Representative Prompts Used During Development

The following are representative examples of the prompts used to generate each major pipeline component. These are reconstructed from the development process. Some prompts were refined across multiple turns before producing the final implementation.

### Script 01 - Data Preparation

"Write a Python script that loads the MultiNLI matched and mismatched dev sets from JSONL files. It should remove rows where gold_label is '-' (ambiguous), then do stratified sampling to create three non-overlapping sets: 200-sample dev set from matched genres, 800-sample test set from matched genres, and 400-sample test set from mismatched genres. Use random_seed=42 throughout. Also verify there is zero pair-level overlap between all three sets and print the label and genre distributions for each. Save outputs to data/ as CSV files."

Follow-up: "The data directory path is hardcoded replace it with an environment variable MULTINLI_DIR with a sensible fallback default so other team members can run it without editing the file."

### Script 02 - Encoder Baselines

"Write a PyTorch script that evaluates five encoder models on NLI in inference-only mode: bert-base-uncased, roberta-base, deberta-v3-small, deberta-v3-base, deberta-v3-large. Load the 800-sample matched test set and 400-sample mismatched test set. For each model, evaluate on both test sets and save per-row predictions to a CSV with columns for each model's predicted label and softmax confidence score. Use seed=42, eval mode for inference, torch.no_grad(). I need the confidence scores from the softmax output saved per row -- these will be used as gating signals in a hybrid system later."

### Script 03 - GPT-4o Prompting

"Write a script that calls the OpenAI GPT-4o API to classify NLI pairs from an 800-row CSV. I need to test 4 prompt strategies: P1 zero-shot, P2 zero-shot with definitions, P3 few-shot with 3 balanced examples, P4 few-shot CoT with step-by-step reasoning. Use temperature=0.0 and seed=42. Parse the output to extract exactly one of: entailment, neutral, contradiction. Track token counts and compute cost per query."

Follow-up: "Add exponential backoff retry logic (3 retries) for rate limit and API errors. Also add a label parser that handles edge cases where the model returns the label mid-sentence or with punctuation around it."

### Script 05a - Hybrid Gatekeeper v1 and v2

"Write a hybrid gatekeeper system that routes NLI queries through a local encoder first. If the DeBERTa softmax confidence is above a threshold theta, use the encoder prediction (free). If below, escalate to GPT-4o using the P3 few-shot prompt. Test thresholds 0.85, 0.90, 0.95. Run on both matched (800) and mismatched (400) test sets."

### Script 05d - Hybrid v5 (Ensemble Gate)

"Implement a novel ensemble gating strategy. Instead of using confidence threshold, use disagreement between three DeBERTa models as the gating signal. If all three models agree on the label (unanimous), accept that label. If any model disagrees, escalate to GPT-4o P4 CoT. Save gate decision, per-group accuracy, and full results to hybrid_v5_results.csv."

### Script 07 - Figures

"Write a matplotlib/seaborn script that generates 12 publication-quality figures from the results CSVs: accuracy bar chart, cost-accuracy Pareto frontier, matched vs mismatched comparison, per-class F1 grouped bars, confusion matrices, genre heatmap, hybrid threshold trade-off chart, ensemble gate breakdown, gating strategy comparison. Save all to figures/ at 300 DPI."

---

## 3. What Was NOT AI-Generated

The following contributions are entirely the team's own work:

- Experimental design: The four-tier architecture (encoders, GPT-4o, other LLMs, hybrid gatekeeper) was the team's original design. The hybrid gatekeeper concept, the choice of DeBERTa as the gate, and the ensemble disagreement variant (v5) were the team's ideas.
- The annotation ceiling finding: The discovery that GPT-4o scores only 63% on ensemble-disagreement-gated rows and the interpretation of this as annotation ambiguity rather than model failure emerged from running the experiments and was not suggested by an AI tool.
- Prompt design (P1-P5): The five NLI prompt strategies were designed by the team through iterative dev-set evaluation. The documented prompts in PROMPTS.md are the team's own.
- Result verification: All numbers in the report were manually cross-checked against the source CSVs in results/.
- Interpretation and analysis: All discussion sections, error case studies, and business implications reflect the team's own reasoning about the results.

---

## 4. AI Tools Used

| Tool | Provider | Primary Use |
|------|----------|-------------|
| Claude (Sonnet 4.5) | Anthropic | Code generation, script drafting, figure debugging |
| ChatGPT (GPT-4o) | OpenAI | Code generation, alternative implementations, debugging |

Both tools were accessed via their standard web interfaces (claude.ai and chat.openai.com). No API-based automation was used for code generation - all prompts were entered manually.

---

## 5. Summary Statement

AI coding assistants were used extensively to accelerate code development across all pipeline scripts. The nature of use is best described as collaborative development: the team specified what was needed, the AI generated a working draft, and the team iterated, corrected, and validated the outputs.

The experimental design, intellectual contributions, novel findings, and result interpretation are the original work of the project team. The use of AI tools did not substitute for understanding, it accelerated execution.

---

Declared by: Ronit Jitesh, Luis David, Muskan, Famul, Bipasha, Deepansh
Date: 18 March 2026
Module: LLM-based NLP | MSc Business Analytics | University of Edinburgh
