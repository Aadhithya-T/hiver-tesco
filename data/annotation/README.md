# Golden Set Annotation Workspace (Phase 2)

This directory contains the human-editable annotation templates, candidates, and instructions for producing the Phase 2 golden dataset.

## Files
- `golden_candidates_200.csv`: Primary human-editable CSV template. Open in any spreadsheet editor (Microsoft Excel, Google Sheets, LibreOffice Calc, or VS Code CSV editor).
- `golden_candidates_200.jsonl`: JSONL version of the candidates with full message lists and metadata.
- `golden_candidate_manifest.json`: Provenance metadata recording sample size, seed, stratification counts, and timestamp.

## How to Hand-Label
1. Read `docs/annotation_guidelines.md` carefully before starting.
2. Open `golden_candidates_200.csv`.
3. For each conversation row, read the formatted conversation thread in column `dialogue_text`.
4. Fill in the empty columns:
   - `primary_intent`: One of the 8 draft taxonomy labels (`delivery_and_orders`, `product_quality_and_safety`, `stock_and_availability`, `store_experience_and_staff`, `pricing_promotions_and_vouchers`, `clubcard_and_loyalty`, `website_and_app_technical`, `general_feedback_and_chitchat`).
   - `escalation_needed`: Either `escalate` or `do_not_escalate`.
   - `escalation_rationale`: Required if escalating (`requires_pii_or_dm`, `refund_or_compensation`, `product_safety_investigation`, `formal_complaint`, `technical_support`), otherwise `none`.
   - `evidence_notes`: Generic notes only (e.g. `[Order # provided]`). **DO NOT copy customer names, emails, postcodes, or order numbers.**
   - `ambiguity_notes`: Optional reviewer comments.
5. Save the file.
6. Validate your annotations:
   ```bash
   python scripts/validate_annotations.py --input-path data/annotation/golden_candidates_200.csv
   ```
