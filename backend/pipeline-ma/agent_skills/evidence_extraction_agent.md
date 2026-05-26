# Evidence Extraction Agent Skill

Return one compact JSON object with `candidate_evidence`. No markdown, no
outside knowledge, no novelty conclusion.

Use only the provided candidate text. Return one row per candidate. For each
row, include at most two strongest `matched_elements`; put the rest in
`missing_elements` when unsupported.

Keep every text field short. Evidence text must be a short quote or close
paraphrase from the provided candidate text. Prefer claims, then abstract, then
mixed snippets. Do not treat retrieval score alone as evidence.

Copy `claim_element` and `missing_elements` exactly from the provided claim
elements so runtime matching stays stable. Runtime code adds Vietnamese display
labels after sanitizing the JSON.

Write explanations (`reason`, `gap_or_limitation`, `overall_relevance`) in
Vietnamese with accents. Keep `evidence_text` in the original source language
because it is grounded evidence.

Match calibration:

- Use `exact` only when the provided text supports all central constraints in
  the claim element: component, material/structure, position, and function.
- Use `partial` only when the text supports the named central component plus at
  least one central function or structure constraint.
- Use `weak` when the text only mentions a generic related component, material,
  or function. Do not count generic burner/porous-body text as `partial` for an
  element that also requires a glass plate or a position relative to glass.
- Positional constraints such as over, under, beneath, below, upstream, or
  downstream are central. If the position is not disclosed, record the gap and
  downgrade the match or put the element in `missing_elements`.
- If a central differentiating feature such as a heat-resistant glass top is
  absent, make that absence explicit in `gap_or_limitation` and
  `missing_elements`.
