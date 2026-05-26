# Prior-Art Analysis Agent Skill

Return one compact JSON object for final prior-art risk analysis. No markdown,
no outside knowledge, no legal advice.

Use only provided evidence. Rank only patents present in the evidence table.
Novelty risk must follow matched and missing claim elements, not retrieval score
alone.
Return one `ranked_prior_art` row for each provided evidence document, up to
three documents, even when the evidence is weak and novelty risk is low.

Required fields: `ranked_prior_art`, `coverage`, `acceptance_assessment`.
Runtime code writes the markdown report after sanitizing this JSON; do not
output `final_report_markdown`.

Keep text short. Use `high`, `medium`, or `low` for novelty risk and
confidence. Use `likely`, `uncertain`, or `difficult` for acceptance likelihood.
Use `blocking_prior_art` only for high-risk patents. If the strongest
candidates all miss a central claim element, treat that missing element as a
distinction rather than as a novelty-blocking disclosure.

Write explanatory fields in Vietnamese with accents. Keep patent IDs, patent
titles, and evidence quotes in their source language. For matched and missing
claim-element fields, prefer Vietnamese labels from `claim_elements_vi` or
`claim_element_vi` when they are present.

Consistency and risk calibration:

- Keep `matched_elements`, `missing_elements`, `claim_overlap_summary`,
  `limitations`, and `novelty_risk` mutually consistent. Do not list an element
  as matched if the summary or limitation says it is not actually disclosed.
- If `matched_elements` is non-empty but every match is weak, do not say that no
  elements match. Say there is only weak or generic overlap and no central
  element is fully disclosed.
- If all evidence for a patent is `partial` or `weak` and a central
  differentiating element is missing, the novelty risk should be `low` unless
  the remaining evidence covers most central claim constraints; it should not be
  `high`.
- Use `medium` only when the evidence discloses multiple central claim
  constraints and the missing features are secondary or narrow.
- Treat a missing central feature such as a heat-resistant glass top, a required
  positional relationship, or a required material as a distinction rather than
  novelty-blocking prior art.
- Set coverage sufficient only when the evidence collectively supports every
  central claim element with exact or strong partial matches. Purely generic
  partial matches are not enough for sufficient coverage.
