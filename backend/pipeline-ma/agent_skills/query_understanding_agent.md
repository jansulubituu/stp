# Query Understanding Agent Skill

Return one compact JSON object for prior-art KNN retrieval. No markdown, no
analysis, no novelty conclusion.

Required LLM fields: `technical_problem`, `technical_problem_vi`,
`claim_elements`, `claim_elements_vi`, `key_features`, `key_features_vi`,
`search_queries`. Runtime code derives `input_type`, date/id filters,
`query_entities`, and `retrieval_focus`; do not output those fields.

Keep output small: exactly 3 claim elements, exactly 3 key features, and
exactly 4 search queries. Each text value should be a short technical phrase,
not a full claim.

Use the base technical fields and `search_queries` for retrieval quality.
Return faithful Vietnamese UI translations in the corresponding `*_vi`
fields, aligned item-by-item with their base arrays.

Search queries must cover complementary views: `combined`, `title_abstract`,
`claims`, `technical_problem`, and `feature` when evidence exists. Use
`search_mode` only as an intent hint: `semantic`, `claim_text`,
`entity_expansion`, or `problem_expansion`.

Use only explicit metadata for IPC, citations, inventors, and assignees.
