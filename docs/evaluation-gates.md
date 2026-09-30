# Evaluation Gates Reference

Samsung's requirements come from [guide.md](../guide.md). This is an internal,
version-specific status check refreshed 30 September 2026, not a Samsung
acceptance decision. Historical measurements do not establish that later code
passes the same gate.

| Gate | Requirement | Current evidence and limit |
|---|---|---|
| G1 | One-command clean container launch and unattended replay | Bundled CPU Compose launched without keys/GPU/host assets; final image completed two queries with networking disabled. The CPU audit completed 200 runs. Linux AMD64 development hardware was exercised, not an arbitrary judge PC. |
| G2 | Retrieval before final transcript on ≥80% of eligible queries, low false triggers | Latest CPU audit: actual search dispatch before final delivery in 118/130 answerable/partial cases (90.77%). Negative controller fixtures are a separate small set, not proof of robust natural false-trigger performance. |
| G3 | Identify/isolate ≥70% of compound utterances | Saved owner-reviewed 100-compound/100-single provider run: 92/100 compounds matched under a one-to-one lexical equivalence metric. Latest outcomes reconstruct 351 successful decomposition checkpoints; these are not a raw billing/call ledger or independent semantic judgments. |
| G4 | ≥85% citation support and zero fabricated document IDs; internal faithfulness target ≥80% | Historical 200-query dashboard audit: 98.64% mean judged faithfulness on 92 factual responses, with 108 abstentions excluded, and lexical citation overlap 99.94%. Latest CPU run has zero invalid IDs but 59/100 answerable cases uncertain/uncited. No fresh latest-release 200-query semantic pass; answer quality remains open. |
| G5 | Late constraints update answers without losing unaffected state or restarting the original query | Versioned sessions, delta queries and presentation-only suppression implemented. Nine checks passed on one real corpus case, plus historical controlled tests. Broader restrictive-refinement accuracy remains unverified. |
| G6 | 100% execution trace coverage with times, triggers, citations, versions and token cost | Latest CPU audit: 200/200 exact execution IDs validated. Provider-free usage/cost is zero; real provider traces have nonzero usage and estimated—not invoiced—cost. |

## Evidence protocol

See [benchmark report](benchmark-evaluation-report.md) for retrieval baseline,
two retrieval ablations, controller ablation, edge-failure analysis and source
artifact names. Raw generated results are shipped in submission bundles, not
tracked as large Git blobs.

- Timing uses observed corpus search dispatch versus actual final-chunk delivery.
  Simulated transcript timestamps alone are not runtime concurrency proof.
- Trace validation filters append-only logs by exact expected execution IDs and
  checks missing, duplicate and malformed rows.
- Citation IDs must come from retrieved evidence. Valid IDs alone do not prove
  that factual assertions are supported.
- Citation overlap and answer-shape matches are explicitly proxies. Report
  answerable abstention, unsupported negative answers and the factual-response
  denominator alongside faithfulness.
- G3 reviewed compound fixtures are separate from the 200 dashboard query set;
  category names/rule previews are not a human count of multi-intent questions.
- Real provider runs use server-side Cerebras GPT-OSS-120B at five requests/minute.
  Unit tests and CI do not make provider requests.
- Input is interval-delivered simulated ASR text. Answer output is not token-streamed.

## Remaining acceptance work

Recover answerable queries without weakening grounding, independently evaluate
current-release semantic/citation support, broaden refinement validation, and
record/upload the human demo video. Launch/trace success must not be presented
as every deliverable being complete.
