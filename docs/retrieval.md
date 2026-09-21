# Retrieval Reference

> ✅ **Standing rule:** cross-encoder reranking is allowed and recommended when it
> fits the latency budget. The ban is specifically on **generative
> LLM-as-judge reranking** (pointwise, listwise, or generation-based relevance
> judgment), which was tested and rejected. Use hybrid dense+sparse retrieval,
> optional cross-encoder reranking, and weighted/nested RRF for fusion.

## Validated architecture: hybrid dense+sparse + weighted RRF

Convergent finding across three independent SemEval-2026 Task 8 system papers
(Sifei, RaguTeam-adjacent findings, AILS-NTUA). Build this, don't second-guess it:

- **Vector database:** LanceDB is the selected local vector store for this project.
  Use it for dense vector search and persistent local indexes over the cleaned
  passage-level Cloud/Govt corpus files.
- **Dense retriever + sparse (SPLADE/ELSER-class) retriever, fused via Reciprocal
  Rank Fusion.**
- Fusion formula: `score(d) = 1/(k + rank_dense(d)) + α · 1/(k + rank_sparse_or_reranker(d))`
- Starting hyperparameters to tune from (not copy blindly — these were tuned on a
  specific corpus): `k ≈ 10–60` depending on corpus size, `α ≈ 0.15–0.5`.
- If your vector DB ships RRF as a built-in reranker (LanceDB, Chroma both do), use
  the built-in rather than hand-rolling it.

## Do NOT do these — tested and rejected by multiple teams

1. **LLM-based reranking.** Pointwise, listwise, and generation-based formulations
   were all tested against plain weighted RRF on the same candidate pool. All three
   lost, at 20–200× the cost and 5–20× the latency. The failure mode: once weighted
   RRF already achieves high top-20 recall, marginal relevance differences are too
   subtle for a general-purpose LLM judge to resolve reliably. Don't build this.

2. **Multi-retriever ensembling once a single retriever is tuned via query
   diversity.** Documented as "the ensemble paradox": as a single retriever improves
   through query rewriting, adding more retrievers (a second dense model, BM25, etc.)
   *degrades* R@5 even while improving R@100. The mechanism: unique documents the
   extra retrievers surface land at average rank 37–54 after fusion — below any
   practical top-10/top-20 cutoff — while the fusion process simultaneously displaces
   borderline-correct hits from the primary retriever. **Diversify via query
   reformulation (multiple rewrites of the same query, fused with nested RRF — see
   below), not via multiple retriever backends.**

3. **Passage-Informed Rewriting** (feeding first-pass retrieval results back into the
   query rewriter for a second-stage query). Tested and found to never exceed its own
   seed quality once first-stage R@100 exceeds ~0.90 — at that point the bottleneck is
   ranking precision, not coverage, and broadening the query just adds fusion noise.

4. **Weak/uncalibrated query rewriters.** A rewriter with insufficient
   instruction-following (tested: FlanT5) scored *below* no rewriting at all in one
   study. Don't assume any small model is adequate for query rewriting — verify against
   a no-rewrite baseline before trusting a smaller/cheaper rewriting model.

5. **Semantic re-chunking, document-level chunking, or passage packing.** All tested
   against simple fixed-window chunking; none beat it. Use fixed-window chunking
   (~512 tokens, ~100 token overlap) and spend your engineering time elsewhere.

## Nested (two-level) RRF — use this if fusing multiple query reformulations

If your Multi-Intent Decomposer or query-rewriting step produces several
reformulations of variable reliability (e.g., a precise coreference-resolution
rewrite plus a broader hypothetical-document-style rewrite), don't fuse them as equal
peers with flat RRF. Instead:

- **Level 1:** pre-aggregate the noisier/high-variance reformulation strategies into
  one "weak consensus" ranking.
- **Level 2:** fuse that consensus ranking with your stable, high-precision
  reformulation(s) via weighted RRF.

This prevents noisy strategies from drowning out reliable ones in the final ranking.
Measured gain over flat RRF: ~1.8 points, larger on corpora with more informal/varied
language.

## Retrieval recall vs. reranking sophistication

Across the papers reviewed, roughly 85% of hard retrieval failures were **recall**
failures (the relevant passage never made it into the candidate pool), not reranking
failures. **Prioritize retrieval recall (better queries, better fusion) over reranker
sophistication when debugging.** If your system is underperforming, check recall@100
before tuning your reranker.

## Query rewriting

- Use an LLM (not a small/weak model, not classical pseudo-relevance feedback — PRF
  collapsed to very low recall in testing due to query drift) for standalone-query
  rewriting.
- Base formulation: `concat(last-turn ∥ standalone-rewrite)`.
- Include conversation history in the rewriter's context — but expect diminishing
  returns; performance from including history typically saturates around 4–6 prior
  user turns, with assistant turns adding little further benefit.

## What NOT to over-invest in

- A cross-encoder reranker helps (validated, use it) but is not where your biggest
  gains come from — query rewriting and hybrid fusion contribute far more.
- Don't chase exotic chunking strategies — see point 5 above.
