# Benchmarking and evaluation report — submission working copy, 30 September 2026

## Evidence versions and current limits

The tables below describe saved runs dated 29 September, not a new evaluation of
the latest Docker/frontend implementation. Their matched baseline and ablations
remain useful, but do not prove that every current gate passes.

The later dashboard audit (`g4-dashboard-200-judge-final-report-20260930.json`)
contains 200 unique cases: 92 factual responses and 108 abstentions. Mean judged
faithfulness on factual responses was 98.64%; citation overlap was 99.94%, a
lexical heuristic rather than independent semantic citation support. Among 100
answerable cases, 33 abstained; 12 negative-labelled cases returned citations.
These behavior failures prevent treating the high grounding mean as overall
answer quality. Abstentions are excluded from factual-claim scoring.

Subsequent changes retain source lists and labelled measurement cells, resolve
conversation references more conservatively, reject explicit jurisdiction
mismatches, and semantically filter provider-selected exact spans. Two live
checks recovered the online-services and planet-day-length queries with valid
retrieved citation IDs. This is a targeted regression check, not a fresh
200-query semantic score. Current CPU execution measurements appear below.

G3 intent equivalence remains a lexical proxy on owner-reviewed examples, not a
blind semantic judgment. G5 has controlled continuity tests and one historical
corpus smoke case, not a broad accuracy study. Current CPU trace coverage is
verified against exact execution IDs, independently of historical GPU runs.

### Release verification (30 September)

The full regression suite passed **275 tests, with two environment-dependent
skips**, including the final provider-date guard. Clean CPU Compose startup without `.env`, model
downloads, GPU access or host corpus/index mounts reached readiness in **24.1
seconds** on the development PC. The original raw corpus remains **122,049
passages**; the packaged, pre-existing cleaned dense index and sparse loader have
**101,763 passages**. Packaging did not rebuild or shrink either source.
The final image additionally completed two cited queries under `--network none`,
with no mounts, keys or GPU; both exact execution traces passed validation.
Subsequent source-only HTTP hardening passed **36 focused dashboard/security and
packaging tests**. The full suite and binary image were not rerun/rebuilt after
that change; the original image evidence retains its earlier version scope.

An actual 12-query browser replay saved screenshots and all visible fields. It
exposed topic-only offline answers; the dashboard now requires direct evidence
support, trading answer recall for fewer misleading responses. A five-query
Cerebras browser replay observed both provider calls and zero-call routing, and
found a date-constraint failure in a changelog. A mandatory date check rejects
undated or differently dated selected fragments. Its live regression correctly
withheld the unsupported GitLab-date claim while retaining the cited Delivery
Pipeline answer. Do not treat the pre-fix screenshots as a final pass.

Final-image session checks passed all nine checks on one real corpus case:
immutable parent state, no retrieval for presentation, retained citations, valid
version lineage, delta-only retrieval and exact prior-answer preservation during
an additive storage follow-up. This is one successful case, not broad refinement
accuracy. Restrictive corrections still require conservative omission when
applicability cannot be verified.

The CPU coverage/behavior audit completed **200 unique queries and 200 valid
execution traces**, with zero invalid citation IDs. Actual search dispatch
preceded final-chunk delivery in **118/130 eligible cases (90.77%)**. Median
pipeline latency was **11.10 seconds**, with a **129.63-second maximum** under
resource contention. This is not semantic scoring. Its frozen runtime snapshot
predates later metadata, refinement and provider-date changes; final-image
checks are recorded separately.

Critically, **59/100 answerable cases were uncertain or uncited**. Of 30 partial
cases, 23 were uncertain or uncited; 3/40 unanswerable cases returned citations;
all 30 underspecified cases were uncertain or uncited. Citation presence alone
does not establish correctness. The stricter offline extraction remains too
conservative to claim answer-quality completion. Raw results, report and exact
traces are packaged; high historical faithfulness must not conceal these failures.

This report uses executable code and saved result rows. G1 and the demonstration video are outside this report. Numbers below are internal measurements, not a Samsung acceptance decision.

## Protocol and baseline comparison

The retrieval ablations used the same first 50 qrel-bearing Cloud tasks, the complete 101,763-passage cleaned Cloud/Govt index, BGE-small on an NVIDIA RTX 3050 6 GB, identical query expansion, 30 candidates per retrieval leg, and top five output. The shared code is `scripts/benchmark_delivery.py`; row-level hits and gold IDs are in `data/curated_dataset/evaluation/retrieval-ablation-20260929.json`. Selecting the first 50 is reproducible but not a randomized or blind test split.

| Retrieval configuration | Success@5 | Mean Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Dense only + cross-encoder rerank (baseline) | 68.00% | 49.15% | 0.5573 |
| Hybrid dense/BM25, no rerank | 62.00% | 44.80% | 0.4707 |
| Hybrid dense/BM25 + cross-encoder rerank | **80.00%** | **58.49%** | **0.5970** |

**Ablation 1, retrieval leg:** with reranking fixed, hybrid improves Success@5 by 12 percentage points and mean Recall@5 by 9.34 points over dense only. **Ablation 2, reranking:** with hybrid candidates fixed, the cross-encoder improves Success@5 by 18 points and mean Recall@5 by 13.69 points. Hybrid without reranking is worse than the dense baseline; fusion alone is not the gain here.

A third, controller-only ablation replayed the same saved timestamped ASR streams with a no-op retriever so only trigger decisions were measured. On 64 eligible Cloud and 88 eligible Govt streams, the default rule controller triggered early on 100% in each domain with 0% false triggers on saved no-retrieval cases. The opt-in BGE-small semantic controller triggered early on 48.44% Cloud and 60.23% Govt, also with 0% false triggers. It remains opt-in because it misses the ≥80% timing bar on these streams. These are controller decisions only; they do not measure retrieved-passage quality. Raw decisions and denominators are in `data/curated_dataset/evaluation/controller-ablation-20260929.json`.

## Three analyzed edge failures

All three are top-five misses in every retrieval configuration above; the IDs and ranks can be checked in the row-level ablation artifact.

1. **"How can I create one?"** The utterance has no antecedent in this query-only retrieval protocol. Gold passages are about a specific object, while the top result is another IBM Cloud topic. The failure is unresolved conversational reference, not a reranker ordering error; a dialogue-aware query is needed before either retrieval leg.
2. **"Pagination options"** Gold `ibmcld_00620-14227-15704` is a Cloudant Go example; the top ranked `ibmcld_02578-1633-3881` discusses generic pagination links. The two-word query omits the product and API, so the system retrieves a plausible but wrong pagination family. The benchmark should preserve surrounding context where available and score this as a miss rather than crediting topical similarity.
3. **"what Keystores are available, please?"** Gold `ibmcld_08597-10745-12722` concerns a cryptographic keystore's `sessionauth` setting; top result `ibmcld_08565-4180-5993` concerns a target keystore assigned to a vault. Both use the same key term in different IBM services. The current query lacks a service qualifier, and neither fusion nor reranking resolves that ambiguity.

## Gate measurements beyond retrieval ranking

| Gate | Independent evidence from this run or saved rows | Result and limit |
|---|---|---|
| G2 | 200 reviewed interval-delivered curated streams, full GPU hybrid/rerank corpus: 90.77% early retrieval on 130 answerable/partial cases; 90.00% by recorded stability point. Controller-only negative streams had 0% false triggers. | Meets the ≥80% timing target on these sets. The two sets have different denominators. |
| G3 | 200 owner-accepted cases: 92/100 compound queries fully matched; 351/351 latest streamed Cerebras decomposition checkpoints succeeded, zero fallbacks. | Meets ≥70%. Intent equivalence uses one-to-one content-token F1, not a semantic human judge. |
| G4 | 200 domain-balanced MTRAG queries (100 Cloud, 100 Govt): 158 factual responses average **94.51% faithfulness** and **99.23% citation support**; 42 abstentions are excluded from claim scoring. 91.14% of factual rows score at least 80% faithfulness and 98.10% at least 85% citation support. A direct audit found all 200 citation-ID sets are subsets of their retrieved context IDs. | Internal threshold pass and zero fabricated IDs in this sample. Citation support is token overlap; faithfulness was judged by Cerebras GPT-OSS-120B against cited passages. |
| G5 | Controlled two-turn replay: final ASR revision supersedes a partial; a targeted delta query replaces an affected claim, keeps an unrelated claim and citation, and increments version; unsupported delta preserves state with uncertainty; presentation follow-up makes no search. A separate full-corpus GPU replay updated an encryption answer for a DDoS-only late constraint using only delta queries and cited a DDoS sentence. | 4/4 controlled checks plus one real-corpus smoke pass. This establishes state behavior and one end-to-end case, not broad corpus-level refinement accuracy. |
| G6 | Full GPU run produced and validated 200/200 JSONL traces, with timestamps, triggers, citations, version lineage, stage latency, token counts, and cost estimate fields. Four additional validated G5 traces show versions 1→4. A live Cerebras provider trace recorded 219 tokens and an estimated $0.00010185. | 100% trace coverage on the measured runs. Dollar estimate uses the [published Cerebras model prices](https://inference-docs.cerebras.ai/api-reference/models/public-models) as of this report, not an invoice. |

The full GPU streaming run also produced a 58.46% qrel citation hit rate and a **47.00% deterministic behavior-match proxy**. That proxy is much weaker than the G4 grounding result and must not be presented as an overall 85% answer-quality score: G4 uses a separate 200-query judged set, while the 200 curated-stream proxy measures task behavior with different assumptions. The full run is in `data/curated_dataset/evaluation/g6-full-streaming-gpu-20260929.json` with its sibling JSONL trace file. G5 records are `data/curated_dataset/evaluation/g5-refinement-20260929.json` and `data/curated_dataset/evaluation/g5-real-corpus-smoke-20260929.json`; the latest G4 dataset, unique 200-row results, and report are `data/curated_dataset/evaluation/g4-current-200-dataset-20260929.jsonl`, `data/curated_dataset/evaluation/g4-current-200-ragchecker-clean-20260929.jsonl`, and `data/curated_dataset/evaluation/g4-current-200-report-20260929.json`. The earlier append-only interrupted attempt log is retained separately and is not used for the reported metrics.

The provider route uses the `.env` Cerebras key and `gpt-oss-120b`, paced at no more than five requests per minute, including retries for hourly-limit responses. The live answer is currently extractive after a Cerebras evidence-span call; it is not token-streamed. The full 200-stream run is deterministic, so its inference token cost is zero. The separate live provider trace demonstrates nonzero usage and price estimation.
