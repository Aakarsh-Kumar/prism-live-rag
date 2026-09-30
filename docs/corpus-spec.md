# Corpus Specification — Streaming Live RAG

**Status:** design spec. Nothing here is built yet.
**Authority:** implements the augmentation recipe in `docs/dataset-and-augmentation.md` §4
and the format requirements from §2. Where this doc and `docs/data-working-set.md`
disagree, this doc wins on corpus shape; `data-working-set.md` wins on which files exist.
**Supersedes nothing** — this is additive to `docs/dataset-and-augmentation.md`, which
remains the methodology authority.

---

## 0. Purpose and design constraints

This spec covers three questions: what corpus we need, what the data must look like on
disk, and what text streams we need — sized so the system can be **trained** and **tested**
for robustness, not merely demonstrated.

Binding constraints. Anything that violates these is out of scope regardless of how much it
would help a metric.

1. **80% of test material is traceable to MTRAG-UN**; at most 20% is hand-augmented
   (`dataset-and-augmentation.md` §4). Augmentation transforms structure, never content.
   Every gold answer must remain a real corpus passage so citations stay checkable.
2. **Partial transcripts use the TERTiUS format**, not an invented one. A final ASR result
   *obsolesces* prior partials — state management must handle supersession.
3. **Stability detection uses LocalAgreement-n**, not a hand-tuned confidence threshold.
   Current values: `AGREEMENT_N=2`, `MIN_STABLE_WORDS=3` (`stream.py:12,11`).
4. **Settling time is reported** alongside G2's early-retrieval rate, per SETTLE's framing.
5. **Corpus files stay byte-identical to the official MTRAG distribution.** All cleaning
   happens in `iter_passages()` at load time. Rationale: we can still say "we used the
   official collection," which is worth real credibility with a jury.
6. **No dataset-specific hardcoding.** The `expansion.py` "South America → São Paulo"
   rewrite must not exist in any judged commit.
7. **Human annotation is available** (§5). Where a label requires human judgement, this spec
   asks for it rather than substituting a proxy and calling it ground truth.
8. **Every number is either derived from MTRAG-UN or from a named, cited methodology.** No
   ad-hoc heuristics presented as findings.

---

## 1. Inventory — what we actually have

Measured 2026-09-27. All figures read from the files, not from documentation.

### 1.1 Passage corpora

| | cloud | govt |
|---|---|---|
| passages | 72,442 | 49,607 |
| distinct source pages (url) | 8,578 | 7,661 |
| byte-exact duplicates | 9,937 (13.7%) | 174 (0.4%) |
| mean chars / passage | 1,513 | 1,708 |
| passages containing a raw URL | 71.4% | — |
| passages containing a markdown link | 67.5% | — |
| passages with a doubled first line | 0.0% | 15.2% |
| distinct host sites | — | 560 |
| gold qrel IDs resolvable | 248/248 | 256/256 |

**Do not use the official corpus counts as a checksum.** `data/corpora/README.md` gives
Govt as 8,578 docs / 72,422 passages and FiQA as 7,661 / 49,607 — a two-number exact match
to our *crossed* files. URL inspection settles it: `cloud.jsonl` contains `cloud.ibm.com`
and `ibmcld_*` IDs, `govt.jsonl` contains `*.ca.gov` / `va.gov` / `nasa.gov` and hash-prefixed
IDs. The content is correctly matched; the README table is not a reliable reference. State
this in the report rather than quietly matching a number.

### 1.2 Task corpus — `reference.jsonl`, 507 rows

All `task_type=rag`, `dataset=MT-RAG 2.0`, one row per `conversation_id` (507 distinct, so
task↔conversation is 1:1 and splitting on either is equivalent).

| Collection | rows | with `contexts` | distinct ctx IDs | in local corpus | has qrels |
|---|---|---|---|---|---|
| govt | 157 | 125 | 435 | 435 (100%) | yes |
| ibmcloud | 131 | 86 | 248 | 248 (100%) | yes |
| clapnq | 142 | 108 | 312 | **0** | not cloned |
| fiqa | 77 | 58 | 157 | **0** | not cloned |

`answerability` across the two covered collections (288 rows):

| | ANSWERABLE | UNANSWERABLE | UNDERSPECIFIED | PARTIAL |
|---|---|---|---|---|
| govt | 88 | 27 | 25 | 17 |
| ibmcloud | 81 | 36 | 9 | 5 |
| **total** | **169** | **63** | **34** | **22** |

MTRAG-UN's README states *"the qrels cover the 468 answerable and partially answerable
questions."* Verified against the local files: of 288 covered tasks, **191 have retrieval
qrels and 97 do not — and the 97 are exactly the 63 UNANSWERABLE + 34 UNDERSPECIFIED, with
zero PARTIAL among them.** The absence of retrieval gold is by design, not a gap in the
clone. Two consequences:

- The 97 are the **G4 abstention test set**, already labelled by the benchmark.
- **PARTIAL (22) is the only class usable for both** retrieval and abstention evaluation,
  which makes it disproportionately valuable per example. Do not let stratification
  starve it (§2.2).

`Multi-Turn` in the covered collections: **Follow-up 198, Clarification 46, N/A 44.**
`Question Type`: Non-Question 22, Conversational 2, Composite 28, plus 40/3/28 across all
507 rows respectively.

### 1.3 Unused benchmark fields

These exist in `reference.jsonl` and are read by nothing in the codebase. `data.py:103-107`
keeps only the **final** user turn and discards all prior context.

| Field | Value in covered collections | Used? |
|---|---|---|
| `answerability` | 119 non-answerable/underspecified/partial | **No** |
| `Question Type` | 22 Non-Question, 2 Conversational, 28 Composite | **No** |
| `Multi-Turn` | 198 Follow-up, 46 Clarification | **No** |
| `turn` | 1–8+, mostly 2–7 | **No** |
| `targets` | gold agent answers | **No** |
| `contexts` | gold supporting passages, inline | **No** |
| `input[].metadata.created_at` | real per-turn timestamps | **No** |

**`answerability` is the most valuable unused field in the repo.** 119 labelled
non-answerable / underspecified / partial tasks is a benchmark-provided G4 abstention set —
larger than anything hand-written, and free of the self-grading circularity that makes the
current negatives worthless. `risks-and-open-problems.md` §1 instructs us to "build explicit
test cases for ambiguous/underspecified queries early." They already exist.

### 1.4 Stream fixtures — 231 streams

| | value |
|---|---|
| categories | `early_retrieval` 191, `multi_intent` 20, `no_retrieval` 20 |
| **`refinement`** | **0 — never generated** |
| chunks per stream | mean 10.8, max 40 |
| streams with exactly one `is_final` chunk | 231/231 |
| streams with mid-stream confidence revisions | 231/231 |

### 1.5 What does not exist yet

- **No train/dev/test split.** Zero occurrences in `src/` or `scripts/`. Every task is a
  test task today.
- **No human stability labels.** `dataset-and-augmentation.md` §2c and
  `data-working-set.md` both record this as a known limitation.
- **No training data for any learned component.**
- **No independent negative set** (the current one is circular — §6).

---

## 2. Split discipline — build this first

Nothing else is safe to build until the split exists, because without it every measurement
is contaminated by anything tuned against it.

### 2.1 Unit and ratios

Split on **`conversation_id`** (equivalently `task_id`, 1:1). Never split on passage — a
passage can be gold for more than one task, and passage-level splitting leaks directly into
the retrieval metric.

| split | ratio | covered-collection tasks | purpose |
|---|---|---|---|
| **train** | 60% | ~173 | tuning hyperparameters, fitting the model-based controller, calibrating thresholds |
| **dev** | 20% | ~58 | all tuning decisions; every threshold, `rrf_k`, `α`, agreement-n, extractiveness band |
| **test** | 20% | ~57 | touched **once**, at the end, for reported numbers |

**Test is not a development set.** If a number in the brief was ever produced while looking
at it, it is a dev number and must be labelled as such. Maintain a written record of which
task_ids were inspected.

### 2.2 Stratification

Stratify jointly on `(collection, answerability, phenomenon)` so each split holds the same
proportion of UNanswerable / UNDERSPECIFIED / PARTIAL cases. With 63 unanswerable and 34
underspecified in total, unstratified random assignment would put ~12 unanswerable in test —
too few to measure an abstention rate with any precision.

| stratum | total | train | dev | test |
|---|---|---|---|---|
| ibmcloud ANSWERABLE | 81 | 48 | 17 | 16 |
| govt ANSWERABLE | 88 | 52 | 18 | 18 |
| UNANSWERABLE (both) | 63 | 37 | 13 | 13 |
| UNDERSPECIFIED (both) | 34 | 20 | 7 | 7 |
| PARTIAL (both) | 22 | 13 | 5 | 4 |

Emit `data/splits/{train,dev,test}.json` as sorted `task_id` lists so the assignment is
version-controlled and reproducible. `prism-rag` should take `--split <name>` everywhere it
currently takes `--max-tasks`.

### 2.3 Resolution limit — state this in the report

| split | n | 95% CI half-width on a proportion near 0.5 |
|---|---|---|
| test, all covered tasks | 57 | ±13% |
| test, qrel-bearing only | ~38 | ±16% |
| test, unanswerable only | 13 | ±27% |

**This is the honest reason augmentation is required, not preference.** At n=13 you cannot
distinguish an abstention rate of 50% from 70%. Report a point estimate *with its interval*,
never a bare percentage, and never a percentage from a stratum too small to bound.

---

## 3. Layer 1 — Passage corpus

### 3.1 Required cleaning operations

| Defect | Evidence | Operation | Passages touched | Gold at risk |
|---|---|---|---|---|
| Web-archive crawl index posing as content | 10,186 records from one `webarchive.library.unt.edu/eot2008/query?...&count=10000` pagination URL = **20.5% of govt**; text is year-filter dropdowns and raw query parameters | **Drop** | −10,186 | **0** (verified) |
| Byte-exact duplicate cloud passages | 9,937 | **Drop, keeping the gold-referenced copy of each duplicate group** | −9,926 | 12 protected |
| Markdown links + bare URLs in cloud prose | 18.8% + 0.6% of cloud characters | **Extract from text into a `links` field, and index those links in the sparse leg** | 0 | 0 |
| Per-host nav chrome in govt | lines in ≥60% of a host's own pages | **Strip lines** | 0 | **0 of 256** |

**Measured after implementation** (`prism-rag validate-data`):

| Domain | Raw | Cleaned | Dropped | % dropped | Gold resolved |
|---|---|---|---|---|---|
| cloud | 72,442 | 62,516 | 9,926 | 13.7% | 248 / 248 |
| govt | 49,607 | 39,247 | 10,360 | 20.9% | 256 / 256 |
| **total** | **122,049** | **101,763** | **20,286** | **16.6%** | **504 / 504** |

Govt's 10,360 lands exactly on the predicted 10,186 crawl rows + 174 duplicates. Cloud
came in one passage under the 9,925 prediction because of the gold rule below.

### 3.1.0 Does cleaning actually improve retrieval? Measured, and honestly: barely

The justification for cleaning is retrieval quality, so it was measured rather than
assumed. Paired bootstrap over the 191 qrel-bearing tasks, production `SparseIndex`
(BM25), raw MTRAG text vs cleaned text:

| k | raw | cleaned | delta | 95% CI | tasks gained / lost |
|---|---|---|---|---|---|
| R@1 | 30.9% | 30.9% | +0.0 | [+0.0, +0.0] | +0 / −0 |
| R@10 | 58.6% | 59.2% | +0.5 | [+0.0, +1.6] | +1 / −0 |
| R@100 | 82.7% | 82.7% | +0.0 | [+0.0, +0.0] | +0 / −0 |

**Claim to make in the report:** cleaning is *retrieval-neutral with a marginal gain* — it
removes 20,286 noise passages and loses **zero** gold tasks at every k, but it does not
measurably raise lexical recall. The single R@10 gain is one govt task and the CI touches
zero. Do not claim cleaning improved retrieval.

What cleaning *does* buy, stated honestly: 16.6% fewer passages, a 20.9% reduction on
govt, 504/504 gold preserved, a per-passage audit trail, and an index that no longer
contains 10,186 web-archive query-parameter records. That is corpus hygiene and
reproducibility, not a retrieval win.

Two of the four operations were **actively harmful** when first implemented, and both were
found by this measurement rather than by reading the code:

#### Link extraction was destroying vocabulary, not relocating it

Cleaning moved URLs out of `text` into `Passage.links`, but `SparseIndex` only indexed
`title + " " + text`. The URLs were therefore dropped from the BM25 vocabulary entirely:

| condition | docs | R@1 | R@10 | R@100 |
|---|---|---|---|---|
| raw | 122,046 | 23.6% | 47.1% | 68.6% |
| + crawl-drop + dedup | 101,763 | 24.6% | 48.2% | 69.6% |
| + whitespace | 101,763 | 24.6% | 48.2% | 69.6% |
| + link extraction | 101,763 | 23.6% | 47.1% | 69.6% |
| + chrome at DF 0.30 (original) | 101,763 | 22.0% | 46.6% | 70.2% |

(numbers from the standalone BM25 harness, so the absolute level differs from the table
above; the deltas are what matter)

Fix: `SparseIndex` now indexes `title + text + links`. Whitespace normalisation measured
exactly 0.0 change and is retained only for readability. Crawl-drop + dedup is the one
clear win (+1.0 R@1, +1.1 R@10).

#### The chrome threshold was guessed, and 0.30 deleted real content

Swept against BM25 recall:

| chrome DF | R@1 | R@10 | R@100 |
|---|---|---|---|
| off | 23.6% | 47.1% | 69.6% |
| ≥ 0.30 | 22.0% | 46.6% | 70.2% |
| ≥ 0.40 | 23.0% | 46.6% | 70.2% |
| ≥ 0.50 | 23.6% | 46.1% | 69.1% |
| **≥ 0.60** | **23.6%** | **46.6%** | **69.6%** |
| ≥ 0.75 | 23.6% | 47.1% | 69.6% |
| ≥ 0.90 | 23.6% | 47.1% | 69.6% |

0.30 cost 1.6 points of R@1 and R@10: a line present on 30% of a host's pages is often
genuine content on the remaining pages. ≥ 0.50 is indistinguishable from not stripping, and
≥ 0.90 is exactly chrome-off. **0.60 is chosen**: full recall retained, worst boilerplate
still removed. The threshold is a hyperparameter chosen on this corpus, and the docs
already commit to stating that corpus-specific hyperparameters are a limitation.

### 3.1.1 Gold-vs-gold duplicates — the rule that had to change

12 gold passage IDs **share identical normalised text with another gold ID** — MTRAG-UN
registers the same passage under two IDs. Examples: `ibmcld_02426-5026-7158` /
`ibmcld_09336-5017-7149`, and the `7-1470` / `7-2102` / `306636-308961` pairs.

A naive "keep one copy per duplicate group" rule drops one gold ID from each pair, and
`validate_domain` fails with missing-passage errors. The rule is therefore:

> When a duplicate group contains **two or more** gold IDs, keep **every** gold copy.

The cost is 12 retained passages out of 101,763 — noise. The alternative is 12 qrels that
retrieval can never satisfy, which is a permanent recall ceiling.

### 3.1.2 Gold protection is automatic, not opt-in

`iter_passages()` loads the domain's qrel IDs and protects them **by default**.
`retrieval.py` (both index-build paths) and `evaluation.py` had no way to opt in, and the
first index build silently contained 101,751 passages — exactly 12 short. Protection is
now the default so no consumer can forget it; pass `protected_ids=()` to opt out.

### 3.1.3 Order of operations: chrome before links

Chrome detection is **line-based**, and link extraction normalises whitespace, which
collapses the newlines chrome detection keys on. Stripping links first silently disabled
chrome removal on every real passage. The required order is:

1. `_strip_chrome(raw_text)` — on the original line structure
2. `extract_links(text)` — on what survives
3. `_normalize_whitespace(text)` — horizontal runs to one space, blank-line runs to one
   blank line, single newlines preserved

Step 3 must use `[ \t]+` and **not** `\s+`. A `\s+` collapse also matches newlines and
flattens every passage to a single line, which discards the paragraph structure raw MTRAG
carries and makes the blank-line collapse dead code.

`chars_removed` accumulates across all three operations, with `whitespace` reported
separately so link removal and tidy-up are distinguishable. It does not count deduplicated
text, which is a whole-record drop rather than a character edit.

### 3.1.4 Whitespace normalisation is retained for readability, not retrieval

Measured contribution: **exactly 0.0** at every k. It is kept because cleaned passages
should be legible for citation display and because it is free, not because it helps
retrieval.

### 3.2 Chrome detection must be per-host

Chrome is site-specific. A global blocklist catches almost nothing:

- `www.va.gov` (2,271 passages) — **zero** lines common to even 30% of its own pages
- `www.nasa.gov` (1,889) — emits `Highlights`, `4 min read`, `5 min read`, `article`
- `webarchive.library.unt.edu` (10,553) — emits `UNT Web Archive`, `3 versions`, `4 versions`

`Search` / `Home` / `Contact Us` appear in 5–12% of passages each, which means they are not
global boilerplate — they are the chrome of a minority of sites. Any global threshold either
deletes real content or deletes nothing.

**Rule:** for each URL host with ≥10 passages, compute document frequency of every line of
length 2–60; a line is chrome for that host iff it occurs in ≥30% of that host's passages.

### 3.3 Scale: do we need more passages?

**No — not for retrieval evaluation.** 102k cleaned passages against 191 qrel-bearing tasks
is already a hard retrieval problem (measured recall@5 ≈ 0.47). Adding pages makes the
metric harder without making the system better. `dataset-and-augmentation.md` §1 is right
that retrieval quality is the ceiling on everything downstream.

ClapNQ (183,408 passages, 4,293 docs) and FiQA (7,661 / 49,607) are downloadable and not
cloned. **Do not add them for retrieval.** A corpus that large with 191 queries measures
mostly your embedding model. Add ClapNQ only if you want a **document-level** test (see
§9.3), where 4,293 documents is the variable of interest.

### 3.4 On-disk format (unchanged)

MTRAG's own schema, byte-identical (constraint §0.5). `data.py:66-75` reads `_id`, `text`,
`title`, `url`. Leave it.

```json
{"_id": "ibmcld_00422-0-387", "id": "ibmcld_00422-0-387", "url": "https://cloud.ibm.com/docs/CDN?topic=CDN-set-up-web-app-video", "text": "\n\n\n\n  Video - Setting up a web app...", "title": ""}
```

### 3.5 Canonical cleaned format (produced by `iter_passages()`, consumed by indexer)

```json
{
  "_id": "ibmcld_00422-0-387",
  "id": "ibmcld_00422-0-387",
  "domain": "cloud",
  "url": "https://cloud.ibm.com/docs/CDN?topic=CDN-set-up-web-app-video",
  "title": "",
  "text": "Video - Setting up a web app with security groups, load balancer & CDN. IBM Cloud network services make it easy to build cloud-native applications.",
  "text_raw_sha256": "3f9a…",
  "links": ["https://cloud.ibm.com/docs/CDN?topic=CDN-set-up-web-app-video"],
  "cleaning": {
    "dropped_duplicate_of": null,
    "stripped": ["markdown_links", "nav_chrome"],
    "chrome_host": null,
    "chars_removed": 412
  }
}
```

- `text_raw_sha256` — hash of the pre-cleaning text, so every cleaning claim is auditable
  and reversible from the untouched source file.
- `links` — extracted rather than deleted: a URL is real information about a product page
  even when it is noise inside a sentence.
- `cleaning` — per-passage provenance. A judge asking "how did you decide what to remove"
  gets a field-level answer.
- `dropped_duplicate_of` — set only on a gold copy that displaced a non-gold incumbent.
  Recorded because `duplicate_of` is redirected to the gold id, so the loser is otherwise
  unrecoverable.

`cleaning` and `links` are optional; the loader must tolerate their absence so raw MTRAG
files load unchanged.

**In-memory `Passage`.** The dict above is the on-disk canonical form. `iter_passages()`
flattens it into the frozen `Passage` dataclass (`src/prism_live_rag/models.py`):

```python
Passage(
    id, domain, text, title, url,
    links: tuple[str, ...],
    text_raw_sha256: str,
    dropped_duplicate_of: str,
    cleaning: tuple[tuple[str, str], ...],   # e.g. (("chars_removed", "412"),
                                            #       ("stripped", "nav_chrome"))
)
```

`cleaning` values are coerced to `str` because `stripped` is a list and `chrome_host` can
be `None`; neither would keep the frozen dataclass hashable, and the field is documented as
hashable. `retrieval.py:150` constructs `Passage` without these fields, so they default to
empty at read time — provenance is a loader concern, not a retrieval concern.

---

## 4. Layer 2 — Task corpus

### 4.1 MTRAG-UN's four phenomena as the organising taxonomy

Per `dataset-and-augmentation.md` §1, use these as the top-level label on every task:

| Phenomenon | Source field | n (covered) | Gate |
|---|---|---|---|
| **UNanswerable** | `answerability: UNANSWERABLE` | 63 | G4 — must abstain |
| **UNderspecified** | `answerability: UNDERSPECIFIED` + `PARTIAL` | 56 | G4 — must ask, not guess |
| **NONstandalone** | `Multi-Turn: Follow-up` + `Clarification` | 244 | G2/G3 — needs prior context |
| **UNclear responses** | `targets` where the agent declines or hedges | to be counted | G4/G5 |

Every task in every layer below carries a `phenomenon` label from this set.

### 4.2 Use the conversation, not just the final turn

`data.py:103-107` currently keeps only `user_turns[-1]`. For 244 of 288 covered tasks that
throws away the context that makes the question interpretable — the NONstandalone case,
which is precisely what G2/G3 grade.

Add a `context_turns` field to the task record holding prior user turns (capped, most recent
first). Use them for (a) building streams, (b) the `concat(last-turn ∥ standalone-rewrite)`
query formulation that `architecture.md` §② records as a validated zero-latency gain, and
(c) the underspecified-question test, where the whole point is that the *prior* turn
supplies the missing constraint.

### 4.3 The clapnq / fiqa unlock — and its trap

Inline `contexts` hold 1,152 distinct passage IDs. For govt and ibmcloud they are a **100%
subset** of what we already have (435/435, 248/248) — worthless for corpus building. For
clapnq and fiqa the intersection is **0**, so they are the only passages we have for those
219 tasks (43% of `reference.jsonl`).

- **Permitted:** answerability / abstention / generation-side evaluation, where gold context
  is supplied and corpus size is irrelevant.
- **Prohibited:** any retrieval metric. A 312-passage corpus containing every gold answer
  returns recall 1.0 and measures nothing. **A number computed that way must never appear
  next to a 72,442-passage number.**

This trap is worth stating explicitly in the report, because the resulting figure would look
excellent and mean nothing.

---

## 5. Layer 3 — Human annotation

This is the layer that makes the G2 claim non-circular, and it is the only part of this spec
that cannot be automated. `dataset-and-augmentation.md` §4 step 3 requires a human judgement
of whether intent is "stable" at each prefix; §2c and `data-working-set.md` both record that
this is currently approximated by LocalAgreement-n and flag it as a known limitation.

**Why the proxy is not good enough.** LocalAgreement-n defines stability as *n consecutive
updates agreeing on a prefix* — a word-agreement policy, not a human judgement of intent.
"I heard the toolchain" is four agreeing words and no answerable question. A prefix can be
perfectly stable by word count and semantically incomplete.

### 5.1 Volume

| | value | rationale |
|---|---|---|
| streams annotated | **100** | ±10% CI half-width at n=100; n=231 would give ±6.5% but is 2.3× the work |
| chunks per stream | **6** | 4 partials spanning pre/post-stability + the final + 1 margin |
| total judgements | **~600** | |
| annotators | **2**, disjoint 50/50 | |
| overlap subset | **20 streams** (20%) | for inter-annotator agreement |

Stratify the 100 across both domains, all stream categories, and both sides of the
`stability_chunk_index` boundary — the disagreement cases are the interesting ones and
random sampling under-samples them.

### 5.2 Sidecar format

A sidecar file, not a rewrite of the committed fixtures. Fixtures stay byte-stable and
regenerable from seed 0; the human layer is versioned separately and can be extended.

`data/annotations/stability/{domain}.jsonl`:

```json
{"stream_id": "cloud-0000", "domain": "cloud", "annotator_id": "a1",
 "labels": [
   {"chunk_index": 0, "stable": false, "note": "single token, no intent"},
   {"chunk_index": 1, "stable": false, "note": ""},
   {"chunk_index": 2, "stable": false, "note": "subject only, no predicate"},
   {"chunk_index": 3, "stable": true,  "note": "full proposition: subject+verb+negation"},
   {"chunk_index": 4, "stable": true,  "note": ""},
   {"chunk_index": 9, "stable": true,  "note": "final, trivially stable"}
 ],
 "label_source": "human"}
{"stream_id": "cloud-0000", "domain": "cloud", "annotator_id": "a2",
 "labels": [ … ], "label_source": "human"}
```

- `labels[].stable` — boolean. Also allow `"unclear"` as a third value; forcing a binary
  judgement on genuinely ambiguous prefixes manufactures fake agreement.
- `note` — free text, optional, but capture the reasoning where it disagrees with
  LocalAgreement-n. Those cases are the most valuable thing in the dataset.
- One row per `(stream_id, annotator_id)`. Never merge annotators into one file.

### 5.3 Derived artefact

`data/annotations/stability/agreement.json`:

```json
{"overlap_streams": 20, "labels_compared": 118,
 "raw_agreement": 0.847, "cohen_kappa": 0.79,
 "disagreements": [{"stream_id": "cloud-0042", "chunk_index": 3,
                    "a1": true, "a2": false, "note": "constraint implied by verb tense"}]}
```

Report `cohen_kappa`, not just raw agreement — with a skewed base rate (most prefixes are
unstable early and stable late) raw agreement flatters itself.

### 5.4 Consequences for reporting

Set `stability.label_source = "human"` on annotated streams, then report G2 **twice**:

1. against computed LocalAgreement-n labels (n=231, all streams)
2. against human labels (n=100, annotated subset)

**If the two diverge, that divergence is the most interesting number in the report.** It
quantifies exactly how much of the current G2 claim rests on a word-count heuristic. A small
divergence is a good result and should be stated as such — it means the cheap policy is
adequate, which is a finding rather than a failure.

---

## 6. Layer 4 — Independent negative set

### 6.1 The current one is provably worthless

`stream.py:27` `NO_RETRIEVAL_PHRASES` holds 10 phrases. `controller.py:430`
`_is_presentation_only` is `lowered.startswith(command)` over `PRESENTATION_COMMANDS`
(`controller.py:147-167`).

**All 10 negatives match by prefix — 10/10.** `false_trigger_rate` is 0.0 by construction and
could never be anything else. It is not evidence of anything. A test that cannot fail is not
a test.

### 6.2 Replacement

**(a) Benchmark-provided — primary.** `reference.jsonl` rows with `Question Type:
Non-Question` (22 in the covered collections) or `Conversational` (2). Real human utterances
carrying no retrievable intent. Convert each to a TERTiUS stream as in §7. Category
`no_retrieval`, `provenance.derivation: "benchmark"`.

**(b) Natural backchannel — secondary, hand-authored.** 20–30 items: affirmations,
hesitations, topic-adjacent fragments with no question, self-repair ("no wait, sorry").
**Constraint: no item may begin with any prefix in `PRESENTATION_COMMANDS`.** Enforce with an
assertion in the generator so the check cannot silently regress.

**Acceptance:** `prism-rag validate-streams` prints a substring scan proving zero prefix
overlap with `PRESENTATION_COMMANDS`. If any overlap, validation fails.

---

## 7. Layer 5 — Text streams

The layer Samsung grades and the one MTRAG-UN cannot supply. Both benchmarks evaluate
turn-complete utterances; neither has a partial transcript.

### 7.1 Categories

| Category | target | source | Gate | status |
|---|---|---|---|---|
| `early_retrieval` | 191 keep | real MTRAG-UN query → TERTiUS expansion | G2 | exists |
| `no_retrieval` | 20 → **45** | §6 | G2 | exists, **circular** |
| `multi_intent` | 20 → **50** | 2–3 **related** questions, same conversation | G3 | exists, needs rework — §7.3 |
| `refinement` | **40** | real multi-turn conversation + inserted constraint turn | G5 | **missing — build this** |
| `answerability` | **60** | `answerability` ∈ {UNANSWERABLE, UNDERSPECIFIED, PARTIAL} | G4 | **missing — free from the benchmark** |
| `underspecified` | **40** | `Multi-Turn: Clarification` (46 available) | G2/G4 | **missing — free** |

### 7.2 `refinement` — the G5 stream

Per `dataset-and-augmentation.md` §4 step 5 and `architecture.md` §"Session refinement":

1. Take a real multi-turn conversation (244 Follow-up/Clarification rows available).
2. Pick the turn where a genuine constraint was added.
3. Insert a **constraint-bearing turn not present in the original data**, late enough that a
   controller waiting for `is_final` misses it.
4. Label the expected classification and which claims must be superseded.

The guide's travel-reimbursement example carries **two** constraints in one chunk
("international **and** after travel"). A single-delta design fails its own acceptance test,
so `delta_constraints` is a **list**, and **≥40% of fixtures carry 2+**.

```json
{
  "stream_id": "govt-refine-0003", "task_id": "…", "category": "refinement",
  "domain": "govt", "qrel_passage_ids": ["…", "…"],
  "stability_chunk_index": 4, "settling_ms": 1980, "sub_intents": [], "chunks": [ … ],
  "refinement": {
    "base_stream_id": "govt-0042", "base_turn": 2, "inserted_before_turn": 3,
    "delta_text": "and the trip was international, booked after travel",
    "delta_constraints": ["international", "booked_after_travel"],
    "expected_classification": "modify", "expected_behaviour": "patch",
    "must_not": ["full_retrieval", "restart"]
  }
}
```

`must_not` is a testable assertion, not a comment: if the pipeline issues a full-corpus
re-retrieval, the fixture fails. That is the entire difference between "refines" and
"restarts," and nothing currently tests it.

### 7.3 `multi_intent` — the current fixtures are not examples

`stream.py:268` builds them by joining unrelated queries with `" . "`. The result is not a
compound utterance but two questions with punctuation between them — and
`evaluation.py:177` then **excludes the category from scoring entirely**. It is decorative.

`dataset-and-augmentation.md` §4 step 4 is correct and not yet followed: combine 2–3 real
questions **from the same domain/conversation** into one natural utterance. Target 50, each
sub-intent carrying its own gold so the hit criterion can require *each* sub-query's
retrieval to surface its own passage. `SubIntent` (`stream.py:41`) already exists and is
just not populated meaningfully.

### 7.4 Stream JSON schema

Required keys are exactly those `SimulatedStream.from_dict` reads today, so old fixtures
still load. New keys are additive and must be read with `.get()` defaults.

```json
{
  "stream_id": "cloud-0000",
  "task_id": "b3b321e9ea81d1d90e528f85fff72d63<::>8",
  "category": "early_retrieval",
  "domain": "cloud",
  "split": "train",
  "qrel_passage_ids": ["ibmcld_00691-7371-8668"],
  "stability_chunk_index": 3,
  "settling_ms": 1710,
  "sub_intents": [{"text": "…", "stable_at": 3}],

  "chunks": [
    {"timestamp_s": 0.43, "text": "I", "is_final": false, "confidence": 0.51},
    {"timestamp_s": 3.76, "text": "I heard the toolchain is not available",
     "is_final": true,  "confidence": 0.96}
  ],

  "phenomenon": "NONstandalone",
  "answerability": "ANSWERABLE",
  "question_types": ["Factoid"],
  "multi_turn": "Follow-up",
  "base_utterance": "I heard the toolchain is not available in South America.",
  "context_turns": ["…"],

  "stability": {
    "policy": "local_agreement_n", "agreement_n": 2, "min_stable_words": 3,
    "label_source": "computed",
    "human_label_file": "data/annotations/stability/cloud.jsonl",
    "human_annotated": false
  },

  "asr": {
    "engine": "simulated", "revisions": 4,
    "supersedes": [0, 1, 2], "superseded_by": [3],
    "obsoletes_prior_partials": true,
    "words_per_second": 2.5, "punctuation_pause_s": 0.32,
    "revision_log": [{"from": "working fire me", "to": "working from me", "chunk_index": 3}]
  },

  "provenance": {
    "source": "mtragun", "collection": "ibmcloud",
    "derivation": "auto", "seed": 0, "recipe_step": 2
  }
}
```

`refinement` (§7.2) is present only on refinement-category streams. All other new keys are
optional.

### 7.5 `asr.supersedes` must be explicit

TERTiUS's defining detail per `dataset-and-augmentation.md` §2a is that **a final ASR result
obsolesces all prior partials**. The fixtures have `confidence < 0.99` on mid-stream chunks
(good) but no field marking which chunk superseded which. The controller's state management
must handle a partial being *replaced*, not only appended; nothing tests that path today.

The `CONFUSIONS` map at `stream.py:14-25` is where revision content comes from — keep it, and
log pre/post text for every revision so SETTLE's finding ("a naive system can fire on a
transcript before it's corrected") is reproducible.

### 7.6 Settling time

`settling_ms` already exists per stream (`stream.py:205`). Report the distribution, not just
the mean: p50 / p90 / p99, split by whether the controller fired before or after settling.
SETTLE found p99 scales ~3.5× with the ASR engine's max-delay setting — that spread is the
interesting quantity and a mean hides it.

---

## 8. Layer 6 — Training data

### 8.1 What actually needs training

| Component | Trained? | Decision |
|---|---|---|
| Dense encoder (BGE-small) | no | off-the-shelf, frozen |
| Cross-encoder reranker (`ms-marco-MiniLM-L-12-v2`) | **no** | already trained on MS MARCO. **Retraining on 191 queries would be strictly worse.** Say so in the report. |
| `RuleBasedRetrievalController` | no | rule-based; tune thresholds on dev only |
| `SemanticRetrievalController` (ablation) | **yes** | needs the §5 labels |
| Answerability / abstention classifier | **yes** | needs the 119 `answerability` labels |
| RRF `k`, `α`, candidate depth | no | dev-set sweep, reported as a sweep |
| Answerability confidence threshold | no | dev-set calibration |

`architecture.md` §① already frames the rule-vs-model controller as the required ablation.
Training the semantic controller is what makes that ablation real rather than rhetorical.

### 8.2 Training examples from the human labels

Each annotated chunk becomes one training example:

```json
{"stream_id": "cloud-0000", "chunk_index": 3, "domain": "cloud",
 "text": "I heard the toolchain is not available",
 "n_agreements": 2, "committed_prefix_len": 5, "confidence": 0.76,
 "label_source": "human", "label": 1}
```

- **Features are the ones the controller already has** — committed prefix length, agreement
  count, confidence, chunk timing. Do not add features the runtime does not have, or the
  classifier will not be deployable.
- **~600 examples from §5.1**, minus the 20-stream overlap used only for agreement
  measurement → ~540 training examples.
- Adequate for a **frozen-encoder logistic regression or gradient-boosted trees**, which is
  the right capacity for this problem and defensible in a brief. A fine-tuned transformer on
  540 examples would overfit and be harder to defend than the rule-based baseline.
- Report the ablation as: rule-based vs logistic-regression vs
  frozen-encoder+MLP, all three on the same test streams.

### 8.3 Abstention classifier training set

| class | source | n (covered) |
|---|---|---|
| positive (answer) | `answerability: ANSWERABLE` | 169 |
| negative (abstain) | `UNANSWERABLE` | 63 |
| negative (ask) | `UNDERSPECIFIED` | 34 |
| negative (partial) | `PARTIAL` | 22 |

284 examples is thin for a learned abstention model, and `risks-and-open-problems.md` §1
records that top-ranked systems on the closest benchmark achieved only **21.8% recall on
genuinely unanswerable questions** — this is a hard, structurally biased problem, not a
prompt bug. So:

- Fit on the 284 available examples, evaluate on the held-out test split only.
- **Do not claim a solved abstention gate.** Report the confusion matrix and be explicit
  that recall on unanswerable is the metric that matters and the one the literature says
  will be low.
- If the classifier does not beat a constant-abstain baseline on the test split, report that
  and keep the rule. A negative result, honestly reported, is better than a model that
  scores well and generalises to nothing.

---

## 9. Layer 7 — Hard negatives and distractors

Not gate-required, but the cheapest available fix if abstention does not move after §3 and
§7.

### 9.1 Ranking hard negatives

For each gold passage, the 5 nearest **non-gold** passages by embedding similarity. The
system must still cite the gold. Cheapest available test of whether ranking does anything.
~504 gold passages × 5 = ~2,500 pairs, generated, not hand-labelled.

### 9.2 Cross-domain distractors

Inject cloud gold-rankers into govt candidate sets and vice versa. Should collapse recall. If
it does not, retrieval is leaking across domains and the reported number is inflated.

### 9.3 Abstention distractors

Passages that are topically adjacent but do **not** contain the answer. This is the case the
system currently fails: the demo answer for "is the toolchain available in South America" is
four topically-adjacent IBM sentences that never address the question. A grounded non-answer
still scores zero on functionality (30%).

**Source:** `reference.jsonl` rows with `answerability: UNANSWERABLE` have a `contexts` field
that, for those rows, is empty or absent by construction — the benchmark did not supply
supporting passages because there are none. If clapnq/fiqa contexts are added (§4.3), the
same structure applies. So the benchmark hands us correct negative examples directly.

### 9.4 Document-level corpus (optional, ClapNQ only)

If a document-level experiment is wanted, ClapNQ is the right corpus: 4,293 documents is a
variable of interest rather than noise. Do not mix document-level results with
passage-level results in the same table.

---

## 10. What we deliberately do not build

- **Banking / Telco corpora** — not in the clone. Do not build against them.
- **Retraining the reranker** — off-the-shelf is better at n=191 (§8.1).
- **A fine-tuned transformer abstention model** — 284 examples cannot support it (§8.3).
- **A second evaluation harness** — `data/scripts/evaluation/` is the vendored benchmark
  evaluator and stays, LICENSE intact.
- **A second decomposition system** — `RuleBasedDecomposer` is reused for refinement delta
  extraction rather than replaced.
- **Cross-session or multi-session memory** — out of scope for a single-session demo.
- **Real ASR capture.** `risks-and-open-problems.md` §2 and §5 both flag this: the
  controller is validated against simulated partials, not a live engine. Stated as a
  limitation, never silently glossed.

---

## 11. Build order and acceptance checks

Each row states the check that decides whether it is done. Order matters: the split is first
because everything after it is contaminated without it.

| # | Item | Acceptance check | Status |
|---|---|---|---|
| 1 | **Split**: `data/splits/{train,dev,test}.json`, stratified per §2.2, `--split` flag threaded through every eval entrypoint | sums to 288; per-stratum proportions within 1 task across splits; test task_ids listed in a committed file | pending |
| 2 | **Loader cleaning** in `iter_passages()`: archive-index drop, gold-preserving dedup, markdown-link extraction, per-host chrome | `validate-data` still reports 248/248 + 256/256 gold resolved; ≥20,000 passages dropped; 0 gold dropped | **done** — 20,286 dropped, 504/504 gold, index rebuilt at 101,763. Retrieval effect measured and reported as neutral (§3.1.0) |
| 3 | **Human stability labels** — 100 streams × 6 chunks, 2 annotators, 20-stream overlap | `agreement.json` reports raw agreement **and** Cohen's κ; `label_source` set on annotated streams | pending |
| 4 | **Independent negatives** (§6) | `validate-streams` substring scan proves 0 prefix overlap with `PRESENTATION_COMMANDS`, or fails | pending |
| 5 | **`answerability` category** — 60 streams | 60 fixtures; `answerability` ∈ {UNANSWERABLE, UNDERSPECIFIED, PARTIAL}; `provenance.derivation = "benchmark"` | pending |
| 6 | **`underspecified` category** — 40 streams | drawn from the 46 `Clarification` rows; prior turn retained in `context_turns` | pending |
| 7 | **`refinement` category** — 40 streams, ≥40% with 2+ constraints | every fixture has `refinement.must_not`; a full-retrieval run **fails** the fixture | pending |
| 8 | **`multi_intent` rework** — 50 streams, same-conversation related questions | every sub-intent has its own gold passage; category no longer excluded at `evaluation.py:177` | pending |
| 9 | **`asr.supersedes` populated** on all fixtures | every stream has an explicit supersession chain and a revision log | pending |
| 10 | **Hard negatives** (§9.1) | ~2,500 pairs generated; recall on gold-with-hard-neighbours reported separately from clean recall | pending |
| 11 | **Controller ablation** trained on §8.2 | rule vs logistic-regression vs MLP on the same test streams, with the feature set documented | pending |
| 12 | **Phenomenon coverage report** | counts per UNanswerable / UNderspecified / NONstandalone / UNclear, per split | pending |

Row 2 shipped in `src/prism_live_rag/corpus_clean.py` with tests in
`tests/test_corpus_clean.py` (§3.1.1–3.1.4 record the four rules it forced). Full suite:
**130 passed, 2 skipped** (was 44 passed, 2 skipped).

Row 2's acceptance check is met on its own terms, but note it is a *hygiene* check, not a
retrieval check. §3.1.0 measures the retrieval effect and it is neutral. If a reviewer asks
whether cleaning made the system better at answering questions, the honest answer is that
it made the corpus defensible, not that it raised recall.

### 11.0 Streaming prerequisite — done alongside row 2

Full-stream consumption was a prerequisite for row 7, not an independent change.
`StreamingRagPipeline.run()` previously took a `list[TranscriptChunk]` and broke at the
first retrieval, so **no stream ever reached its final chunk** and G5 refinement was
structurally impossible. It now:

- accepts any `Iterable[TranscriptChunk]`, including a generator, so ASR text can arrive lazily;
- consumes every chunk and records a decision for each;
- tracks the last chunk without indexing, removing the `chunks[-1]` assumption.

Full consumption is safe because `RuleBasedRetrievalController` latches after its first
stability fire (`_fired`) and always retrieves on `is_final`, so a normal stream still
yields at most two retrievals. `refine_on_final` now defaults to `True`. Verified on
`cloud-0000`: 11/11 chunks consumed, provisional retrieval at t=3.09s, final at t=6.08s,
answer synthesised from the refined query.

### 11.0.1 Presentation-command detection was broken in both directions

Auditing the streaming change surfaced a pre-existing defect in
`RuleBasedRetrievalController._is_presentation_only` and its duplicate in
`SemanticRetrievalController`: matching was `lowered.startswith(command)` over a tuple that
included bare verbs. It failed symmetrically.

**Under-coverage** — common phrasings were not recognised, so they caused *spurious*
retrieval:

| utterance | before | after |
|---|---|---|
| "can you repeat that" | Retrieve | No-Retrieval |
| "say that again" | Retrieve | No-Retrieval |
| "what did you just say" | Retrieve | No-Retrieval |
| "read that back to me" | Retrieve | No-Retrieval |
| "go over that again" | Retrieve | No-Retrieval |

**Over-coverage** — genuine questions beginning with the same letters were *silently
suppressed*, which is a live recall bug, not a labelling nit:

| utterance | before | after |
|---|---|---|
| "stoplight timing in Denver Colorado" | No-Retrieval | Retrieve |
| "hold on to the waiver requirement" | No-Retrieval | Retrieve |
| "speak up about the enrollment deadline" | No-Retrieval | Retrieve |
| "start over with the new form" | No-Retrieval | Retrieve |

Replaced with one shared `is_presentation_only()`: word-boundary anchored, leading
politeness stripped, and the whole utterance capped at 5 words. 20 suppression cases and 9
must-retrieve cases pass, asserted against **both** controllers so they cannot diverge.
The live demo beat — provisional retrieval, then "repeat that" as the final chunk, prior
citations reused with no new query — is now a test.

Note this also removes the circularity flagged in §11.1: the independent-negative set was
provably worthless because every `NO_RETRIEVAL_PHRASES` entry was already matched by
`PRESENTATION_COMMANDS`. A hand-written negative set cannot be valid until this matcher is
correct, so §6's acceptance check must be re-run after this change.

### 11.1 Re-measure before building anything else

After rows 1, 2 and 5, re-measure **G4 abstention** and **G2 timing**, then stop and compare.

Both current numbers were produced with a broken span extractor and a circular negative set:

- G2 = 0.97 / 0.95, where an "early hit" is *any* provisional retrieval before the final
  chunk (`evaluation.py:179`) — a controller that waited until the last chunk would score
  100%. `stability_chunk_index` is recorded at `:170` and used in **no** hit criterion
  anywhere in the repo.
- G4 = 70% abstention (35/50) and 0.482 faithfulness, both produced by
  `build_eval_dataset.py:110-111` handing DeepSeek link fragments and duplicated nav
  headers as "the supplied facts."

Do not carry either number forward. And note the interaction: abstention is 43.8%
(119/272 non-answerable-ish in the covered collections) *in principle* — the 70% figure
measures a broken harness, not the system.

### 11.2 Deliverables

1. `docs/corpus-spec.md` — this document.
2. `docs/corpus-report.md` — measured before/after per layer, with §3.5 cleaning provenance
   and the §11 phenomenon coverage, plus the §2.3 resolution limits stated up front.
3. A limitations section, in this order: controller validated against simulated partials not
   live ASR; stability labels computed by LocalAgreement-n and human-labelled on a sample
   only; hyperparameters corpus-specific; test-time distribution shift expected; retrieval
   recall bounds grounding.

**Traceability assertion for the brief:** every fixture carries
`provenance.source ∈ {mtragun, augmented}` and `provenance.recipe_step`. State the resulting
ratio and confirm it lands near 80/20 per `dataset-and-augmentation.md` §4. That sentence is
the difference between "we built a test set" and "we built a test set on a published
benchmark using a published methodology."
