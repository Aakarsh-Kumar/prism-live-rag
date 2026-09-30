# LLM Provider Selection — Streaming Live RAG

> Research-era provider comparison; quoted prices, quotas and recommendations
> are dated findings, not verified current service terms. Current dashboard
> routing uses server-side Cerebras GPT-OSS-120B at five requests/minute and may
> make no call for a query. Final answers are not token-streamed. See the README
> and benchmark report for implemented behavior rather than the proposals below.

> ✅ **Standing rule:** latency-critical live-path stages use the fastest acceptable
> streaming provider. Quality-heavy non-streaming stages use the strongest affordable
> model. For final grounded generation, prefer the best fast streaming model that
> meets the grounding bar rather than the globally strongest model.

Budget: up to $5 API spend, or free/open-source — cloud-hosted only, no local hosting.
Compiled by researching what real hackathon projects and the closest comparable
research competition (SemEval-2026 Task 8) actually used, plus provider-specific
details that don't show up in generic "best free LLM API" roundups.

---

## 0. The single fact that reframes this decision

**Groq and NVIDIA announced a non-exclusive inference-technology licensing agreement
in December 2025, not an acquisition.** Groq said it remains independent and that
GroqCloud continues to operate; NVIDIA separately uses Groq technology in its own
inference roadmap under the LPX branding. Practical implications for you:
- Treat Groq as an independent provider with a strategically important NVIDIA
  relationship, not as an NVIDIA-owned service.
- The latency recommendation still stands: Groq is useful here because of fast
  TTFT and streaming behavior, not because of ownership assumptions.
- Cerebras remains a credible throughput-focused alternative for heavier,
  less latency-critical work. Neither provider choice should be justified by a
  supposed acquisition.

---

## 1. What "fast" actually means here — two different metrics, pick based on your bottleneck

This distinction is usually glossed over in generic comparisons, but it matters
directly for your architecture:

- **Time-to-first-token (TTFT)** — how quickly the *first* word appears. This is
  what your **Retrieval Controller** and **streaming answer delivery** care about,
  since the whole premise of Gate G2 is starting to respond before the user
  finishes speaking.
- **Throughput (tokens/sec after the first token)** — how fast the *rest* of the
  response streams out, or how fast you can process many requests. Matters more for
  bulk offline tasks (corpus preprocessing, batch evaluation) than for the live
  conversational path.

**Groq optimizes for TTFT** (sub-100ms, independently benchmarked) — its LPU uses
deterministic SRAM-based memory specifically to eliminate the latency variance GPUs
have. One cited case: switching a customer support chatbot from H100 inference to
Groq produced a 34% increase in user satisfaction, attributed entirely to faster
response onset — nothing about answer quality changed.

**Cerebras optimizes for raw throughput** (2,000–2,900 tok/s on Llama-class models,
independently verified as up to 19× faster than the fastest GPU cloud) via its
wafer-scale chip, which stores entire models on-chip.

**Implication for your build:** use Groq specifically for the Retrieval Controller
and the final answer stream (both are TTFT-bound by the streaming/full-duplex
constraint). Cerebras is the better fit for a component doing heavier, less
latency-critical work at higher volume — but for a hackathon-scale corpus, Groq's
free tier alone likely covers your whole pipeline without needing Cerebras at all.

---

## 2. Provider comparison

| Provider | Cost | Speed | Models available | Notes |
|---|---|---|---|---|
| **Groq** | Free, no card | TTFT < 100ms, ~600–1000 tok/s | Llama 3.1/3.3, GPT-OSS-20B/120B, Qwen3-32B, GLM-4.6 | 30 req/min; per-model daily token ceilings (500K/day Llama-3.1-8B, 100K/day Llama-3.3-70B, 200K/day GPT-OSS). Open-weight models only. |
| **Cerebras** | Free tier (~1M tok/day per some sources, context-limited per others) + $5 credit (30-day expiry) unlocks full access | 2,000–2,900 tok/s throughput | Llama 3.3-70B, Qwen3-235B, GLM-4.6, Qwen3-32B, Qwen3-Coder-480B | $5 credit maps exactly to your stated budget. Best for throughput-heavy, less latency-critical stages. |
| **DeepSeek (direct API)** | ~$0.23–0.28/MTok input, $0.34–0.42/MTok output | Slow: ~30–36 tok/s, ~1.25s TTFT | DeepSeek-V3.2 | Cheap enough that $5 buys 15–20M tokens. **This is what the actual top-ranked SemEval-2026 Task 8 teams used** for query rewriting, span extraction, and judging — validated for RAG quality, not just cost. Wrong choice for anything latency-critical. |
| **Google Gemini** | Free tier | Moderate | Gemini 2.5 Flash-Lite (10 RPM), Gemini 2.5 Flash (5 RPM), Gemini 3 Flash Preview (5 RPM) | Gemini 2.5 models shut down **Oct 16, 2026** — close enough to a hackathon timeline to matter; prefer Gemini 3 Flash Preview or 3.1 Flash-Lite (stable) if you use this at all. Free-tier prompts are used to improve Google's products per their terms. |

---

## 3. Niche finds — providers that won't show up in a first-pass search

These came from cross-referencing multiple maintained "awesome free LLM API" GitHub
lists rather than a single source, and from reading past the headline pricing page
into rate-limit fine print:

- **LLM7.io** — no signup required at all: 30 RPM anonymous, up to 120 RPM with a
  free email token, and a genuinely large rolling daily allowance (up to 5M
  tokens/day with the free token). Hosts DeepSeek-R1 and Qwen 2.5. Essentially
  unknown outside these curated lists — worth having as a zero-friction fallback
  since it requires no account setup during a time-pressured hackathon.
- **Cloudflare Workers AI** — free tier measured in "neurons" (~10,000/day, roughly
  300,000/month), hosts Llama 3.1/3.2, Mistral 7B, Qwen 1.5 7B. Useful if you're
  already deploying on Cloudflare's edge network for other reasons — inference
  co-located with your app reduces network latency on top of the model's own TTFT,
  which compounds with the Groq recommendation above rather than competing with it.
- **Z.AI (GLM)** — GLM-4.5-Flash / GLM-4.7-Flash at roughly 1 request/second,
  ~1,000 requests/day free. GLM models scored competitively in the RaguTeam
  ensemble (one of your existing reference papers) — a legitimate quality option
  most people researching "free LLM APIs" generically won't think to look for by
  name.
- **`freeflow-llm`** (PyPI package) — chains Groq → Gemini → GitHub Models with
  automatic fallback on HTTP 429, prioritizing the fastest provider first. This is
  exactly the resilience pattern worth building (or borrowing) so a single
  provider's rate limit doesn't take down your demo mid-presentation.
- **GitHub Models** — free access gated by your GitHub Copilot subscription tier,
  hosting a rotating set of models including some frontier ones. Easy to overlook
  since it's bundled into a developer tool rather than marketed as an LLM API.
- **1M free tokens per model from Qwen (Alibaba DashScope)** on signup, 90-day
  expiry, no card — a clean, no-strings option for Qwen3-Max/Plus specifically if
  you want a non-Groq-hosted route to Qwen's larger variants.

---

## 4. Recommended routing (ties directly to the model-routing pattern already
validated in your knowledge base)

This applies the validated cost/quality routing pattern to a streaming system:
latency-critical hot-path stages use the fastest acceptable provider, while slower
quality-focused calls are reserved for stages that do not block response onset.

| Pipeline stage | Provider | Rationale |
|---|---|---|
| Retrieval Controller (Wait/Retrieve/Suppress decision) | **Groq**, free | TTFT-bound; this decision has to happen before the utterance ends. |
| Multi-Intent Decomposer | **Groq**, free | Same latency class — sub-queries need to dispatch fast for parallel retrieval to pay off. |
| Query rewriting / evidence span extraction | **DeepSeek API**, paid (~$5 budget) | Validated by the actual top SemEval-2026 teams for exactly this role; not latency-critical, so DeepSeek's slower TTFT doesn't hurt here. |
| Final grounded generation (streamed to user) | **Groq**, free | Still under the full-duplex latency constraint — keep it on the fast free tier, not DeepSeek. Current code defaults to `openai/gpt-oss-120b`; override with `GROQ_MODEL` if account-tier availability changes. |
| Offline G4 faithfulness judging | **Cerebras GPT-OSS-120B** | Uses `CEREBRAS_API_KEY`; the evaluator spaces calls at 12.5 seconds and checks cited passages only. This does not affect the live response path. |
| Fallback when a free tier's daily cap is hit | **Gemini free tier** or **LLM7.io** | Different provider/infrastructure entirely, so one provider's outage or rate limit doesn't take down your whole demo. Consider wiring in `freeflow-llm`'s fallback pattern rather than building your own from scratch. |

**Total expected cost: comfortably under your $5 ceiling** — the entire hot path
runs on free tiers, and the $5 budget is spent only where a slower but
competition-validated model (DeepSeek) earns its cost on quality rather than speed.

---

## 5. One open risk worth flagging

Free-tier rate limits (Groq's 30 req/min, Gemini's 5–10 RPM) are per-organization,
not per-key — creating a second API key doesn't raise your ceiling. If your team
tests heavily in parallel close to submission time, you can hit these limits faster
than expected. Build the fallback-provider pattern in section 3 early, not as a
last-minute patch, and treat it as part of your Gate G1 (reproducibility) robustness
rather than an afterthought.
