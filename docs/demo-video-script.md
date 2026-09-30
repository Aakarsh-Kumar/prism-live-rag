# Demo video script (maximum five minutes)

Record only actual live runs on the final submission image. Show saved benchmark
results separately and label them historical. Do not conceal abstentions or use
saved answers as live output.

1. **0:00–0:30 — Launch.** Show the one-command Docker launch, readiness, and
   dashboard URL. Disclose the earlier first-time automatic asset download if
   recording from a warm cache. Never show the key or `.env` contents. State that
   input is simulated ASR chunks, not microphone audio.
2. **0:30–1:20 — Early retrieval.** Select a verified example; show chunks arriving,
   Wait becoming Retrieve before final, and the final answer with evidence.
3. **1:20–2:10 — Multi-intent.** Select a verified compound query; show distinct
   subqueries, retrieved sources, and which parts are answered or unsupported.
4. **2:10–3:10 — Session refinement.** Add a constraint in the same session. Show
   the targeted delta query, preserved evidence and answer-version lineage.
   Use the dashboard's Continue session control rather than a new independent run.
5. **3:10–3:40 — Suppression.** Ask to reformat the previous answer; show no new
   retrieval and retained citations.
6. **3:40–4:20 — Grounding and uncertainty.** Open a cited passage and show an
   insufficient-evidence/clarification case. Explain passage-level provenance.
7. **4:20–4:55 — Results and limitations.** Show the two matched ablation tables,
   trace validation and cost fields. Identify historical semantic scoring and
   current known failures. Finish with repo, image and report locations.

Prepare and verify scenario IDs before recording. Use accelerated transcript
pacing if necessary; disclose it. Provider pacing is five requests/minute, so
avoid depending on a quota-limited provider run during recording.
