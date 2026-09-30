(() => {
  const $ = (id) => document.getElementById(id);
  const state = { scenarios: [], selected: null, runId: null, source: null, events: [], passageMap: new Map(), revisions: new Set(), llmCalls: 0, providerModel: null };
  const els = {
    filter: $('scenario-filter'), scope: $('scenario-set'), scenario: $('scenario'), scenarioCount: $('scenario-count'), scenarioMeta: $('scenario-meta'),
    speed: $('speed'), mode: $('mode'), multi: $('multi-intent'), llmDecomposer: $('llm-decomposer'), llmWrap: $('llm-decomposer-wrap'),
    run: $('run'), replay: $('replay'), indicator: $('run-indicator'), status: $('run-status'), runId: $('run-id'),
    transcript: $('transcript'), transcriptEmpty: $('transcript-empty'), chunkCounter: $('chunk-counter'), timeline: $('timeline'),
    timelineEmpty: $('timeline-empty'), timelineCount: $('timeline-count'), answer: $('answer'), answerState: $('answer-state'),
    answerNotice: $('answer-notice'), citations: $('citations'), citationCount: $('citation-count'), claimDetails: $('claim-details'),
    claimMap: $('claim-map'), raw: $('raw-log'), retrieval: $('metric-retrieval'), generation: $('metric-generation'),
    total: $('metric-total'), llm: $('metric-llm'), model: $('metric-model'), tokens: $('metric-tokens'), trace: $('trace-id'), cost: $('cost'),
    validation: $('validation'), providerTop: $('provider-top'),
  };

  function safeText(value, fallback = '') { return typeof value === 'string' && value ? value : fallback; }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function setStatus(label, kind = 'idle') {
    els.status.textContent = label;
    els.indicator.className = `status-dot ${kind}`;
  }
  function prettyDomain(value) { return value === 'govt' ? 'Government' : value === 'cloud' ? 'Cloud' : value || 'Unknown domain'; }
  function shortQuery(row) {
    const query = safeText(row.description, row.expected_behavior || row.id).replace(/\s+/g, ' ').trim();
    return query.length > 96 ? `${query.slice(0, 93)}…` : query;
  }

  // The filter changes selection only: labels never influence answer routing.
  const categoryFilter = document.createElement('select');
  categoryFilter.setAttribute('aria-label', 'Filter scenario category');
  for (const [value, label] of [['', 'All query types'], ['answerable', 'Answerable'], ['partial', 'Partially answerable'], ['unanswerable', 'Unanswerable'], ['underspecified', 'Needs clarification'], ['multi_intent', 'Multi-intent · rule preview'], ['early_retrieval', 'Early retrieval']]) {
    const option = document.createElement('option'); option.value = value; option.textContent = label; categoryFilter.append(option);
  }
  els.filter.before(categoryFilter);
  categoryFilter.addEventListener('change', renderScenarioOptions);

  const followupPanel = document.createElement('section'); followupPanel.className = 'panel controls';
  const followupInput = document.createElement('input'); followupInput.placeholder = 'Add a late constraint, or ask to reformat the answer';
  followupInput.setAttribute('aria-label', 'Follow-up in the current session'); followupInput.maxLength = 2000;
  followupInput.style.cssText = 'flex:1;min-width:200px;padding:12px;border:1px solid #e1e8e3;border-radius:8px';
  const followupKind = document.createElement('select'); followupKind.setAttribute('aria-label', 'Follow-up type');
  for (const [value, label] of [['refinement', 'Late constraint · delta retrieval'], ['presentation', 'Presentation only · reuse answer']]) {
    const option = document.createElement('option'); option.value = value; option.textContent = label; followupKind.append(option);
  }
  const followupButton = document.createElement('button'); followupButton.className = 'button secondary';
  followupButton.textContent = 'Continue session'; followupButton.disabled = true;
  followupPanel.append(followupInput, followupKind, followupButton);
  document.querySelector('.statusline').after(followupPanel);
  followupButton.addEventListener('click', () => {
    if (!followupInput.value.trim() || !state.runId) return;
    startRun({ parent_run_id: state.runId, follow_up: followupInput.value.trim(), follow_up_kind: followupKind.value });
  });

  async function loadScenarios() {
    try {
      const response = await fetch('/api/scenarios', { cache: 'no-store' });
      if (!response.ok) throw new Error(`Scenario request failed (${response.status})`);
      const payload = await response.json();
      state.scenarios = payload.scenarios || [];
      state.providerModel = payload.provider_model || null;
      const retrievalNote = $('retrieval-note');
      if (retrievalNote) {
        const device = payload.dense_enabled && payload.dense_device ? ` on ${payload.dense_device}` : '';
        retrievalNote.textContent = `${payload.retrieval_mode || 'Retrieval mode unavailable'}${device}; ${Number(payload.corpus_passages_loaded || 0).toLocaleString()} passages loaded. Reranking status is shown per retrieval.`;
      }
      els.scenarioCount.textContent = `· ${state.scenarios.filter((row) => row.is_evaluated_test).length} evaluated queries`;
      els.providerTop.textContent = payload.provider_available ? `${payload.provider_model} available` : 'offline-ready · no provider key';
      els.mode.querySelector('option[value="provider"]').disabled = !payload.provider_available;
      els.mode.querySelector('option[value="provider"]').textContent = payload.provider_available
        ? `Automatic · ${payload.provider_model}` : 'Automatic · unavailable (no server key)';
      // Enable selective query routing when a provider is configured. The
      // server reports each call/skip decision and enforces shared pacing.
      if (payload.provider_available) els.mode.value = 'provider';
      els.llmDecomposer.disabled = !els.multi.checked || els.mode.value !== 'provider';
      els.llmWrap.classList.toggle('disabled', els.llmDecomposer.disabled);
      // renderScenarioOptions selects the first row inside the active scope
      // (the judge-facing 200-query set). Do not overwrite it with the first
      // scenario from the unfiltered API payload below.
      renderScenarioOptions();
    } catch (error) {
      els.scenario.innerHTML = '<option value="">Could not load scenarios</option>';
      els.scenarioMeta.textContent = error.message;
      setStatus('Dashboard could not load scenario data', 'bad');
    }
  }

  function renderScenarioOptions() {
    const query = els.filter.value.trim().toLowerCase();
    const inScope = state.scenarios.filter((row) => els.scope.value === 'all' || row.is_evaluated_test);
    const rows = inScope.filter((row) => (!categoryFilter.value || row.case_class === categoryFilter.value || row.category === categoryFilter.value || (categoryFilter.value === 'multi_intent' && row.predicted_intent_count > 1)) && `${row.id} ${row.domain} ${row.category} ${row.case_class} ${row.expected_behavior} ${row.description} ${row.source}`.toLowerCase().includes(query));
    clear(els.scenario);
    for (const row of rows) {
      const option = document.createElement('option');
      option.value = row.id;
      const label = row.case_class || row.category;
      const set = row.is_evaluated_test ? 'TEST' : 'DEMO';
      option.textContent = `${set} · ${prettyDomain(row.domain)} · ${label} · ${shortQuery(row)}`;
      els.scenario.append(option);
    }
    if (!rows.length) {
      const option = document.createElement('option'); option.value = ''; option.textContent = 'No matching examples'; els.scenario.append(option);
    } else if (!rows.some((row) => row.id === state.selected)) {
      state.selected = rows[0].id;
      els.scenario.value = state.selected;
    } else {
      els.scenario.value = state.selected;
    }
    updateScenarioMeta();
  }

  function updateScenarioMeta() {
    const row = state.scenarios.find((item) => item.id === els.scenario.value);
    state.selected = row?.id || null;
    if (!row) { els.scenarioMeta.textContent = 'Try a different filter.'; return; }
    const revisionText = row.has_revision ? ' · includes ASR revisions' : '';
    const expectedText = row.expected_behavior ? ` · expected: ${row.expected_behavior} (${row.case_class})` : '';
    const sourceText = row.is_evaluated_test ? 'evaluated 200-query test set' : row.source;
    els.scenarioMeta.textContent = `${row.chunk_count} streamed chunks · ${prettyDomain(row.domain)} · ${row.category}${expectedText}${revisionText} · ${sourceText}`;
    if (row.category === 'multi_intent') els.multi.checked = true;
  }

  function resetView() {
    document.getElementById('conversation-context')?.remove();
    document.getElementById('benchmark-diagnostics')?.remove();
    state.events = []; state.passageMap.clear(); state.revisions.clear(); state.llmCalls = 0;
    clear(els.transcript); clear(els.timeline); clear(els.citations); clear(els.claimMap);
    els.transcriptEmpty.classList.remove('hidden'); els.timelineEmpty.classList.remove('hidden');
    els.chunkCounter.textContent = '0 chunks'; els.timelineCount.textContent = '0 events';
    els.answer.innerHTML = '<span class="placeholder">The final answer appears after the stream completes.</span>';
    els.answerState.textContent = 'Waiting'; els.answerState.className = 'badge neutral';
    els.answerNotice.classList.add('hidden'); els.citationCount.textContent = '0';
    els.claimDetails.classList.add('hidden'); els.retrieval.textContent = '—'; els.generation.textContent = '—'; els.total.textContent = '—';
    els.llm.textContent = '0'; els.model.textContent = 'No calls observed'; els.tokens.textContent = 'Unavailable';
    els.trace.textContent = 'Not assigned yet'; els.cost.textContent = 'Unavailable'; els.validation.textContent = 'Not run';
    els.raw.textContent = 'Events will be recorded here.';
  }

  async function startRun(followup = null) {
    if (!state.selected) return;
    if (state.source) state.source.close();
    resetView();
    els.run.disabled = true; els.replay.disabled = true; followupButton.disabled = true;
    setStatus('Starting selected scenario…', 'busy');
    try {
      const response = await fetch('/api/runs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_id: state.selected, ...(followup?.parent_run_id ? followup : {}), options: {
          mode: els.mode.value, speed: Number(els.speed.value), multi_intent: els.multi.checked,
          llm_decomposer: els.llmDecomposer.checked,
        } }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `Could not start run (${response.status})`);
      state.runId = payload.run_id;
      els.runId.textContent = `run ${state.runId.slice(0, 12)}`;
      connectRun(state.runId);
    } catch (error) {
      setStatus(error.message, 'bad'); els.run.disabled = false;
      followupButton.disabled = !state.runId;
    }
  }

  function connectRun(runId) {
    const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
    state.source = source;
    source.onmessage = (message) => {
      let event;
      try { event = JSON.parse(message.data); } catch { return; }
      if (state.events.some((known) => known.id === event.id)) return;
      state.events.push(event);
      appendRaw(event);
      handleEvent(event);
    };
    source.onerror = () => {
      if (source.readyState === EventSource.CLOSED) {
        fetch(`/api/runs/${encodeURIComponent(runId)}`).then((r) => r.json()).then((snapshot) => {
          for (const event of snapshot.events || []) if (!state.events.some((e) => e.id === event.id)) {
            state.events.push(event); appendRaw(event); handleEvent(event);
          }
          if (snapshot.status === 'complete' || snapshot.status === 'error') finishRun(snapshot.status);
        }).catch(() => setStatus('Connection ended; run snapshot unavailable', 'bad'));
      } else setStatus('Reconnecting to event stream…', 'busy');
    };
  }

  function appendRaw(event) {
    const line = JSON.stringify(event);
    if (els.raw.textContent === 'Events will be recorded here.') els.raw.textContent = '';
    els.raw.textContent += `${line}\n`;
    if (els.raw.parentElement.open) els.raw.scrollTop = els.raw.scrollHeight;
  }

  function handleEvent(event) {
    switch (event.type) {
      case 'conversation_context': {
        document.getElementById('conversation-context')?.remove();
        const details = document.createElement('details');
        details.id = 'conversation-context';
        const summary = document.createElement('summary');
        const turns = event.turns || [];
        summary.textContent = `Earlier conversation · ${turns.length} turns`;
        details.append(summary);
        if (!turns.length) {
          const note = document.createElement('p'); note.textContent = 'No earlier context: this is a standalone question.'; details.append(note);
        }
        for (const turn of turns) {
          const p = document.createElement('p'); p.textContent = `${turn.speaker}: ${turn.text}`; details.append(p);
        }
        els.transcript.parentElement.before(details);
        break;
      }
      case 'run': setStatus(event.status === 'running' ? 'Stream is running' : `Run ${event.status}`, event.status === 'running' ? 'busy' : 'idle'); break;
      case 'chunk': addChunk(event); addTimeline('ASR chunk received', event.is_final ? 'final hypothesis' : 'partial hypothesis', event.text || '', event.timestamp_s); break;
      case 'decision': addTimeline('controller', event.decision || 'Decision', `${event.reason || 'Reason unavailable'} · “${event.text || ''}”`, event.timestamp_s, event.decision === 'Retrieve' ? 'retrieve' : ''); break;
      case 'retrieval_start': addTimeline('retrieval pipeline started', event.requested_trigger ? `candidate trigger: ${event.requested_trigger}` : '', event.query || '', event.timestamp_s, 'retrieve'); break;
      case 'search_dispatch': addTimeline('corpus search dispatched', event.trigger || 'retrieval', `${(event.queries || []).join(' | ')} · observed ${event.observed_elapsed_s ?? 'unavailable'}s after run creation`, event.timestamp_s, 'retrieve'); break;
      case 'retrieval_done': addRetrieval(event); break;
      case 'llm_routing':
        addTimeline('LLM router', `${event.stage || 'answer'} · ${event.decision || 'skip'}`, `${event.reason || 'Reason unavailable'} · “${event.query || ''}”`, event.timestamp_s, event.eligible ? 'llm' : '');
        break;
      case 'llm_call':
        if (event.status === 'started') state.llmCalls += 1;
        els.llm.textContent = String(state.llmCalls);
        els.model.textContent = event.model || 'Model name unavailable';
        addTimeline('provider request', `${event.status} · ${event.model || 'model unavailable'}`, event.error || 'Observed at the actual provider client call boundary.', null, event.status === 'failed' ? 'fail' : 'llm');
        break;
      case 'answer_ready': paintAnswer(event, false); break;
      case 'chunk_processed': break;
      case 'answer': paintAnswer(event, true); paintMetrics(event); break;
      case 'error': setStatus(event.message || 'Run failed', 'bad'); els.answerState.textContent = 'Error'; els.answerState.className = 'badge warning'; break;
      case 'done': finishRun(event.status); break;
      default: addTimeline(event.type || 'event', 'Additional trace event', JSON.stringify(event), null, '');
    }
    els.timelineCount.textContent = `${Math.max(0, els.timeline.children.length)} events`;
  }

  function addChunk(event) {
    els.transcriptEmpty.classList.add('hidden');
    const li = document.createElement('li');
    const revised = state.selected && state.scenarios.find((x) => x.id === state.selected)?.revision_log?.some((r) => r.chunk_index === Number(els.transcript.children.length));
    const final = Boolean(event.is_final);
    li.className = `chunk${final ? ' final' : ''}${revised ? ' revised' : ''}`;
    const head = document.createElement('div'); head.className = 'chunk-head';
    const time = document.createElement('span'); time.className = 'chunk-time'; time.textContent = `${Number(event.timestamp_s || 0).toFixed(2)}s`;
    const tag = document.createElement('span'); tag.className = 'chunk-tag'; tag.textContent = final ? 'final · supersedes partials' : revised ? 'revised partial' : 'partial hypothesis';
    head.append(time, tag);
    const text = document.createElement('div'); text.className = 'chunk-text'; text.textContent = event.text || '';
    li.append(head, text); els.transcript.append(li);
    els.chunkCounter.textContent = `${els.transcript.children.length} chunks`;
    els.transcript.parentElement.scrollTop = els.transcript.parentElement.scrollHeight;
  }

  function addTimeline(title, detail, extra, timestamp, kind = '') {
    els.timelineEmpty.classList.add('hidden');
    const item = document.createElement('li'); item.className = `timeline-item ${kind}`;
    const head = document.createElement('div'); head.className = 'event-head';
    const label = document.createElement('span'); label.className = 'event-kind'; label.textContent = title;
    const time = document.createElement('span'); time.className = 'event-time'; time.textContent = timestamp == null ? new Date().toLocaleTimeString() : `${Number(timestamp).toFixed(2)}s`;
    head.append(label, time);
    const body = document.createElement('div'); body.className = 'event-detail';
    const strong = document.createElement('strong'); strong.textContent = detail;
    body.append(strong);
    if (extra) { body.append(document.createTextNode(` — ${extra}`)); }
    item.append(head, body); els.timeline.append(item);
    els.timeline.scrollTop = els.timeline.scrollHeight;
  }

  function addRetrieval(event) {
    const passages = event.passages || [];
    for (const passage of passages) state.passageMap.set(passage.id, passage);
    const subs = (event.sub_queries || []).join(' | ');
    const mode = event.retrieval_mode || 'retrieval mode unavailable';
    const methods = event.retrieval_method_counts || {};
    const flags = `dense ${event.dense_enabled ? `${methods.dense_results ?? 'unknown'} candidates` : 'off'} · sparse ${methods.sparse_results ?? 'unknown'} candidates · reranker ${event.reranker_status || (event.reranker_enabled ? 'enabled' : 'disabled')}`;
    addTimeline('retrieval complete', `${passages.length} passages · ${event.trigger || 'retrieval'} · ${mode}`, `query: ${event.query || 'unavailable'}${subs ? ` · sub-queries: ${subs}` : ''} · ${flags}`, event.timestamp_s, 'retrieve');
    if (!passages.length) return;
    const item = els.timeline.lastElementChild;
    const details = document.createElement('details'); details.className = 'passage-mini';
    const summary = document.createElement('summary'); summary.textContent = `Inspect ${passages.length} returned passages and rank signals`;
    details.append(summary);
    for (const p of passages) {
      const row = document.createElement('p');
      row.textContent = `${p.id} · ${p.title || 'Untitled source'} · fusion score ${Number(p.score).toFixed(5)} · dense rank ${p.dense_rank ?? 'n/a'} · sparse rank ${p.sparse_rank ?? 'n/a'}${p.url ? ` · ${p.url}` : ''}`;
      details.append(row);
    }
    item.append(details);
  }

  function paintAnswer(event, final) {
    const answer = safeText(event.answer);
    els.answer.textContent = answer || (event.uncertainty ? 'No supported answer was produced.' : 'No answer text returned.');
    const claims = Object.keys(event.claim_citations || {});
    const normalized = (text) => text.replace(/\s+/g, ' ').trim();
    // Layout-only: render exact mapped spans separately when they cover the
    // entire answer. Never rewrite facts or drop unmapped text.
    if (claims.length > 1 && normalized(claims.join(' ')) === normalized(answer)) {
      clear(els.answer);
      const list = document.createElement('ul'); list.className = 'answer-facts';
      for (const claim of claims) { const item = document.createElement('li'); item.textContent = claim; list.append(item); }
      els.answer.append(list);
    }
    els.answerState.textContent = final ? (event.uncertainty ? 'Abstained / qualified' : 'Answer returned · support not independently scored') : 'Answer ready';
    els.answerState.className = `badge ${event.uncertainty ? 'warning' : 'success'}`;
    const evaluation = event.evaluation;
    let evaluationText = '';
    if (final && evaluation?.expected_behavior) {
      let check;
      if (['answer', 'partial_answer'].includes(evaluation.expected_behavior)
          && typeof evaluation.qrel_citation_overlap_proxy === 'boolean') {
        check = evaluation.qrel_citation_overlap_proxy ? 'gold passage ID overlap' : 'no gold passage ID overlap';
      } else if (typeof evaluation.actual_behavior_match_proxy === 'boolean') {
        check = evaluation.actual_behavior_match_proxy
          ? `structural ${evaluation.expected_behavior} proxy matched`
          : `structural ${evaluation.expected_behavior} proxy not matched`;
      } else {
        check = 'behavior check unavailable';
      }
      evaluationText = `Benchmark proxy only: expected ${evaluation.expected_behavior} (${evaluation.case_class}); ${check}. This is not an answer-correctness or faithfulness score.`;
    }
    els.answerNotice.textContent = event.uncertainty || '';
    els.answerNotice.classList.toggle('hidden', !event.uncertainty);
    document.getElementById('benchmark-diagnostics')?.remove();
    if (evaluationText) {
      const details = document.createElement('details'); details.id = 'benchmark-diagnostics';
      const summary = document.createElement('summary'); summary.textContent = 'Benchmark diagnostics · not a correctness verdict';
      const note = document.createElement('p'); note.textContent = evaluationText;
      details.append(summary, note); els.answerNotice.after(details);
    }
    renderCitations(event.citations || []);
    renderClaimMap(event.claim_citations || {});
  }

  function renderCitations(ids) {
    clear(els.citations); els.citationCount.textContent = String(ids.length);
    if (!ids.length) { const empty = document.createElement('span'); empty.className = 'placeholder'; empty.textContent = 'No passage citations returned.'; els.citations.append(empty); return; }
    for (const id of ids) {
      const passage = state.passageMap.get(id);
      const card = document.createElement('article'); card.className = 'citation';
      const top = document.createElement('div'); top.className = 'citation-top';
      const code = document.createElement('code'); code.className = 'citation-id'; code.textContent = id;
      const title = document.createElement('span'); title.className = 'citation-title'; title.textContent = passage?.title || 'Corpus passage · title not supplied';
      top.append(code, title); card.append(top);
      const meta = document.createElement('div'); meta.className = 'citation-meta';
      meta.textContent = passage ? `${passage.domain || 'domain unavailable'} · retrieval score ${Number(passage.score).toFixed(5)} · dense ${passage.dense_rank ?? 'n/a'} / sparse ${passage.sparse_rank ?? 'n/a'}` : 'Passage payload unavailable for this citation';
      card.append(meta);
      if (passage?.url) {
        try {
          const sourceUrl = new URL(passage.url);
          if (sourceUrl.protocol === 'https:' || sourceUrl.protocol === 'http:') {
            const link = document.createElement('a'); link.href = sourceUrl.href; link.target = '_blank'; link.rel = 'noopener noreferrer'; link.textContent = 'Open source ↗'; link.className = 'citation-meta'; card.append(link);
          }
        } catch { /* malformed source URL stays visible as metadata in the raw trace only */ }
      }
      if (passage?.text) { const content = document.createElement('details'); content.className = 'citation-text'; const summary = document.createElement('summary'); summary.textContent = 'Show retrieved passage text'; const p = document.createElement('p'); p.textContent = passage.text; content.append(summary, p); card.append(content); }
      els.citations.append(card);
    }
  }

  function renderClaimMap(mapping) {
    clear(els.claimMap);
    const entries = Object.entries(mapping || {});
    els.claimDetails.classList.toggle('hidden', entries.length === 0);
    for (const [sentence, ids] of entries) {
      const row = document.createElement('div'); row.className = 'claim-row';
      const text = document.createElement('span'); text.textContent = sentence;
      const links = document.createElement('code'); links.textContent = ids.join(', ');
      row.append(text, links); els.claimMap.append(row);
    }
  }

  function paintMetrics(event) {
    const latency = event.stage_latency_ms || {};
    els.retrieval.textContent = Number.isFinite(latency.retrieval) ? `${latency.retrieval.toFixed(1)} ms` : 'Unavailable';
    els.generation.textContent = Number.isFinite(latency.generation) ? `${latency.generation.toFixed(1)} ms` : 'Unavailable';
    els.total.textContent = Number.isFinite(latency.total) ? `${latency.total.toFixed(1)} ms` : 'Unavailable';
    els.trace.textContent = event.trace_id || 'Unavailable';
    els.cost.textContent = event.estimated_cost_usd == null ? 'Unavailable' : `$${Number(event.estimated_cost_usd).toFixed(8)} · ${event.cost_estimate_basis || 'basis unavailable'}`;
    const usage = event.token_usage;
    els.tokens.textContent = usage && Number.isFinite(usage.total_tokens) ? `${usage.total_tokens} total` : 'Unavailable';
    const validation = event.trace_validation;
    els.validation.textContent = validation
      ? `${validation.valid ? 'valid' : 'invalid'} · ${validation.traces} persisted traces`
      : 'Trace validation unavailable';
  }

  function finishRun(status) {
    const failed = status === 'error';
    setStatus(failed ? 'Run finished with an error' : 'Scenario completed', failed ? 'bad' : 'good');
    els.run.disabled = false; els.replay.disabled = false;
    followupButton.disabled = failed;
    if (state.source) { state.source.close(); state.source = null; }
  }

  els.filter.addEventListener('input', renderScenarioOptions);
  els.scope.addEventListener('change', renderScenarioOptions);
  els.scenario.addEventListener('change', updateScenarioMeta);
  els.run.addEventListener('click', startRun);
  els.replay.addEventListener('click', startRun);
  els.multi.addEventListener('change', () => {
    els.llmDecomposer.disabled = !els.multi.checked || els.mode.value !== 'provider';
    els.llmWrap.classList.toggle('disabled', els.llmDecomposer.disabled);
  });
  els.mode.addEventListener('change', () => {
    els.llmDecomposer.disabled = !els.multi.checked || els.mode.value !== 'provider';
    els.llmWrap.classList.toggle('disabled', els.llmDecomposer.disabled);
  });
  loadScenarios();
})();
