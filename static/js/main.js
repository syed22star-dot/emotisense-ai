'use strict';

const EMOTION_META = {
  sadness: { emoji: '😢', color: '#2563eb' },
  anger: { emoji: '😠', color: '#dc2626' },
  love: { emoji: '❤️', color: '#db2777' },
  surprise: { emoji: '😲', color: '#0d9488' },
  fear: { emoji: '😨', color: '#7c3aed' },
  joy: { emoji: '😄', color: '#d97706' },
};

const CHIP_SAMPLES = {
  joy: 'I feel so incredibly happy and grateful for everything happening in my life right now!',
  sadness: 'I feel a deep crushing sadness in my chest, like the world lost its warmth.',
  anger: 'I am furious about how unfairly this was handled — it really frustrated me.',
  fear: 'I am terrified about what might happen next and cannot stop worrying.',
  love: 'I love spending quiet evenings with my family — it fills my heart with gratitude.',
  surprise: 'Oh wow, I was totally shocked and did not expect that announcement at all!',
  negation: 'I did not feel angry or upset at all, actually I felt relieved and grateful.',
};

const HISTORY_KEY = 'emotisense_history_v22';
let history = loadHistory();
let benchmarksData = null;
let batchCsvExport = '';
let lastAnalyzedText = '';
let agentHistory = [];
let hesitationTimer = null;
let nudgedEmotion = false;

let sessionChart = null;
let batchChart = null;
let benchFoldsChart = null;
let ablationChart = null;
let calibrationChart = null;
let emotionAblationMetric = 'both';
let emotionFoldsMode = 'top';

const $ = id => document.getElementById(id);

window.addEventListener('DOMContentLoaded', async () => {
  lucide.createIcons();
  initDraggableChatbot('agent-drawer', 'agent-drag-handle', 'agent-fab-btn', 'emoti_chat_pos');
  loadSavedAgentSettings();
  renderHistory();
  renderSessionChart();
  await Promise.all([checkHealth(), loadBenchmarks()]);

  setTimeout(() => {
    $('agent-drawer').classList.remove('closed');
    resetAgentGreeting();
  }, 1100);
});

/* ---------------- Proactive Concierge + 3-Second Hesitation ---------------- */
function armEmotionHesitation() {
  clearTimeout(hesitationTimer);
  if (nudgedEmotion || $('textInput').value.trim().length > 0) return;
  hesitationTimer = setTimeout(() => {
    if (nudgedEmotion || $('textInput').value.trim().length > 0) return;
    nudgedEmotion = true;
    $('agent-drawer').classList.remove('closed');
    appendCoachMessage(
      'Not sure what to type yet? Tell me how your day feels right here (or tap one of these) and I will load it into the Studio and run all 4 NLP models live on screen!',
      [],
      [
        "Analyze: 'I did not feel angry at all, I felt relieved and grateful!'",
        'Run 10-entry batch journal demo',
        'Open NLP Research Lab for joy',
      ]
    );
  }, 3200);
}

function resetAgentGreeting() {
  agentHistory = [];
  $('agent-messages').innerHTML = '';
  const hasKey = Boolean(sessionStorage.getItem('emotisense_agent_apikey')) || $('agent-key-badge').textContent.includes('LLM');
  const keyHint = hasKey ? '' : '\n\n💡 _Tip: Click the **🔑 Key** icon above to plug in your `GEMINI_API_KEY` for full open-ended NLP LLM coaching & custom rewrites._';
  appendCoachMessage(
    'Hi! 👋 I’m **Tone Helper**, your autonomous NLP site agent.\n\nShare any message, feeling, or journal lines with me — I can **automatically enter it into Emotion Studio, highlight the trigger words, compare all 4 models, or run Batch Journal screening live on screen!**' + keyHint,
    [],
    [
      "Analyze: 'I did not feel angry at all, I felt relieved and grateful!'",
      'Run 10-entry batch journal demo',
      'Open NLP Research Lab for anger',
    ]
  );
}

/* ---------------- Autonomous Website Automation Executor ---------------- */
async function executeUiCommands(commands) {
  if (!Array.isArray(commands) || !commands.length) return;
  const ctxPill = $('agent-context-pill').firstElementChild;
  const prev = ctxPill.textContent;
  ctxPill.textContent = '🤖 Agent operating website live...';

  for (const cmd of commands) {
    if (cmd.type === 'switch_tab' && cmd.tab) {
      switchTab(cmd.tab);
      await sleep(160);
    } else if (cmd.type === 'analyze_text' && cmd.text) {
      if (cmd.model) $('modelSelect').value = cmd.model;
      $('textInput').value = cmd.text;
      $('charCount').textContent = cmd.text.length;
      await runPredictionSubmit();
      await sleep(180);
    } else if (cmd.type === 'run_batch_lines' && Array.isArray(cmd.lines)) {
      $('batchTextArea').value = cmd.lines.join('\n');
      await runPastedBatch();
      await sleep(160);
    } else if (cmd.type === 'run_batch_demo') {
      await runDemoBatch();
      await sleep(160);
    } else if (cmd.type === 'inspect_vocab') {
      if (cmd.model) $('vocabModelSelect').value = cmd.model;
      if (cmd.emotion) $('vocabEmotionSelect').value = cmd.emotion;
      renderResearchLab();
      await sleep(150);
    } else if (cmd.type === 'set_text_and_analyze' && cmd.text) {
      if (cmd.model) $('modelSelect').value = cmd.model;
      $('textInput').value = cmd.text;
      $('charCount').textContent = cmd.text.length;
      await runPredictionSubmit();
      await sleep(160);
    } else if (cmd.type === 'set_batch_and_run' && Array.isArray(cmd.lines)) {
      $('batchTextArea').value = cmd.lines.join('\n');
      await runPastedBatch();
      await sleep(150);
    } else if (cmd.type === 'run_ab_arena') {
      await runLiveAbArena(cmd.text);
      await sleep(150);
    } else if (cmd.type === 'run_movie_vibe' && cmd.result) {
      switchPulseMode('movie');
      $('movieQueryInput').value = cmd.title || cmd.result.query || '';
      renderMovieVibeResults(cmd.result);
      await sleep(140);
    } else if (cmd.type === 'run_topic_pulse' && cmd.result) {
      switchPulseMode('topic');
      $('topicQueryInput').value = cmd.topic || cmd.result.query || '';
      renderTopicPulseResults(cmd.result);
      await sleep(140);
    } else if (cmd.type === 'run_tone_guard' && cmd.result) {
      if (cmd.text) { $('textInput').value = cmd.text; $('charCount').textContent = cmd.text.length; }
      renderToneGuardResults(cmd.result);
      await sleep(140);
    }
  }
  if (ctxPill.textContent.includes('operating website')) ctxPill.textContent = prev;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

/* ---------------- Health & Metadata ---------------- */
async function checkHealth() {
  const pill = $('header-health-pill');
  const badge = $('agent-key-badge');
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const serverOk = Boolean(data?.llm_agent?.configured);
    const hasSession = Boolean(sessionStorage.getItem('emotisense_agent_apikey'));
    pill.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span><span>${data.models_loaded?.length || 7} NLP Pipelines Ready (Word+Char+Syntax) · sklearn ${data.sklearn_version}</span>`;

    if (serverOk || hasSession) {
      badge.textContent = 'NLP LLM Active';
      badge.className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-800 font-bold';
    } else {
      badge.textContent = 'Autopilot (Add 🔑)';
      badge.className = 'text-[10px] px-2 py-0.5 rounded bg-amber-50 border border-amber-200 text-amber-800 font-bold';
    }
  } catch {
    pill.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500"></span><span>Backend Offline</span>`;
  }
}

async function loadBenchmarks() {
  try {
    const res = await fetch('/api/benchmarks');
    benchmarksData = await res.json();
    if (!benchmarksData?.benchmarks?.length) return;
    const champ = benchmarksData.benchmarks[0];
    if ($('kpiCleanRows')) $('kpiCleanRows').textContent = Number(benchmarksData.dataset?.retained_rows || 15939).toLocaleString();
    if ($('kpiChampTest')) $('kpiChampTest').textContent = `${(champ.test_accuracy * 100).toFixed(2)}%`;
    if ($('kpiChampCv')) $('kpiChampCv').textContent = `${(champ.cv_accuracy * 100).toFixed(2)}%`;
    if ($('resWinnerName')) $('resWinnerName').textContent = champ.model.replace('Hybrid Soft Ensemble (LR+SVC+CNB)', 'Hybrid Ensemble');
    if ($('resWinnerMetrics')) $('resWinnerMetrics').textContent = `${(champ.cv_accuracy * 100).toFixed(2)}% CV · ${(champ.test_accuracy * 100).toFixed(2)}% Test`;
    if ($('resMacroF1')) $('resMacroF1').textContent = `${champ.macro_f1.toFixed(4)} / ECE ${(champ.ece || 0).toFixed(4)}`;
    if ($('resVocabDim')) $('resVocabDim').textContent = `${Number(champ.vocab_size || 0).toLocaleString()} Feats`;
  } catch (err) { console.error(err); }
}

function switchTab(tab) {
  clearTimeout(hesitationTimer);
  ['studio', 'research', 'batch'].forEach(id => {
    $(`tab-${id}`).classList.toggle('hidden', id !== tab);
    $(`tab-btn-${id}`).classList.toggle('active', id === tab);
  });
  if (tab === 'research') setTimeout(renderResearchLab, 80);
}

$('textInput').addEventListener('input', () => { clearTimeout(hesitationTimer); $('charCount').textContent = $('textInput').value.length; $('inputError').textContent = ''; });
$('textInput').addEventListener('keydown', e => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); runPredictionSubmit(); } });

function loadChipSample(key) {
  $('textInput').value = CHIP_SAMPLES[key] || '';
  $('charCount').textContent = $('textInput').value.length;
  $('inputError').textContent = '';
  runPredictionSubmit();
}

function clearInput() { $('textInput').value = ''; $('charCount').textContent = '0'; $('inputError').textContent = ''; $('resultSection').classList.add('hidden'); }

$('predictForm').addEventListener('submit', e => { e.preventDefault(); runPredictionSubmit(); });

async function runPredictionSubmit() {
  const text = $('textInput').value.trim();
  if (!text) { $('inputError').textContent = 'Please type a sentence first.'; return; }
  const btn = $('analyzeBtn'); btn.disabled = true; $('analyzeBtnLabel').textContent = 'Analysing...';
  try {
    const res = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model: $('modelSelect').value, compare_models: $('compareToggle').checked }),
    });
    const data = await res.json();
    if (!res.ok) { $('inputError').textContent = data.error; return; }
    lastAnalyzedText = text;
    renderResult(data);
    pushHistory(data);
  } catch (err) { $('inputError').textContent = err.message; }
  finally { btn.disabled = false; $('analyzeBtnLabel').textContent = 'Analyse Emotion'; }
}

function renderResult(data) {
  const meta = EMOTION_META[data.emotion] || { emoji: '🤖', color: '#6d28d9' };
  $('resultSection').classList.remove('hidden');
  $('resultEmoji').textContent = meta.emoji;
  $('resultEmotion').textContent = data.emotion;
  $('resultEmotion').style.color = meta.color;
  $('resultModelBadge').textContent = `${data.model_name} · Test ${data.accuracy}`;
  $('resultRunnerUp').textContent = `Lead over ${data.runner_up.emotion} (${data.runner_up.confidence}%): +${data.margin_pct}%`;

  const ring = $('confRingFill'); ring.style.stroke = meta.color; ring.style.strokeDashoffset = String(97.4 * (1 - data.confidence / 100));
  $('confPct').textContent = `${Math.round(data.confidence)}%`;

  $('sentenceHeatmap').innerHTML = (data.sentence_tokens || []).map(tok => {
    const cls = tok.impact === 'positive' ? 'token-pos' : (tok.impact === 'negative' ? 'token-neg' : 'token-oov');
    return `<span class="token-pill ${cls}" title="Weight: ${tok.score}">${escapeHtml(tok.word)}</span>`;
  }).join('');

  $('probGrid').innerHTML = Object.entries(data.all_probabilities).sort((a, b) => b[1] - a[1]).map(([lbl, pct]) => {
    const m = EMOTION_META[lbl] || { emoji: '🤖', color: '#6d28d9' };
    return `<div class="flex items-center gap-3 text-xs"><div class="w-24 shrink-0 font-bold capitalize text-slate-700">${m.emoji} ${lbl}</div><div class="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden"><div class="prob-fill h-full rounded-full" style="width:${pct}%;background:${m.color}"></div></div><div class="w-12 text-right font-mono font-bold text-slate-700">${pct.toFixed(1)}%</div></div>`;
  }).join('');

  $('xaiPosChips').innerHTML = (data.token_importance || []).length ? data.token_importance.map(t => `<span class="token-pill token-pos">+${t.score.toFixed(2)} ${escapeHtml(t.word)}</span>`).join('') : '<span class="text-xs text-slate-400">None matched.</span>';
  $('xaiNegChips').innerHTML = (data.opposing_tokens || []).length ? data.opposing_tokens.map(t => `<span class="token-pill token-neg">${t.score.toFixed(2)} ${escapeHtml(t.word)}</span>`).join('') : '<span class="text-xs text-slate-400">None matched.</span>';

  const ab = data.ab_live || {};
  if ($('abControlVal') && ab.control && ab.variant) {
    $('abControlVal').textContent = `${ab.control.emoji} ${capitalise(ab.control.emotion)} (${ab.control.confidence.toFixed(1)}%)`;
    $('abVariantVal').textContent = `${ab.variant.emoji} ${capitalise(ab.variant.emotion)} (${ab.variant.confidence.toFixed(1)}%)`;
  }
  const syn = data.syntax_signals || {};
  if ($('syntaxSignalsBox')) {
    const chips = [];
    if (syn.negations_found?.length) chips.push(`<span class="px-2 py-0.5 rounded bg-white border border-purple-200 font-semibold text-violet-800">Negations: ${escapeHtml(syn.negations_found.join(', '))}</span>`);
    if (syn.pivot_cues?.length) chips.push(`<span class="px-2 py-0.5 rounded bg-white border border-purple-200 font-semibold text-amber-800">Pivots: ${escapeHtml(syn.pivot_cues.join(', '))}</span>`);
    if (syn.negated_negative_cues?.length) chips.push(`<span class="px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200 font-semibold text-emerald-800">Flip Guard: ${escapeHtml(syn.negated_negative_cues.join(', '))}</span>`);
    if (syn.intensifiers?.length) chips.push(`<span class="px-2 py-0.5 rounded bg-white border border-purple-200 text-pink-800">Intensifiers: ${escapeHtml(syn.intensifiers.join(', '))}</span>`);
    chips.push(`<span class="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-600 font-mono">TTR ${syn.lexical_diversity ?? '--'} · ${syn.word_count ?? 0} tok</span>`);
    $('syntaxSignalsBox').innerHTML = `<span class="font-mono text-violet-800 font-bold">Affective Syntax:</span> ${chips.join(' ')}`;
  }

  if (data.arena) {
    $('arenaSection').classList.remove('hidden');
    const a = data.arena;
    $('arenaConsensusBadge').textContent = a.unanimous ? `Unanimous (${a.consensus_count}/${a.total_models} ${capitalise(a.consensus_emotion)})` : `Majority (${a.consensus_count}/${a.total_models} ${capitalise(a.consensus_emotion)})`;
    $('arenaGrid').innerHTML = a.models.map(m => `<div class="soft-card p-3.5 rounded-xl flex items-center justify-between gap-2"><div><div class="text-xs font-bold text-slate-900">${escapeHtml(m.name)}</div><div class="text-[10px] font-mono text-slate-500">Test ${m.test_accuracy ?? '--'}% · ${escapeHtml(m.badge || '')}</div></div><div class="text-right"><div class="text-sm font-bold capitalize" style="color:${m.color}">${m.emoji} ${m.emotion}</div><div class="text-[11px] font-mono text-slate-600">${m.confidence.toFixed(1)}%</div></div></div>`).join('');
  } else {
    $('arenaSection').classList.add('hidden');
  }

  $('agent-context-pill').firstElementChild.textContent = `Live Sync: "${data.original_text.slice(0, 28)}..." -> ${capitalise(data.emotion)} (${data.confidence}%)`;
}

/* ---------------- History & Session Chart ---------------- */
function loadHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; } }
function pushHistory(data) { history.unshift({ emotion: data.emotion, confidence: data.confidence, text: data.original_text, ts: Date.now() }); if (history.length > 25) history.pop(); localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); renderHistory(); renderSessionChart(); }
function clearHistory() { history = []; localStorage.removeItem(HISTORY_KEY); renderHistory(); renderSessionChart(); }
function renderHistory() {
  $('sessionCountBadge').textContent = `${history.length} entries`;
  const list = $('historyList');
  if (!history.length) { list.innerHTML = '<li class="text-xs text-slate-400 text-center py-6">Run your first sentence on the left!</li>'; return; }
  list.innerHTML = history.map(h => {
    const m = EMOTION_META[h.emotion] || { emoji: '🤖', color: '#6d28d9' };
    return `<li onclick="reloadFromHistory(${JSON.stringify(h.text).replace(/"/g, '&quot;')})" class="soft-card p-2.5 rounded-xl flex items-center justify-between gap-2 cursor-pointer"><div class="flex items-center gap-2.5 min-w-0"><span class="text-lg">${m.emoji}</span><div class="min-w-0"><div class="text-xs font-bold capitalize" style="color:${m.color}">${h.emotion}</div><div class="text-[11px] text-slate-500 truncate">${escapeHtml(h.text)}</div></div></div><span class="text-[11px] font-mono font-bold text-slate-700 shrink-0">${h.confidence.toFixed(0)}%</span></li>`;
  }).join('');
}
function reloadFromHistory(txt) { $('textInput').value = txt; $('charCount').textContent = txt.length; runPredictionSubmit(); }
function renderSessionChart() {
  const counts = {}; Object.keys(EMOTION_META).forEach(k => { counts[k] = 0; }); history.forEach(h => { counts[h.emotion] = (counts[h.emotion] || 0) + 1; });
  if (sessionChart) sessionChart.destroy();
  sessionChart = new Chart($('sessionChart'), { type: 'doughnut', data: { labels: Object.keys(counts).map(capitalise), datasets: [{ data: Object.values(counts), backgroundColor: Object.keys(counts).map(k => EMOTION_META[k].color), borderWidth: 0 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { boxWidth: 10, font: { size: 11 } } } } } });
}

/* ---------------- NLP Research & A/B Lab ---------------- */
const EMOTION_STORY_TEXT = {
  1: '<strong>Why Step 1 Matters:</strong> Many beginner scripts strip stopwords (&ldquo;not&rdquo;, &ldquo;never&rdquo;, &ldquo;but&rdquo;), turning &ldquo;I am <strong>not</strong> happy&rdquo; into &ldquo;happy&rdquo;! Keeping helper words &amp; expanding apostrophes like <code>didn&rsquo;t</code> &rarr; <code>did not</code> preserves the true meaning.',
  2: '<strong>Why Step 2 Matters:</strong> Single-word Bag-of-Words fails on sarcasm &amp; mixed sentences. Our <code>FeatureUnion</code> blends word pairs (<code>did not feel</code>), character chunks (<code>happp</code> for typos/stretches), and 15 negation/pivot guard cues simultaneously.',
  3: '<strong>Why Step 3 Matters:</strong> Raw counts &amp; TF-IDF vectors sit on tiny decimals while word counts reach 3&ndash;4. Scaling with <strong>MaxAbsScaler [0..1]</strong> keeps sparsity intact and outperforms StandardScaler (`88.80%` CV vs `88.24%` CV).',
  4: '<strong>Why Step 4 Matters:</strong> Combining <strong>Logistic Regression</strong>, <strong>Calibrated LinearSVC</strong>, and <strong>ComplementNB</strong> cuts probability error (<code>Brier -0.0185</code>) and wins over the old BoW baseline in <strong>97%</strong> of bootstrap resamples.',
};

function selectEmotionStoryStep(step) {
  [1, 2, 3, 4].forEach(i => {
    const el = $(`estep-${i}`);
    if (!el) return;
    el.className = i === step
      ? 'soft-card p-3.5 rounded-xl text-left border-2 border-violet-600 bg-purple-50/40'
      : 'soft-card p-3.5 rounded-xl text-left hover:border-violet-300';
  });
  const det = $('emotion-story-detail');
  if (det) {
    det.innerHTML = `<div>${EMOTION_STORY_TEXT[step] || ''}</div><button type="button" onclick="askCoachTopic('Explain simply how VibeCheck understands negations like not angry and why Hybrid Ensemble beats simple Bag of Words')" class="shrink-0 px-3 py-1.5 rounded-lg bg-white border border-purple-300 text-violet-800 font-bold hover:bg-purple-50">Ask NLP Tutor</button>`;
  }
}

async function runEmotionLabDemo(kind) {
  const map = {
    negation: "I didn't feel angry at all, actually I felt relieved and grateful!",
    pivot: 'I was terrified before the surgery, but ended up thrilled and smiling afterward!',
    elongation: 'Wait what?! I am sooooo happpppy and stunned we actually won!',
  };
  const txt = map[kind] || map.negation;
  await runLiveAbArena(txt);
}

function setEmotionAblationMetric(mode) {
  emotionAblationMetric = mode;
  $('eabl-btn-both').className = mode === 'both' ? 'px-2.5 py-1 rounded bg-white text-violet-800 shadow-sm' : 'px-2.5 py-1 rounded text-slate-600';
  $('eabl-btn-f1').className = mode === 'f1' ? 'px-2.5 py-1 rounded bg-white text-violet-800 shadow-sm' : 'px-2.5 py-1 rounded text-slate-600';
  drawEmotionAblationChart();
}

function setEmotionFoldsFilter(mode) {
  emotionFoldsMode = mode;
  $('efold-btn-top').className = mode === 'top' ? 'px-2 py-1 rounded bg-white text-violet-800 shadow-sm' : 'px-2 py-1 rounded text-slate-600';
  $('efold-btn-all').className = mode === 'all' ? 'px-2 py-1 rounded bg-white text-violet-800 shadow-sm' : 'px-2 py-1 rounded text-slate-600';
  drawEmotionFoldsChart();
}

function renderAbStatsPanel() {
  const box = $('abStatsPanel');
  const pairKey = $('abPairSelect')?.value || 'bow_vs_hybrid_ensemble';
  const ab = benchmarksData?.ab_tests?.[pairKey];
  if (!box || !ab) return;
  const mc = ab.mcnemar || {};
  const bt = ab.bootstrap || {};
  const c = mc.contingency || {};
  box.innerHTML = `
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-center">
      <div class="soft-card p-3 rounded-xl"><div class="text-[10px] font-mono text-slate-500">Accuracy Gain (95% CI)</div><div class="text-sm font-extrabold ${bt.accuracy_delta_pct?.mean >= 0 ? 'text-emerald-700' : 'text-rose-700'} mt-0.5">${bt.accuracy_delta_pct?.mean >= 0 ? '+' : ''}${bt.accuracy_delta_pct?.mean}%</div><div class="text-[10px] font-mono text-slate-500">${bt.prob_variant_beats_control_acc}% Boot Win</div></div>
      <div class="soft-card p-3 rounded-xl"><div class="text-[10px] font-mono text-slate-500">Balanced Macro F1</div><div class="text-sm font-extrabold ${bt.macro_f1_delta_pct?.mean >= 0 ? 'text-violet-700' : 'text-rose-700'} mt-0.5">${bt.macro_f1_delta_pct?.mean >= 0 ? '+' : ''}${bt.macro_f1_delta_pct?.mean}%</div><div class="text-[10px] font-mono text-slate-500">${bt.prob_variant_beats_control_f1}% Boot Win</div></div>
      <div class="soft-card p-3 rounded-xl"><div class="text-[10px] font-mono text-slate-500">McNemar Test</div><div class="text-sm font-extrabold text-slate-900 mt-0.5">p = ${mc.p_value}</div><div class="text-[10px] font-mono text-slate-500">${mc.net_wins_for_variant >= 0 ? '+' : ''}${mc.net_wins_for_variant} net discordant</div></div>
      <div class="soft-card p-3 rounded-xl"><div class="text-[10px] font-mono text-slate-500">Honesty Error Drop</div><div class="text-sm font-extrabold text-emerald-700 mt-0.5">${bt.brier_delta?.mean}</div><div class="text-[10px] font-mono text-slate-500">BoW ${bt.brier_control?.mean} &rarr; Var ${bt.brier_variant?.mean}</div></div>
    </div>
    <div class="grid grid-cols-4 gap-2 text-center text-[11px] pt-1">
      <div class="p-2 rounded-lg bg-emerald-50 border border-emerald-200"><div class="font-mono font-bold text-emerald-800">${c.both_correct}</div><div class="text-slate-600">Both Got Right</div></div>
      <div class="p-2 rounded-lg bg-violet-50 border border-violet-200"><div class="font-mono font-bold text-violet-800">${c.variant_only_correct}</div><div class="text-slate-600">Hybrid AI Fixed</div></div>
      <div class="p-2 rounded-lg bg-amber-50 border border-amber-200"><div class="font-mono font-bold text-amber-800">${c.control_only_correct}</div><div class="text-slate-600">Old BoW Only</div></div>
      <div class="p-2 rounded-lg bg-slate-100 border border-slate-200"><div class="font-mono font-bold text-slate-700">${c.both_wrong}</div><div class="text-slate-600">Both Missed</div></div>
    </div>
  `;
}

async function runLiveAbArena(overrideText) {
  const box = $('liveAbArena');
  const xray = $('emotion-live-xray');
  if (!box) return;
  const txt = (overrideText || $('textInput').value || CHIP_SAMPLES.negation).trim();
  try {
    const res = await fetch('/api/ab-compare', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: txt, control: 'bow_lr', variant: 'hybrid_ensemble' }),
    });
    const d = await res.json();
    if (!res.ok) return;
    const cm = EMOTION_META[d.control.emotion] || { emoji: '🎭', color: '#334155' };
    const vm = EMOTION_META[d.variant.emotion] || { emoji: '🎭', color: '#6d28d9' };
    box.innerHTML = `
      <div class="p-2.5 rounded-xl bg-slate-50 border border-slate-200 text-[11px] text-slate-700 italic truncate">&ldquo;${escapeHtml(d.text)}&rdquo;</div>
      <div class="grid grid-cols-2 gap-2.5">
        <div class="soft-card p-3 rounded-xl"><div class="text-[10px] font-mono text-slate-500 uppercase">Old Control: BoW+LR</div><div class="text-lg font-extrabold capitalize mt-0.5" style="color:${cm.color}">${cm.emoji} ${d.control.emotion}</div><div class="text-[11px] font-mono text-slate-600">${d.control.confidence.toFixed(1)}%</div></div>
        <div class="soft-card p-3 rounded-xl bg-purple-50/70 border-purple-200"><div class="text-[10px] font-mono text-violet-800 uppercase">New Variant B: Hybrid</div><div class="text-lg font-extrabold capitalize mt-0.5" style="color:${vm.color}">${vm.emoji} ${d.variant.emotion}</div><div class="text-[11px] font-mono text-violet-800">${d.variant.confidence.toFixed(1)}%</div></div>
      </div>
      <div class="p-2.5 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-between text-[11px]">
        <span>Confidence Shift: <strong class="font-mono">${d.confidence_delta > 0 ? '+' : ''}${d.confidence_delta}%</strong></span>
        <span class="font-bold ${d.decision_flip ? 'text-amber-700' : 'text-emerald-700'}">${d.decision_flip ? 'Hybrid Fixed / Flipped Outcome!' : 'Both Models Agree'}</span>
      </div>
    `;
    if (xray) {
      const syn = d.syntax_signals || {};
      const cues = [
        syn.negations_found?.length ? `Neg: ${syn.negations_found.join(',')}` : '',
        syn.pivot_cues?.length ? `Pivot: ${syn.pivot_cues.join(',')}` : '',
        syn.intensifiers?.length ? `Boost: ${syn.intensifiers.join(',')}` : '',
      ].filter(Boolean).join(' · ') || 'Standard Lexical Clues';
      const topW = (d.variant.top_positive_words || []).slice(0, 3).map(w => w.word).join(', ') || 'Subwords';
      xray.innerHTML = `<div class="flex flex-wrap items-center gap-2 text-[11px]"><span class="font-extrabold text-violet-900">Live X-Ray:</span><span class="px-2 py-0.5 rounded bg-white border border-purple-200"><strong>1. Clean:</strong> &ldquo;${escapeHtml(d.variant.cleaned_text.slice(0, 52))}&rdquo;</span><span>&rarr;</span><span class="px-2 py-0.5 rounded bg-white border border-purple-200"><strong>2. Signals:</strong> ${escapeHtml(cues)} | Clues: ${escapeHtml(topW)}</span><span>&rarr;</span><span class="px-2 py-0.5 rounded bg-purple-100 text-violet-900 font-bold border border-purple-200"><strong>4. Team Vote:</strong> ${vm.emoji} ${capitalise(d.variant.emotion)} (${d.variant.confidence}%)</span></div>`;
    }
  } catch (err) { console.error(err); }
}

function drawEmotionAblationChart() {
  const ablation = benchmarksData?.ablation || [];
  if (!$('ablationChart') || !ablation.length) return;
  if (ablationChart) ablationChart.destroy();
  const datasets = emotionAblationMetric === 'f1'
    ? [
      { label: 'CV Macro F1 %', data: ablation.map(a => a.cv_macro_f1), backgroundColor: '#7c3aed', borderRadius: 4 },
      { label: 'Test Macro F1 %', data: ablation.map(a => a.test_macro_f1), backgroundColor: '#db2777', borderRadius: 4 },
    ]
    : [
      { label: '5-Fold CV Acc %', data: ablation.map(a => a.cv_accuracy), backgroundColor: '#7c3aed', borderRadius: 4 },
      { label: 'Held-Out Test %', data: ablation.map(a => a.test_accuracy), backgroundColor: '#059669', borderRadius: 4 },
    ];
  ablationChart = new Chart($('ablationChart'), {
    type: 'bar',
    data: { labels: ablation.map(a => a.stage.replace(/^\d+\.\s*/, '')), datasets },
    options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } }, scales: { x: { min: 70, max: 96 } } },
  });
}

function drawEmotionFoldsChart() {
  if (!benchmarksData?.benchmarks) return;
  const rows = emotionFoldsMode === 'top' ? benchmarksData.benchmarks.slice(0, 4) : benchmarksData.benchmarks;
  const colors = ['#6d28d9', '#db2777', '#0d9488', '#0284c7', '#d97706', '#9333ea', '#64748b'];
  if (benchFoldsChart) benchFoldsChart.destroy();
  benchFoldsChart = new Chart($('benchFoldsChart'), {
    type: 'line',
    data: { labels: ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'], datasets: rows.map((b, i) => ({ label: b.model.replace('Hybrid BoW+TFIDF+Char+Syntax + ', 'Hybrid '), data: (b.cv_fold_accuracies || []).map(v => +(v * 100).toFixed(2)), borderColor: colors[i % colors.length], backgroundColor: colors[i % colors.length], borderWidth: 2, tension: 0.25 })) },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } }, scales: { y: { min: emotionFoldsMode === 'top' ? 80 : 55, max: 96 } } },
  });
}

function renderResearchLab() {
  if (!benchmarksData?.benchmarks) return;
  renderAbStatsPanel();
  runLiveAbArena();
  drawEmotionFoldsChart();
  drawEmotionAblationChart();

  const rows = benchmarksData.benchmarks;
  const ablation = benchmarksData.ablation || [];
  if ($('ablationTbody') && ablation.length) {
    $('ablationTbody').innerHTML = ablation.map((r, i) => `
      <tr class="${i === ablation.length - 1 ? 'bg-purple-50/70 font-semibold' : ''}">
        <td class="p-2.5 text-slate-900">${escapeHtml(r.stage)}</td>
        <td class="p-2.5 font-mono text-slate-500">${Number(r.feature_dim).toLocaleString()}</td>
        <td class="p-2.5 font-mono font-bold text-violet-800">${r.cv_accuracy}%</td>
        <td class="p-2.5 font-mono">${r.cv_macro_f1}%</td>
        <td class="p-2.5 font-mono font-bold text-emerald-700">${r.test_accuracy}%</td>
        <td class="p-2.5 font-mono">${r.test_macro_f1}%</td>
      </tr>
    `).join('');
  }

  const ens = rows.find(r => r.key === 'hybrid_ensemble');
  const bow = rows.find(r => r.key === 'bow_lr');
  if ($('calibrationChart') && ens?.reliability_bins && bow?.reliability_bins) {
    if (calibrationChart) calibrationChart.destroy();
    calibrationChart = new Chart($('calibrationChart'), {
      type: 'line',
      data: {
        datasets: [
          { label: `Hybrid Ensemble (ECE ${ens.ece})`, data: ens.reliability_bins.map(b => ({ x: b.mean_conf, y: b.empirical_acc })), borderColor: '#6d28d9', backgroundColor: '#6d28d9', pointRadius: 4, tension: 0.2 },
          { label: `Old BoW+LR (ECE ${bow.ece})`, data: bow.reliability_bins.map(b => ({ x: b.mean_conf, y: b.empirical_acc })), borderColor: '#db2777', backgroundColor: '#db2777', pointRadius: 3, tension: 0.2 },
          { label: 'Ideal Honesty', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: '#cbd5e1', borderDash: [4, 4], pointRadius: 0 },
        ],
      },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } } }, scales: { x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'AI Confidence' } }, y: { min: 0, max: 1, title: { display: true, text: 'Actual Right Rate' } } } },
    });
  }

  const stress = benchmarksData.stress_suite;
  if ($('stressSuiteTbody') && stress?.cases) {
    $('stressPassBadge').textContent = `Old BoW ${stress.control_pass_count}/${stress.total} vs New Hybrid ${stress.variant_pass_count}/${stress.total}`;
    $('stressSuiteTbody').innerHTML = stress.cases.map(c => {
      const encTxt = encodeURIComponent(c.text);
      return `
      <tr onclick="runLiveAbArena(decodeURIComponent('${encTxt}'))" class="cursor-pointer hover:bg-purple-50/50 transition">
        <td class="p-2.5 font-bold text-slate-800">${escapeHtml(c.category)}</td>
        <td class="p-2.5 text-slate-600 max-w-xs truncate" title="${escapeHtml(c.text)}">${escapeHtml(c.text)}</td>
        <td class="p-2.5 font-bold capitalize">${c.expected}</td>
        <td class="p-2.5 font-mono ${c.control_pass ? 'text-emerald-700 font-bold' : 'text-rose-600 font-bold'}">${c.control_pass ? '✓' : '✗'} ${c.control_pred} (${c.control_conf}%)</td>
        <td class="p-2.5 font-mono ${c.variant_pass ? 'text-emerald-700 font-bold' : 'text-rose-600 font-bold'}">${c.variant_pass ? '✓' : '✗'} ${c.variant_pred} (${c.variant_conf}%)</td>
        <td class="p-2.5 text-right"><button type="button" onclick="event.stopPropagation();loadStressInStudio('${encTxt}')" class="px-2 py-1 rounded bg-violet-100 hover:bg-violet-200 text-violet-900 text-[10px] font-bold">Studio</button></td>
      </tr>`;
    }).join('');
  }

  renderMatrixAndReport(); renderVocabExplorer();
}

function loadStressInStudio(enc) {
  const txt = decodeURIComponent(enc);
  switchTab('studio');
  $('textInput').value = txt;
  $('charCount').textContent = txt.length;
  runPredictionSubmit();
}

function inspectConfusionCell(actLbl, predLbl, count) {
  const box = $('cmClickInsight');
  if (!box) return;
  if (actLbl === predLbl) {
    box.innerHTML = `<strong class="text-emerald-800">Exact Match (${count} cases):</strong> The model correctly recognized <strong>${capitalise(actLbl)}</strong> ${count} times on unseen test texts.`;
  } else {
    box.innerHTML = `<strong class="text-rose-800">Confusion (${count} cases):</strong> Sentences truly labeled <strong>${capitalise(actLbl)}</strong> were mistaken for <strong>${capitalise(predLbl)}</strong> (${count} times) because these two feelings share overlapping words.`;
  }
}

function renderMatrixAndReport() {
  if (!benchmarksData?.benchmarks) return;
  const b = benchmarksData.benchmarks.find(r => r.key === $('cmModelSelect').value) || benchmarksData.benchmarks[0];
  const labels = benchmarksData.labels, cm = b.confusion_matrix, maxVal = Math.max(...cm.flat(), 1);
  let html = '<table class="w-full text-[11px] border-separate border-spacing-1"><thead><tr><th class="p-1 text-slate-400">Act\\Pred</th>';
  labels.forEach(l => { html += `<th class="p-1 capitalize text-slate-700">${l.slice(0, 4)}</th>`; });
  html += '</tr></thead><tbody>';
  cm.forEach((row, rIdx) => {
    html += `<tr><td class="p-1 font-bold capitalize text-slate-700">${labels[rIdx]}</td>`;
    row.forEach((val, cIdx) => {
      const ratio = val / maxVal;
      const bg = rIdx === cIdx ? `rgba(16, 185, 129, ${Math.max(0.16, ratio * 0.85)})` : `rgba(244, 63, 94, ${Math.max(0.06, ratio * 0.65)})`;
      html += `<td onclick="inspectConfusionCell('${labels[rIdx]}','${labels[cIdx]}',${val})" class="cm-cell text-slate-900 cursor-pointer hover:ring-2 hover:ring-violet-500 transition" style="background:${bg}">${val}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  $('confusionMatrixGrid').innerHTML = html;
  $('perClassTbody').innerHTML = labels.map(lbl => { const pc = b.per_class[lbl]; const m = EMOTION_META[lbl] || { emoji: '🤖', color: '#6d28d9' }; return `<tr><td class="p-3 font-bold capitalize" style="color:${m.color}">${m.emoji} ${lbl}</td><td class="p-3 font-mono">${(pc.precision * 100).toFixed(2)}%</td><td class="p-3 font-mono">${(pc.recall * 100).toFixed(2)}%</td><td class="p-3 font-mono font-bold text-slate-900">${(pc.f1 * 100).toFixed(2)}%</td><td class="p-3 font-mono text-slate-500">${pc.support}</td></tr>`; }).join('');
}

function testClueWord(encodedWord, emo) {
  const cleanWord = decodeURIComponent(encodedWord).replace(/^(?:word__|bow__|char__|syntax__)/, '').trim();
  const sample = `Today I feel really ${cleanWord} about how everything went.`;
  loadStressInStudio(encodeURIComponent(sample));
}

function renderVocabExplorer() {
  if (!benchmarksData?.top_tokens) return;
  const emo = $('vocabEmotionSelect').value;
  const data = benchmarksData.top_tokens?.[$('vocabModelSelect').value]?.[emo];
  if (!data) return;
  $('vocabPosList').innerHTML = data.positive.map(t => `<button type="button" onclick="testClueWord('${encodeURIComponent(t.word)}','${emo}')" class="token-pill token-pos hover:ring-2 hover:ring-emerald-400 cursor-pointer">+${t.weight} ${escapeHtml(t.word)}</button>`).join('');
  $('vocabNegList').innerHTML = data.negative.map(t => `<span class="token-pill token-neg">${t.weight} ${escapeHtml(t.word)}</span>`).join('');
}

/* ---------------- Batch Automation ---------------- */
async function runDemoBatch() {
  const lines = [
    'I am thrilled and grateful for our engineering launch today!',
    'I feel exhausted and deeply sad after hearing the news.',
    'This constant delay makes me furious and frustrated.',
    'I am scared about the upcoming test results.',
    'I love spending quiet weekends with my family.',
    'Whoa, I never saw that twist coming at all!',
    'I did not feel angry at all, I felt relieved.',
    'Everything feels lonely tonight.',
    'My heart is bursting with joy and pride!',
    'I feel anxious walking home alone.',
  ];
  await submitBatchJson({ lines });
}
function handleBatchFile(file) { if (!file) return; const fd = new FormData(); fd.append('file', file); fetch('/api/batch', { method: 'POST', body: fd }).then(r => r.json()).then(renderBatchResults); }
async function runPastedBatch() { const raw_text = $('batchTextArea').value.trim(); if (raw_text) await submitBatchJson({ raw_text }); }
async function submitBatchJson(payload) { const res = await fetch('/api/batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }); renderBatchResults(await res.json()); }
function renderBatchResults(data) {
  if (data.error) { alert(data.error); return; }
  $('batchResultsPanel').classList.remove('hidden');
  $('batchTotal').textContent = data.summary.total_entries;
  $('batchDominant').textContent = `${data.summary.dominant_emoji} ${data.summary.dominant_emotion}`;
  $('batchAgree').textContent = `${data.summary.agreement_rate}%`;
  batchCsvExport = data.csv_export || '';
  const triggers = data.summary.top_trigger_words || data.summary.top_triggers || [];
  $('batchTriggerCloud').innerHTML = triggers.map(t => `<span class="token-pill token-pos">${escapeHtml(t.word)} (${t.count})</span>`).join('');
  $('batchTableBody').innerHTML = data.rows.map(r => {
    const m = EMOTION_META[r.emotion] || { color: '#6d28d9' };
    return `<tr><td class="p-2.5 font-mono text-slate-500">${r.row_id}</td><td class="p-2.5 max-w-xs truncate">${escapeHtml(r.text)}</td><td class="p-2.5 font-bold capitalize" style="color:${m.color}">${r.emoji} ${r.emotion} (${r.confidence}%)</td><td class="p-2.5 capitalize">${r.control_emotion || r.tfidf_emotion || '--'}</td><td class="p-2.5 font-mono text-emerald-700">${escapeHtml(r.top_triggers || r.top_trigger || '-')}</td></tr>`;
  }).join('');
  const dist = data.summary.distribution;
  if (batchChart) batchChart.destroy();
  batchChart = new Chart($('batchChart'), { type: 'bar', data: { labels: Object.keys(dist).map(capitalise), datasets: [{ label: 'Entries', data: Object.values(dist), backgroundColor: Object.keys(dist).map(k => EMOTION_META[k].color), borderRadius: 5 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } } });
}
function downloadBatchCsv() { if (!batchCsvExport) return; const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([batchCsvExport], { type: 'text/csv' })); a.download = 'vibecheck_batch_results.csv'; a.click(); }

/* ---------------- Autonomous Agent Drawer ---------------- */
function toggleAgentDrawer() { $('agent-drawer').classList.toggle('closed'); }
function askCoachTopic(prompt) { $('agent-drawer').classList.remove('closed'); $('agent-input').value = prompt; submitAgentChat(new Event('submit')); }

function appendCoachMessage(markdownText, actions = [], quickReplies = [], meta = '') {
  const box = $('agent-messages');
  const wrap = document.createElement('div');
  wrap.className = 'mr-4 p-3 rounded-2xl bg-white border border-purple-100 text-slate-800 shadow-sm space-y-2 agent-msg-prose';
  wrap.innerHTML = renderSimpleMarkdown(markdownText);

  if (quickReplies?.length) {
    const qrRow = document.createElement('div');
    qrRow.className = 'flex flex-wrap gap-1.5 pt-1';
    quickReplies.forEach(q => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'px-2.5 py-1 rounded-full bg-purple-50 hover:bg-purple-100 border border-purple-200 text-violet-900 text-[11px] font-semibold';
      btn.textContent = q;
      btn.onclick = () => askCoachTopic(q);
      qrRow.appendChild(btn);
    });
    wrap.appendChild(qrRow);
  }
  if (meta) {
    const m = document.createElement('div');
    m.className = 'text-[10px] font-mono text-slate-400 pt-1 border-t border-slate-100';
    m.textContent = meta;
    wrap.appendChild(m);
  }
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return wrap;
}

async function submitAgentChat(e) {
  e.preventDefault();
  clearTimeout(hesitationTimer);
  nudgedEmotion = true;
  const input = $('agent-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';

  const box = $('agent-messages');
  const uDiv = document.createElement('div');
  uDiv.className = 'ml-8 p-2.5 rounded-2xl bg-violet-700 text-white font-medium shadow-sm';
  uDiv.textContent = msg;
  box.appendChild(uDiv);
  box.scrollTop = box.scrollHeight;

  const thinking = appendCoachMessage('Analyzing & operating studio tools...');
  const provider = sessionStorage.getItem('emotisense_agent_provider') || 'gemini';
  const apiKey = sessionStorage.getItem('emotisense_agent_apikey') || '';

  try {
    const res = await fetch('/api/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'X-LLM-API-Key': apiKey } : {}) },
      body: JSON.stringify({ message: msg, provider, current_text: lastAnalyzedText || $('textInput').value.trim(), model: $('modelSelect').value, history: agentHistory }),
    });
    const data = await res.json();
    thinking.remove();

    if (data.ui_commands?.length) {
      await executeUiCommands(data.ui_commands);
    }

    agentHistory.push({ role: 'user', content: msg });
    agentHistory.push({ role: 'assistant', content: data.reply || data.error });
    appendCoachMessage(data.reply || data.error, data.actions || [], data.quick_replies || [], data.model || '');
  } catch (err) {
    thinking.remove();
    appendCoachMessage(`Network issue: ${escapeHtml(err.message)}`);
  }
}

/* ---------------- Settings & Utilities ---------------- */
function openAgentSettings() { $('agent-settings-modal').classList.replace('hidden', 'flex'); }
function closeAgentSettings() { $('agent-settings-modal').classList.replace('flex', 'hidden'); }
function loadSavedAgentSettings() { $('agent-provider-select').value = sessionStorage.getItem('emotisense_agent_provider') || 'gemini'; $('agent-apikey-input').value = sessionStorage.getItem('emotisense_agent_apikey') || ''; }
function saveAgentSettings() { sessionStorage.setItem('emotisense_agent_provider', $('agent-provider-select').value); const k = $('agent-apikey-input').value.trim(); if (k) sessionStorage.setItem('emotisense_agent_apikey', k); else sessionStorage.removeItem('emotisense_agent_apikey'); closeAgentSettings(); checkHealth(); resetAgentGreeting(); }
function clearAgentKey() { sessionStorage.removeItem('emotisense_agent_apikey'); $('agent-apikey-input').value = ''; checkHealth(); }
function capitalise(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ''; }
function escapeHtml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function renderSimpleMarkdown(md) {
  let safe = escapeHtml(md).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>');
  const lines = safe.split('\n'); let out = '', inList = false;
  for (const line of lines) { const t = line.trim(); if (t.startsWith('- ') || t.startsWith('* ')) { if (!inList) { out += '<ul>'; inList = true; } out += `<li>${t.slice(2)}</li>`; } else { if (inList) { out += '</ul>'; inList = false; } if (t) out += `<p>${t}</p>`; } }
  if (inList) out += '</ul>'; return out;
}

/* ---------------- Real-World Movie Vibe, Topic Mood & Tone Guard ---------------- */
let lastMovieCard = null;
const DRAFT_EXAMPLES = {
  harsh: "Why did nobody check this before shipping? This delay is ridiculous and I am furious waiting around ASAP.",
  anxious: "I am terrified we missed the deadline and the whole team is stressed that this client presentation will collapse.",
};

function switchPulseMode(mode) {
  const isMovie = mode === 'movie';
  $('pulse-mode-movie').classList.toggle('hidden', !isMovie);
  $('pulse-mode-topic').classList.toggle('hidden', isMovie);
  $('pulse-tab-movie').className = isMovie
    ? 'px-3 py-1.5 rounded-lg text-xs font-bold bg-white text-violet-900 shadow-sm'
    : 'px-3 py-1.5 rounded-lg text-xs font-bold text-slate-600 hover:text-slate-900';
  $('pulse-tab-topic').className = !isMovie
    ? 'px-3 py-1.5 rounded-lg text-xs font-bold bg-white text-violet-900 shadow-sm'
    : 'px-3 py-1.5 rounded-lg text-xs font-bold text-slate-600 hover:text-slate-900';
}

function quickMovieScan(title) { $('movieQueryInput').value = title; scanMovieVibe(); }
function quickTopicPulse(topic) { $('topicQueryInput').value = topic; scanTopicPulse(); }

async function scanMovieVibe() {
  const title = $('movieQueryInput').value.trim();
  const errEl = $('moviePulseError');
  errEl.classList.add('hidden');
  if (!title) { errEl.textContent = 'Type a movie or series name first!'; errEl.classList.remove('hidden'); return; }
  const btn = $('movieScanBtn');
  const origHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span>Scanning live reviews...</span>';
  try {
    const res = await fetch('/api/pulse/movie', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, model: $('modelSelect').value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Scan failed');
    renderMovieVibeResults(data);
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btn.innerHTML = origHtml;
    lucide.createIcons();
  }
}

function renderMovieVibeResults(data) {
  lastMovieCard = data;
  $('moviePulseResults').classList.remove('hidden');
  const meta = data.meta || {};
  const parts = [data.matched_title || data.query];
  if (meta.premiered) parts.push(meta.premiered);
  if (Array.isArray(meta.genres) && meta.genres.length) parts.push(meta.genres.slice(0, 2).join(' · '));
  if (meta.rating) parts.push(`★ ${meta.rating}`);
  $('movieMetaBadge').textContent = parts.join(' · ');
  $('movieVerdictTitle').textContent = data.verdict_title;
  $('movieVerdictSub').textContent = data.verdict_subtitle;
  $('movieDominantEmoji').textContent = data.dominant_emoji || '🎬';
  $('movieSampleNote').textContent = `Scored ${data.sample_count} real quotes (Wiki ${meta.wiki_quotes || 0} + News ${meta.news_quotes || 0})`;

  const vi = data.vibe_indices || {};
  const meterRows = [
    { label: '😄 Feel-Good / Hype', val: vi.hype_joy || 0, color: '#d97706' },
    { label: '😨 Edge-of-Seat Suspense', val: vi.suspense || 0, color: '#7c3aed' },
    { label: '😭 Tearjerker Depth', val: vi.tearjerker || 0, color: '#2563eb' },
    { label: '🔥 Heated Debate', val: vi.heat_rant || 0, color: '#dc2626' },
  ];
  $('movieMeters').innerHTML = meterRows.map(m =>
    `<div class="space-y-1"><div class="flex justify-between text-[11px] font-bold"><span class="text-slate-700">${m.label}</span><span class="font-mono text-slate-900">${m.val}%</span></div><div class="w-full h-2 bg-white rounded-full overflow-hidden border border-purple-100"><div class="h-full rounded-full" style="width:${Math.min(100, m.val)}%;background:${m.color}"></div></div></div>`
  ).join('');

  $('movieTriggersRow').innerHTML = (data.top_triggers || []).slice(0, 8).map(t =>
    `<span class="px-2 py-0.5 rounded-full bg-white border border-purple-200 text-[10px] font-mono text-violet-900 font-semibold">${escapeHtml(t.word)} (${t.count})</span>`
  ).join('');

  $('movieEvidenceList').innerHTML = (data.items || []).map(it =>
    `<div class="p-2.5 rounded-xl bg-slate-50/90 border border-slate-200 space-y-1"><div class="flex items-center justify-between gap-2 text-[10px]"><span class="font-bold text-slate-500 truncate">${escapeHtml(it.source)}${it.date ? ' · ' + escapeHtml(it.date) : ''}</span><span class="shrink-0 px-2 py-0.5 rounded-full font-bold capitalize text-white" style="background:${it.color}">${it.emoji} ${it.emotion} ${Math.round(it.confidence)}%</span></div><p class="text-xs text-slate-800 leading-snug">${escapeHtml(it.text)}</p>${it.top_word ? `<div class="text-[10px] font-mono text-violet-700">Signal word: <strong>${escapeHtml(it.top_word)}</strong></div>` : ''}</div>`
  ).join('');
}

async function copyMovieVibeCard() {
  if (!lastMovieCard) return;
  const vi = lastMovieCard.vibe_indices || {};
  const txt = `${lastMovieCard.matched_title || lastMovieCard.query} — ${lastMovieCard.verdict_title}\nAnalyzed ${lastMovieCard.sample_count} real critic & news quotes with VibeCheck:\nFeel-Good: ${vi.hype_joy}% | Suspense: ${vi.suspense}% | Tearjerker: ${vi.tearjerker}% | Debate: ${vi.heat_rant}%`;
  try { await navigator.clipboard.writeText(txt); $('movieSampleNote').textContent = 'Copied vibe card to clipboard!'; } catch { /* ignore */ }
}

async function scanTopicPulse() {
  const topic = $('topicQueryInput').value.trim();
  const errEl = $('topicPulseError');
  errEl.classList.add('hidden');
  if (!topic) { errEl.textContent = 'Type a topic first!'; errEl.classList.remove('hidden'); return; }
  const btn = $('topicScanBtn');
  const origHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span>Pulsing live mood...</span>';
  try {
    const res = await fetch('/api/pulse/topic', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, model: $('modelSelect').value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Pulse failed');
    renderTopicPulseResults(data);
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btn.innerHTML = origHtml;
    lucide.createIcons();
  }
}

function renderTopicPulseResults(data) {
  $('topicPulseResults').classList.remove('hidden');
  $('topicMoodHeadline').textContent = `${data.query} · ${data.mood_headline}`;
  $('topicDominantEmoji').textContent = data.dominant_emoji || '🌍';
  const vi = data.vibe_indices || {};
  $('topicWarmthVal').textContent = `${vi.warmth_temperature ?? 50}%`;
  $('topicPolarVal').textContent = `${vi.polarization ?? 50}%`;
  $('topicMixPills').innerHTML = Object.entries(data.emotion_mix || {})
    .sort((a, b) => b[1] - a[1])
    .map(([emo, pct]) => `<span class="px-2 py-0.5 rounded-full bg-slate-100 border border-slate-200 text-[10px] font-bold capitalize">${(EMOTION_META[emo]?.emoji || '')} ${emo}: ${pct}%</span>`)
    .join('');

  $('topicFeedList').innerHTML = (data.items || []).map(it =>
    `<div class="p-2.5 rounded-xl bg-slate-50 border border-slate-200 flex items-start justify-between gap-2.5"><div class="space-y-0.5"><div class="text-[10px] font-bold text-slate-500">${escapeHtml(it.source)}</div><div class="text-xs font-semibold text-slate-900">${it.url ? `<a href="${escapeHtml(it.url)}" target="_blank" rel="noopener" class="hover:underline">${escapeHtml(it.text)}</a>` : escapeHtml(it.text)}</div></div><span class="shrink-0 px-2 py-0.5 rounded-full text-[10px] font-bold capitalize text-white" style="background:${it.color}">${it.emoji} ${it.emotion}</span></div>`
  ).join('');
}

function tryDraftToneExample(kind) {
  $('textInput').value = DRAFT_EXAMPLES[kind] || DRAFT_EXAMPLES.harsh;
  $('charCount').textContent = $('textInput').value.length;
  runToneGuard();
}

async function runToneGuard() {
  const text = $('textInput').value.trim();
  if (!text) { $('inputError').textContent = 'Paste a draft message above to check its tone.'; return; }
  $('inputError').textContent = '';
  const btn = $('toneGuardBtn');
  const orig = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = 'Checking Tone...';
  const apiKey = sessionStorage.getItem('emotisense_agent_apikey') || '';
  const provider = sessionStorage.getItem('emotisense_agent_provider') || 'gemini';
  try {
    const res = await fetch('/api/tone-guard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(apiKey ? { 'X-LLM-API-Key': apiKey } : {}) },
      body: JSON.stringify({ text, goal: $('toneGoalSelect').value, provider }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Tone check failed');
    renderToneGuardResults(data);
    runPredictionSubmit();
  } catch (err) {
    $('inputError').textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.innerHTML = orig;
    lucide.createIcons();
  }
}

function renderToneGuardResults(data) {
  $('toneGuardResults').classList.remove('hidden');
  const clsMap = {
    warning: 'bg-rose-50 border-rose-200 text-rose-900',
    caution: 'bg-amber-50 border-amber-200 text-amber-900',
    good: 'bg-sky-50 border-sky-200 text-sky-900',
    great: 'bg-emerald-50 border-emerald-200 text-emerald-900',
  };
  const orig = data.original || {};
  $('toneGuardVerdict').className = `p-3 rounded-xl text-xs font-bold border ${clsMap[data.status_level] || clsMap.good}`;
  $('toneGuardVerdict').innerHTML = `<div>${orig.emoji || ''} ${escapeHtml(data.verdict)}</div><div class="text-[10px] font-mono opacity-80 mt-1">Source: ${escapeHtml(data.rewrite_source || '')}</div>`;

  $('toneGuardRewrites').innerHTML = (data.rewrites || []).map(rw => {
    const enc = encodeURIComponent(rw.text || '');
    const dropTxt = rw.anger_delta < 0 ? `${rw.anger_delta}% anger` : (rw.emotion === 'joy' || rw.emotion === 'love' ? `+warmth verified` : `${rw.confidence}% ${rw.emotion}`);
    return `<div class="p-3 rounded-xl bg-white border border-purple-200 space-y-2 shadow-sm"><div class="flex items-center justify-between gap-2"><span class="text-[11px] font-extrabold text-violet-900">${escapeHtml(rw.label)}</span><span class="px-2 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-800 text-[10px] font-mono font-bold">${rw.emoji} ${capitalise(rw.emotion)} ${Math.round(rw.confidence)}% · ${dropTxt}</span></div><p class="text-xs text-slate-800 leading-relaxed">"${escapeHtml(rw.text)}"</p><div class="text-[10px] text-slate-500">${escapeHtml(rw.why || '')}</div><div class="flex items-center gap-2 pt-1"><button type="button" onclick="applyToneRewrite('${enc}')" class="px-2.5 py-1 rounded-lg bg-violet-50 hover:bg-violet-100 border border-violet-200 text-violet-900 text-[10px] font-bold">Use &amp; Re-Analyze</button><button type="button" onclick="copyToneText('${enc}')" class="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-[10px] font-bold">Copy</button></div></div>`;
  }).join('');
}

function applyToneRewrite(enc) {
  const txt = decodeURIComponent(enc);
  $('textInput').value = txt;
  $('charCount').textContent = txt.length;
  runPredictionSubmit();
}

async function copyToneText(enc) {
  try { await navigator.clipboard.writeText(decodeURIComponent(enc)); } catch { /* ignore */ }
}

/* ---------------- Moveable / Draggable Chatbot Engine ---------------- */
function initDraggableChatbot(drawerId, handleId, fabId, storageKey) {
  const drawer = $(drawerId);
  const handle = $(handleId);
  const fab = $(fabId);
  if (!drawer || !handle) return;

  const clampDrawer = (left, top) => {
    const w = drawer.offsetWidth || 400;
    const h = drawer.offsetHeight || 520;
    return {
      left: Math.max(8, Math.min(window.innerWidth - w - 8, left)),
      top: Math.max(8, Math.min(window.innerHeight - h - 8, top)),
    };
  };

  try {
    const saved = JSON.parse(sessionStorage.getItem(storageKey) || 'null');
    if (saved && typeof saved.left === 'number' && typeof saved.top === 'number') {
      const pos = clampDrawer(saved.left, saved.top);
      drawer.style.left = `${pos.left}px`;
      drawer.style.top = `${pos.top}px`;
      drawer.style.right = 'auto';
      drawer.style.bottom = 'auto';
    }
  } catch { /* ignore */ }

  let dragState = null;
  handle.addEventListener('pointerdown', e => {
    if (e.target.closest('button')) return;
    const rect = drawer.getBoundingClientRect();
    drawer.classList.add('dragging');
    dragState = { startX: e.clientX, startY: e.clientY, origLeft: rect.left, origTop: rect.top };
    handle.setPointerCapture(e.pointerId);
  });
  handle.addEventListener('pointermove', e => {
    if (!dragState) return;
    const pos = clampDrawer(dragState.origLeft + (e.clientX - dragState.startX), dragState.origTop + (e.clientY - dragState.startY));
    drawer.style.left = `${pos.left}px`;
    drawer.style.top = `${pos.top}px`;
    drawer.style.right = 'auto';
    drawer.style.bottom = 'auto';
  });
  const endDrawerDrag = () => {
    if (!dragState) return;
    dragState = null;
    drawer.classList.remove('dragging');
    const r = drawer.getBoundingClientRect();
    sessionStorage.setItem(storageKey, JSON.stringify({ left: Math.round(r.left), top: Math.round(r.top) }));
  };
  handle.addEventListener('pointerup', endDrawerDrag);
  handle.addEventListener('pointercancel', endDrawerDrag);
  handle.addEventListener('dblclick', e => {
    if (e.target.closest('button')) return;
    drawer.style.left = ''; drawer.style.top = ''; drawer.style.right = ''; drawer.style.bottom = '';
    sessionStorage.removeItem(storageKey);
  });

  if (fab) {
    let fabDrag = null;
    fab.addEventListener('pointerdown', e => {
      const r = fab.getBoundingClientRect();
      fabDrag = { startX: e.clientX, startY: e.clientY, origLeft: r.left, origTop: r.top, moved: false };
      fab.setPointerCapture(e.pointerId);
    });
    fab.addEventListener('pointermove', e => {
      if (!fabDrag) return;
      const dx = e.clientX - fabDrag.startX;
      const dy = e.clientY - fabDrag.startY;
      if (Math.hypot(dx, dy) > 6) {
        fabDrag.moved = true;
        fab.classList.add('dragging');
        const left = Math.max(8, Math.min(window.innerWidth - fab.offsetWidth - 8, fabDrag.origLeft + dx));
        const top = Math.max(8, Math.min(window.innerHeight - fab.offsetHeight - 8, fabDrag.origTop + dy));
        fab.style.left = `${left}px`; fab.style.top = `${top}px`; fab.style.right = 'auto'; fab.style.bottom = 'auto';
      }
    });
    const endFab = () => {
      if (!fabDrag) return;
      const wasMoved = fabDrag.moved;
      fabDrag = null;
      fab.classList.remove('dragging');
      if (!wasMoved) toggleAgentDrawer();
    };
    fab.addEventListener('pointerup', endFab);
    fab.addEventListener('pointercancel', endFab);
    fab.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleAgentDrawer(); } });
  }

  window.addEventListener('resize', () => {
    if (!drawer.style.left) return;
    const pos = clampDrawer(parseFloat(drawer.style.left), parseFloat(drawer.style.top));
    drawer.style.left = `${pos.left}px`;
    drawer.style.top = `${pos.top}px`;
  });
}
