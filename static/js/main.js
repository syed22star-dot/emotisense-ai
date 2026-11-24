'use strict';

const EMOTION_META = {
  sadness:  { emoji: '😢', color: '#2563eb' },
  anger:    { emoji: '😠', color: '#dc2626' },
  love:     { emoji: '❤️', color: '#db2777' },
  surprise: { emoji: '😲', color: '#0d9488' },
  fear:     { emoji: '😨', color: '#7c3aed' },
  joy:      { emoji: '😄', color: '#d97706' },
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

const $ = id => document.getElementById(id);

window.addEventListener('DOMContentLoaded', async () => {
  lucide.createIcons();
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
    'Hi! 👋 I’m **EmotiCoach AI**, your autonomous NLP site agent.\n\nShare any message, feeling, or journal lines with me — I can **automatically enter it into Emotion Studio, highlight the trigger words, compare all 4 models, or run Batch Journal screening live on screen!**' + keyHint,
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
      <div class="soft-card p-3 rounded-xl"><div class="text-[10px] font-mono text-slate-500">Acc Delta (95% CI)</div><div class="text-sm font-extrabold ${bt.accuracy_delta_pct?.mean >= 0 ? 'text-emerald-700' : 'text-rose-700'} mt-0.5">${bt.accuracy_delta_pct?.mean >= 0 ? '+' : ''}${bt.accuracy_delta_pct?.mean}%</div><div class="text-[10px] font-mono text-slate-500">[${bt.accuracy_delta_pct?.ci_low}%, +${bt.accuracy_delta_pct?.ci_high}%]</div></div>
      <div class="soft-card p-3 rounded-xl"><div class="text-[10px] font-mono text-slate-500">Macro F1 Delta</div><div class="text-sm font-extrabold ${bt.macro_f1_delta_pct?.mean >= 0 ? 'text-violet-700' : 'text-rose-700'} mt-0.5">${bt.macro_f1_delta_pct?.mean >= 0 ? '+' : ''}${bt.macro_f1_delta_pct?.mean}%</div><div class="text-[10px] font-mono text-slate-500">${bt.prob_variant_beats_control_f1}% Boot Win</div></div>
      <div class="soft-card p-3 rounded-xl"><div class="text-[10px] font-mono text-slate-500">McNemar Test</div><div class="text-sm font-extrabold text-slate-900 mt-0.5">p = ${mc.p_value}</div><div class="text-[10px] font-mono text-slate-500">${mc.significant_alpha_05 ? 'Significant (<0.05)' : 'Discordant Δ ' + mc.net_wins_for_variant}</div></div>
      <div class="soft-card p-3 rounded-xl"><div class="text-[10px] font-mono text-slate-500">Brier Calibration</div><div class="text-sm font-extrabold text-emerald-700 mt-0.5">${bt.brier_delta?.mean}</div><div class="text-[10px] font-mono text-slate-500">Control ${bt.brier_control?.mean} -> Var ${bt.brier_variant?.mean}</div></div>
    </div>
    <div class="grid grid-cols-4 gap-2 text-center text-[11px] pt-1">
      <div class="p-2 rounded-lg bg-emerald-50 border border-emerald-200"><div class="font-mono font-bold text-emerald-800">${c.both_correct}</div><div class="text-slate-600">Both Correct</div></div>
      <div class="p-2 rounded-lg bg-violet-50 border border-violet-200"><div class="font-mono font-bold text-violet-800">${c.variant_only_correct}</div><div class="text-slate-600">Hybrid Only Won</div></div>
      <div class="p-2 rounded-lg bg-amber-50 border border-amber-200"><div class="font-mono font-bold text-amber-800">${c.control_only_correct}</div><div class="text-slate-600">BoW Only Won</div></div>
      <div class="p-2 rounded-lg bg-slate-100 border border-slate-200"><div class="font-mono font-bold text-slate-700">${c.both_wrong}</div><div class="text-slate-600">Both Missed</div></div>
    </div>
  `;
}

async function runLiveAbArena(overrideText) {
  const box = $('liveAbArena');
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
        <div class="soft-card p-3 rounded-xl"><div class="text-[10px] font-mono text-slate-500 uppercase">Control: BoW+LR</div><div class="text-lg font-extrabold capitalize mt-0.5" style="color:${cm.color}">${cm.emoji} ${d.control.emotion}</div><div class="text-[11px] font-mono text-slate-600">${d.control.confidence.toFixed(1)}%</div></div>
        <div class="soft-card p-3 rounded-xl bg-purple-50/70 border-purple-200"><div class="text-[10px] font-mono text-violet-800 uppercase">Variant B: Hybrid</div><div class="text-lg font-extrabold capitalize mt-0.5" style="color:${vm.color}">${vm.emoji} ${d.variant.emotion}</div><div class="text-[11px] font-mono text-violet-800">${d.variant.confidence.toFixed(1)}%</div></div>
      </div>
      <div class="p-2.5 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-between text-[11px]">
        <span>Confidence Shift: <strong class="font-mono">${d.confidence_delta > 0 ? '+' : ''}${d.confidence_delta}%</strong></span>
        <span class="font-bold ${d.decision_flip ? 'text-amber-700' : 'text-emerald-700'}">${d.decision_flip ? 'Decision Flipped by Hybrid Features!' : 'Same Class Consensus'}</span>
      </div>
    `;
  } catch (err) { console.error(err); }
}

function renderResearchLab() {
  if (!benchmarksData?.benchmarks) return;
  renderAbStatsPanel();
  runLiveAbArena();
  const rows = benchmarksData.benchmarks;
  const colors = ['#6d28d9', '#db2777', '#0d9488', '#0284c7', '#d97706', '#9333ea', '#64748b'];
  if (benchFoldsChart) benchFoldsChart.destroy();
  benchFoldsChart = new Chart($('benchFoldsChart'), { type: 'line', data: { labels: ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'], datasets: rows.map((b, i) => ({ label: b.model, data: (b.cv_fold_accuracies || []).map(v => +(v * 100).toFixed(2)), borderColor: colors[i % colors.length], backgroundColor: colors[i % colors.length], tension: 0.25 })) }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 55, max: 96 } } } });

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
  if ($('ablationChart') && ablation.length) {
    if (ablationChart) ablationChart.destroy();
    ablationChart = new Chart($('ablationChart'), {
      type: 'bar',
      data: {
        labels: ablation.map(a => a.stage.replace(/^\d+\.\s*/, '')),
        datasets: [
          { label: '5-Fold CV Acc %', data: ablation.map(a => a.cv_accuracy), backgroundColor: '#7c3aed', borderRadius: 4 },
          { label: 'Held-Out Test %', data: ablation.map(a => a.test_accuracy), backgroundColor: '#059669', borderRadius: 4 },
        ],
      },
      options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, scales: { x: { min: 75, max: 96 } } },
    });
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
          { label: `Control BoW+LR (ECE ${bow.ece})`, data: bow.reliability_bins.map(b => ({ x: b.mean_conf, y: b.empirical_acc })), borderColor: '#db2777', backgroundColor: '#db2777', pointRadius: 3, tension: 0.2 },
          { label: 'Ideal Calibration', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: '#cbd5e1', borderDash: [4, 4], pointRadius: 0 },
        ],
      },
      options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'linear', min: 0, max: 1 }, y: { min: 0, max: 1 } } },
    });
  }

  const stress = benchmarksData.stress_suite;
  if ($('stressSuiteTbody') && stress?.cases) {
    $('stressPassBadge').textContent = `Control ${stress.control_pass_count}/${stress.total} vs Hybrid ${stress.variant_pass_count}/${stress.total}`;
    $('stressSuiteTbody').innerHTML = stress.cases.map(c => `
      <tr>
        <td class="p-2.5 font-bold text-slate-800">${escapeHtml(c.category)}</td>
        <td class="p-2.5 text-slate-600 max-w-xs truncate" title="${escapeHtml(c.text)}">${escapeHtml(c.text)}</td>
        <td class="p-2.5 font-bold capitalize">${c.expected}</td>
        <td class="p-2.5 font-mono ${c.control_pass ? 'text-emerald-700 font-bold' : 'text-rose-600 font-bold'}">${c.control_pass ? '✓' : '✗'} ${c.control_pred} (${c.control_conf}%)</td>
        <td class="p-2.5 font-mono ${c.variant_pass ? 'text-emerald-700 font-bold' : 'text-rose-600 font-bold'}">${c.variant_pass ? '✓' : '✗'} ${c.variant_pred} (${c.variant_conf}%)</td>
      </tr>
    `).join('');
  }

  renderMatrixAndReport(); renderVocabExplorer();
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
      html += `<td class="cm-cell text-slate-900" style="background:${bg}">${val}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  $('confusionMatrixGrid').innerHTML = html;
  $('perClassTbody').innerHTML = labels.map(lbl => { const pc = b.per_class[lbl]; const m = EMOTION_META[lbl] || { emoji: '🤖', color: '#6d28d9' }; return `<tr><td class="p-3 font-bold capitalize" style="color:${m.color}">${m.emoji} ${lbl}</td><td class="p-3 font-mono">${(pc.precision * 100).toFixed(2)}%</td><td class="p-3 font-mono">${(pc.recall * 100).toFixed(2)}%</td><td class="p-3 font-mono font-bold text-slate-900">${(pc.f1 * 100).toFixed(2)}%</td><td class="p-3 font-mono text-slate-500">${pc.support}</td></tr>`; }).join('');
}
function renderVocabExplorer() {
  if (!benchmarksData?.top_tokens) return;
  const data = benchmarksData.top_tokens?.[$('vocabModelSelect').value]?.[$('vocabEmotionSelect').value];
  if (!data) return;
  $('vocabPosList').innerHTML = data.positive.map(t => `<span class="token-pill token-pos">+${t.weight} ${escapeHtml(t.word)}</span>`).join('');
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
function downloadBatchCsv() { if (!batchCsvExport) return; const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([batchCsvExport], { type: 'text/csv' })); a.download = 'emotisense_batch_results.csv'; a.click(); }

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
