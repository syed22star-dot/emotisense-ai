'use strict';

const EMOTION_META = {
  sadness:  { emoji: '😢', color: '#3b82f6' },
  anger:    { emoji: '😠', color: '#ef4444' },
  love:     { emoji: '❤️', color: '#ec4899' },
  surprise: { emoji: '😲', color: '#14b8a6' },
  fear:     { emoji: '😨', color: '#a855f7' },
  joy:      { emoji: '😄', color: '#eab308' },
};

const CHIP_SAMPLES = {
  joy: 'I feel so incredibly happy and grateful for everything happening in my life right now!',
  sadness: 'I feel a deep crushing sadness in my chest, like the world lost its warmth.',
  anger: 'I am furious about how unfairly this was handled — it really frustrated me.',
  fear: 'I am terrified about what might happen next and cannot stop worrying.',
  love: 'I love spending quiet evenings with my family — it fills my heart with gratitude.',
  surprise: 'Oh wow, I was totally shocked and did not expect that announcement at all!',
  negation: 'I did not feel humiliated or angry, actually I felt relieved and grateful.',
};

const HISTORY_KEY = 'emotisense_history_v2';
let history = loadHistory();
let benchmarksData = null;
let batchCsvExport = '';
let lastAnalyzedText = '';
let agentHistory = [];
let previousInteractionId = null;

let sessionChart = null;
let batchChart = null;
let benchModelsChart = null;

const $ = id => document.getElementById(id);

window.addEventListener('DOMContentLoaded', async () => {
  lucide.createIcons();
  loadSavedAgentSettings();
  renderHistory();
  renderSessionChart();
  await Promise.all([checkHealth(), loadBenchmarks()]);
});

/* ---------------- Health & Benchmarks ---------------- */
async function checkHealth() {
  const pill = $('header-health-pill');
  const dot = $('agent-status-dot');
  const keyBadge = $('agent-key-badge');
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const serverOk = Boolean(data?.llm_agent?.configured);
    const hasSession = Boolean(sessionStorage.getItem('emotisense_agent_apikey'));
    pill.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span><span>BoW+LR Ready · ${Number(data.vocab_size || 0).toLocaleString()} tokens · sklearn ${data.sklearn_version}</span>`;

    if (serverOk || hasSession) {
      dot.className = 'w-2 h-2 rounded-full bg-emerald-400';
      keyBadge.textContent = serverOk ? 'Server Key Active' : 'Session Key Active';
      keyBadge.className = 'text-[10px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-500/30';
    } else {
      dot.className = 'w-2 h-2 rounded-full bg-amber-400';
      keyBadge.textContent = 'API Key Needed';
      keyBadge.className = 'text-[10px] px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-500/30';
    }
  } catch {
    pill.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-rose-500"></span><span>Backend Offline</span>`;
  }
}

async function loadBenchmarks() {
  try {
    const res = await fetch('/api/benchmarks');
    benchmarksData = await res.json();
  } catch (err) {
    console.error(err);
  }
}

/* ---------------- Tabs ---------------- */
function switchTab(tab) {
  ['studio', 'batch', 'benchmarks'].forEach(id => {
    $(`tab-${id}`).classList.toggle('hidden', id !== tab);
    $(`tab-btn-${id}`).classList.toggle('active', id === tab);
  });
  if (tab === 'benchmarks') setTimeout(renderBenchmarksTab, 80);
}

/* ---------------- Text Input Controls ---------------- */
$('textInput').addEventListener('input', () => {
  $('charCount').textContent = $('textInput').value.length;
  $('inputError').textContent = '';
});

$('textInput').addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    $('predictForm').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  }
});

function loadChipSample(key) {
  $('textInput').value = CHIP_SAMPLES[key] || '';
  $('charCount').textContent = $('textInput').value.length;
  $('inputError').textContent = '';
  $('textInput').focus();
}

function clearInput() {
  $('textInput').value = '';
  $('charCount').textContent = '0';
  $('inputError').textContent = '';
  $('resultSection').classList.add('hidden');
}

/* ---------------- Prediction Flow ---------------- */
$('predictForm').addEventListener('submit', async e => {
  e.preventDefault();
  const text = $('textInput').value.trim();
  if (!text) {
    $('inputError').textContent = 'Please enter some text to classify.';
    return;
  }

  const btn = $('analyzeBtn');
  btn.disabled = true;
  $('analyzeBtnLabel').textContent = 'Analysing...';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        model: $('modelSelect').value,
        compare_models: $('compareToggle').checked,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      $('inputError').textContent = data.error || 'Prediction failed.';
      return;
    }

    lastAnalyzedText = text;
    renderResult(data);
    pushHistory(data);
  } catch (err) {
    $('inputError').textContent = `Network error: ${err.message}`;
  } finally {
    btn.disabled = false;
    $('analyzeBtnLabel').textContent = 'Analyse Emotion';
  }
});

function renderResult(data) {
  const meta = EMOTION_META[data.emotion] || { emoji: '🤖', color: '#6366f1' };
  $('resultSection').classList.remove('hidden');
  $('resultEmoji').textContent = meta.emoji;
  $('resultEmotion').textContent = data.emotion;
  $('resultEmotion').style.color = meta.color;
  $('resultModelBadge').textContent = `${data.model_name} · Test ${data.accuracy}`;
  $('resultRunnerUp').textContent = `Confidence margin vs ${data.runner_up.emotion} (${data.runner_up.confidence}%): +${data.margin_pct}%`;

  const ring = $('confRingFill');
  ring.style.stroke = meta.color;
  ring.style.strokeDashoffset = String(97.4 * (1 - data.confidence / 100));
  $('confPct').textContent = `${Math.round(data.confidence)}%`;

  // Inline XAI Sentence Heatmap
  const heatmap = $('sentenceHeatmap');
  heatmap.innerHTML = (data.sentence_tokens || []).map(tok => {
    const cls = tok.impact === 'positive' ? 'token-pos' : (tok.impact === 'negative' ? 'token-neg' : 'token-oov');
    const title = tok.impact === 'oov' ? 'Out of vocabulary' : `Weight contribution: ${tok.score > 0 ? '+' : ''}${tok.score}`;
    return `<span class="token-pill ${cls}" title="${title}">${escapeHtml(tok.word)}</span>`;
  }).join('');

  // Probability bars
  const sortedProbs = Object.entries(data.all_probabilities).sort((a, b) => b[1] - a[1]);
  $('probGrid').innerHTML = sortedProbs.map(([lbl, pct]) => {
    const m = EMOTION_META[lbl] || { emoji: '🤖', color: '#6366f1' };
    return `
      <div class="flex items-center gap-3 text-xs">
        <div class="w-24 shrink-0 font-semibold capitalize text-slate-200">${m.emoji} ${lbl}</div>
        <div class="flex-1 h-2 rounded-full bg-slate-900 overflow-hidden">
          <div class="prob-fill h-full rounded-full" style="width:${pct}%; background:${m.color}"></div>
        </div>
        <div class="w-12 text-right font-mono font-bold text-slate-300">${pct.toFixed(1)}%</div>
      </div>
    `;
  }).join('');

  // Trigger chips
  $('xaiPosChips').innerHTML = (data.token_importance || []).length
    ? data.token_importance.map(t => `<span class="token-pill token-pos">+${t.score.toFixed(2)} ${escapeHtml(t.word)}</span>`).join('')
    : '<span class="text-xs text-slate-500">No positive trigger words matched.</span>';

  $('xaiNegChips').innerHTML = (data.opposing_tokens || []).length
    ? data.opposing_tokens.map(t => `<span class="token-pill token-neg">${t.score.toFixed(2)} ${escapeHtml(t.word)}</span>`).join('')
    : '<span class="text-xs text-slate-500">No strong opposing tokens matched.</span>';

  // 4-Model Arena
  const arenaSec = $('arenaSection');
  if (data.arena) {
    arenaSec.classList.remove('hidden');
    const a = data.arena;
    const badge = $('arenaConsensusBadge');
    badge.textContent = a.unanimous
      ? `Unanimous (${a.consensus_count}/${a.total_models} ${capitalise(a.consensus_emotion)})`
      : `Majority (${a.consensus_count}/${a.total_models} ${capitalise(a.consensus_emotion)})`;

    $('arenaGrid').innerHTML = a.models.map(m => `
      <div class="glass-card p-3.5 rounded-xl flex items-center justify-between gap-2">
        <div>
          <div class="text-xs font-bold text-white">${escapeHtml(m.name)}</div>
          <div class="text-[10px] font-mono text-slate-400">${escapeHtml(m.badge)} · Test ${m.test_accuracy}%</div>
        </div>
        <div class="text-right">
          <div class="text-sm font-bold capitalize" style="color:${m.color}">${m.emoji} ${m.emotion}</div>
          <div class="text-[11px] font-mono text-slate-300">${m.confidence.toFixed(1)}%</div>
        </div>
      </div>
    `).join('');
  } else {
    arenaSec.classList.add('hidden');
  }

  // Sync Agent live context banner
  $('agent-context-pill').firstElementChild.textContent =
    `Synced: "${data.original_text.slice(0, 32)}..." -> ${capitalise(data.emotion)} (${data.confidence}%)`;
}

/* ---------------- History & Session Chart ---------------- */
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}

function pushHistory(data) {
  history.unshift({
    emotion: data.emotion,
    confidence: data.confidence,
    text: data.original_text,
    ts: Date.now(),
  });
  if (history.length > 25) history.pop();
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  renderHistory();
  renderSessionChart();
}

function clearHistory() {
  history = [];
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
  renderSessionChart();
}

function renderHistory() {
  $('sessionCountBadge').textContent = `${history.length} samples`;
  const list = $('historyList');
  if (!history.length) {
    list.innerHTML = '<li class="text-xs text-slate-500 text-center py-6">No analyses yet. Submit a sentence on the left!</li>';
    return;
  }
  list.innerHTML = history.map(h => {
    const m = EMOTION_META[h.emotion] || { emoji: '🤖', color: '#6366f1' };
    return `
      <li onclick="reloadFromHistory(${JSON.stringify(h.text).replace(/"/g, '&quot;')})" class="glass-card p-2.5 rounded-xl flex items-center justify-between gap-2 cursor-pointer">
        <div class="flex items-center gap-2.5 min-w-0">
          <span class="text-lg">${m.emoji}</span>
          <div class="min-w-0">
            <div class="text-xs font-bold capitalize" style="color:${m.color}">${h.emotion}</div>
            <div class="text-[11px] text-slate-400 truncate">${escapeHtml(h.text)}</div>
          </div>
        </div>
        <span class="text-[11px] font-mono font-bold text-slate-300 shrink-0">${h.confidence.toFixed(0)}%</span>
      </li>
    `;
  }).join('');
}

function reloadFromHistory(txt) {
  $('textInput').value = txt;
  $('charCount').textContent = txt.length;
  $('predictForm').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
}

function renderSessionChart() {
  const counts = {};
  Object.keys(EMOTION_META).forEach(k => { counts[k] = 0; });
  history.forEach(h => { counts[h.emotion] = (counts[h.emotion] || 0) + 1; });
  if (sessionChart) sessionChart.destroy();
  sessionChart = new Chart($('sessionChart'), {
    type: 'doughnut',
    data: {
      labels: Object.keys(counts).map(capitalise),
      datasets: [{
        data: Object.values(counts),
        backgroundColor: Object.keys(counts).map(k => EMOTION_META[k].color),
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'right', labels: { color: '#cbd5e1', boxWidth: 10, font: { size: 11 } } } },
    },
  });
}

/* ---------------- Batch Automation Tab ---------------- */
async function runDemoBatch() {
  const demoLines = [
    'I am thrilled and grateful for our engineering launch today!',
    'I feel exhausted and deeply sad after hearing the bad news.',
    'This constant delay makes me furious and frustrated.',
    'I am genuinely scared about the upcoming medical test results.',
    'I love spending quiet weekends with my family.',
    'Whoa, I never saw that twist coming at all!',
    'I did not feel angry at all, I actually felt relieved.',
    'Everything feels hopeless and lonely tonight.',
    'My heart is bursting with joy and pride!',
    'I feel anxious and uneasy walking home alone.',
  ];
  await submitBatchJson({ lines: demoLines });
}

function handleBatchFile(file) {
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  fetch('/api/batch', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(renderBatchResults)
    .catch(err => alert(err.message));
}

function runPastedBatch() {
  const raw = $('batchTextArea').value.trim();
  if (!raw) return;
  submitBatchJson({ raw_text: raw });
}

async function submitBatchJson(payload) {
  const res = await fetch('/api/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) { alert(data.error || 'Batch failed'); return; }
  renderBatchResults(data);
}

function renderBatchResults(data) {
  if (data.error) { alert(data.error); return; }
  $('batchResultsPanel').classList.remove('hidden');
  $('batchTotal').textContent = data.summary.total_entries;
  $('batchDominant').textContent = `${data.summary.dominant_emoji} ${data.summary.dominant_emotion}`;
  $('batchAgree').textContent = `${data.summary.agreement_rate}%`;
  batchCsvExport = data.csv_export || '';

  $('batchTriggerCloud').innerHTML = (data.summary.top_triggers || []).map(t =>
    `<span class="token-pill token-pos">${escapeHtml(t.word)} (${t.count})</span>`
  ).join('');

  $('batchTableBody').innerHTML = data.rows.map(r => `
    <tr class="hover:bg-slate-900/45">
      <td class="p-2.5 font-mono text-slate-400">${r.row_id}</td>
      <td class="p-2.5 text-slate-200 max-w-xs truncate" title="${escapeHtml(r.text)}">${escapeHtml(r.text)}</td>
      <td class="p-2.5 font-semibold capitalize" style="color:${r.color}">${r.emoji} ${r.emotion} (${r.confidence}%)</td>
      <td class="p-2.5 capitalize text-slate-300">${r.tfidf_emotion} (${r.tfidf_confidence}%)</td>
      <td class="p-2.5 font-mono text-emerald-300">${escapeHtml(r.top_trigger || '-')}</td>
    </tr>
  `).join('');

  const dist = data.summary.distribution;
  if (batchChart) batchChart.destroy();
  batchChart = new Chart($('batchChart'), {
    type: 'bar',
    data: {
      labels: Object.keys(dist).map(capitalise),
      datasets: [{
        label: 'Entries',
        data: Object.values(dist),
        backgroundColor: Object.keys(dist).map(k => EMOTION_META[k].color),
        borderRadius: 5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function downloadBatchCsv() {
  if (!batchCsvExport) return;
  const blob = new Blob([batchCsvExport], { type: 'text/csv;charset=utf-8;' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'emotisense_batch_results.csv';
  a.click();
}

/* ---------------- Benchmarks & Confusion Matrix Tab ---------------- */
function renderBenchmarksTab() {
  if (!benchmarksData?.benchmarks) return;
  const rows = benchmarksData.benchmarks;
  if (benchModelsChart) benchModelsChart.destroy();
  benchModelsChart = new Chart($('benchModelsChart'), {
    type: 'bar',
    data: {
      labels: rows.map(r => r.model),
      datasets: [
        { label: '5-Fold CV Accuracy (%)', data: rows.map(r => +(r.cv_accuracy * 100).toFixed(2)), backgroundColor: 'rgba(99, 102, 241, 0.8)', borderRadius: 5 },
        { label: 'Held-Out Test Accuracy (%)', data: rows.map(r => +(r.test_accuracy * 100).toFixed(2)), backgroundColor: 'rgba(236, 72, 153, 0.8)', borderRadius: 5 },
        { label: 'Test Macro F1 (%)', data: rows.map(r => +(r.macro_f1 * 100).toFixed(2)), backgroundColor: 'rgba(20, 184, 166, 0.8)', borderRadius: 5 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
        y: { min: 20, max: 100, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
      },
      plugins: { legend: { labels: { color: '#e2e8f0' } } },
    },
  });
  renderMatrixAndReport();
}

function renderMatrixAndReport() {
  if (!benchmarksData?.benchmarks) return;
  const key = $('cmModelSelect').value;
  const b = benchmarksData.benchmarks.find(r => r.key === key) || benchmarksData.benchmarks[0];
  const labels = benchmarksData.labels;
  const cm = b.confusion_matrix;
  const maxVal = Math.max(...cm.flat(), 1);

  let html = '<table class="w-full text-[11px] border-separate border-spacing-1"><thead><tr><th class="p-1 text-slate-500">Act\\Pred</th>';
  labels.forEach(l => { html += `<th class="p-1 capitalize text-slate-300">${l.slice(0, 4)}</th>`; });
  html += '</tr></thead><tbody>';
  cm.forEach((row, rIdx) => {
    html += `<tr><td class="p-1 font-bold capitalize text-slate-300">${labels[rIdx]}</td>`;
    row.forEach((val, cIdx) => {
      const alpha = Math.max(0.12, Math.min(0.85, val / maxVal));
      const bg = rIdx === cIdx ? `rgba(16, 185, 129, ${alpha})` : `rgba(244, 63, 94, ${alpha * 0.7})`;
      html += `<td class="cm-cell text-white" style="background:${bg}">${val}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  $('confusionMatrixGrid').innerHTML = html;

  $('perClassTbody').innerHTML = labels.map(lbl => {
    const pc = b.per_class[lbl];
    const m = EMOTION_META[lbl] || { emoji: '🤖', color: '#6366f1' };
    return `
      <tr>
        <td class="p-3 font-bold capitalize" style="color:${m.color}">${m.emoji} ${lbl}</td>
        <td class="p-3 font-mono">${(pc.precision * 100).toFixed(2)}%</td>
        <td class="p-3 font-mono">${(pc.recall * 100).toFixed(2)}%</td>
        <td class="p-3 font-mono font-bold text-white">${(pc.f1 * 100).toFixed(2)}%</td>
        <td class="p-3 font-mono text-slate-400">${pc.support}</td>
      </tr>
    `;
  }).join('');
}

/* ---------------- 1-on-1 EmotiAgent AI ---------------- */
function toggleAgentDrawer() {
  $('agent-drawer').classList.toggle('closed');
}

function openAgentWithCurrentText() {
  $('agent-drawer').classList.remove('closed');
  $('agent-input').value = 'Explain why this sentence got its emotion prediction, which trigger words drove it, and how I could reframe it.';
  submitAgentChat(new Event('submit'));
}

function sendQuickPrompt(txt) {
  $('agent-input').value = txt;
  submitAgentChat(new Event('submit'));
}

function openAgentSettings() {
  $('agent-settings-modal').classList.remove('hidden');
  $('agent-settings-modal').classList.add('flex');
}

function closeAgentSettings() {
  $('agent-settings-modal').classList.add('hidden');
  $('agent-settings-modal').classList.remove('flex');
}

function loadSavedAgentSettings() {
  $('agent-provider-select').value = sessionStorage.getItem('emotisense_agent_provider') || 'gemini';
  $('agent-apikey-input').value = sessionStorage.getItem('emotisense_agent_apikey') || '';
}

function saveAgentSettings() {
  const p = $('agent-provider-select').value;
  const k = $('agent-apikey-input').value.trim();
  sessionStorage.setItem('emotisense_agent_provider', p);
  if (k) sessionStorage.setItem('emotisense_agent_apikey', k);
  else sessionStorage.removeItem('emotisense_agent_apikey');
  closeAgentSettings();
  checkHealth();
}

function clearAgentKey() {
  sessionStorage.removeItem('emotisense_agent_apikey');
  $('agent-apikey-input').value = '';
  checkHealth();
}

function clearAgentChat() {
  agentHistory = [];
  previousInteractionId = null;
  $('agent-messages').innerHTML = `
    <div class="p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800 text-slate-200 leading-relaxed">
      <p class="font-bold text-pink-300 mb-1">Fresh EmotiAgent Session</p>
      <p>Ask me about token weights, negation effects, tone rewriting, or your mood trends.</p>
    </div>
  `;
}

function appendAgentMsg(role, content, meta = '') {
  const box = $('agent-messages');
  const div = document.createElement('div');
  if (role === 'user') {
    div.className = 'ml-8 p-3 rounded-2xl bg-pink-600/25 border border-pink-500/40 text-white';
    div.textContent = content;
  } else if (role === 'error') {
    div.className = 'mr-4 p-3 rounded-2xl bg-rose-950/60 border border-rose-500/40 text-rose-200 space-y-2';
    div.innerHTML = content;
  } else {
    div.className = 'mr-4 p-3.5 rounded-2xl bg-slate-900/90 border border-slate-800 text-slate-200 agent-msg-prose leading-relaxed';
    div.innerHTML = renderSimpleMarkdown(content) + (meta ? `<div class="mt-2 pt-1.5 border-t border-slate-800 text-[10px] font-mono text-slate-500">${escapeHtml(meta)}</div>` : '');
  }
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

async function submitAgentChat(e) {
  e.preventDefault();
  const input = $('agent-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  appendAgentMsg('user', msg);
  const thinking = appendAgentMsg('assistant', 'Inspecting NLP features and formulating response...');

  const provider = sessionStorage.getItem('emotisense_agent_provider') || 'gemini';
  const apiKey = sessionStorage.getItem('emotisense_agent_apikey') || '';

  const moodCounts = {};
  history.slice(0, 15).forEach(h => { moodCounts[h.emotion] = (moodCounts[h.emotion] || 0) + 1; });

  try {
    const res = await fetch('/api/agent/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey ? { 'X-LLM-API-Key': apiKey } : {}),
      },
      body: JSON.stringify({
        message: msg,
        provider,
        current_text: lastAnalyzedText || $('textInput').value.trim(),
        model: $('modelSelect').value,
        session_summary: { recent_count: history.length, mood_counts: moodCounts },
        history: agentHistory,
        previous_interaction_id: previousInteractionId,
      }),
    });
    const data = await res.json();
    thinking.remove();

    if (!res.ok) {
      const btn = data.setup_required
        ? `<button onclick="openAgentSettings()" class="px-2.5 py-1 rounded-lg bg-pink-600 text-white text-[11px] font-bold">Configure API Key</button>`
        : '';
      appendAgentMsg('error', `<div>${escapeHtml(data.error || 'Agent error.')}</div>${btn}`);
      return;
    }

    if (data.interaction_id) previousInteractionId = data.interaction_id;
    agentHistory.push({ role: 'user', content: msg });
    agentHistory.push({ role: 'assistant', content: data.reply });
    appendAgentMsg('assistant', data.reply, `${data.provider} · ${data.model}`);
  } catch (err) {
    thinking.remove();
    appendAgentMsg('error', `Network error: ${escapeHtml(err.message)}`);
  }
}

/* ---------------- Utilities ---------------- */
function capitalise(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ''; }
function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function renderSimpleMarkdown(md) {
  let safe = escapeHtml(md);
  safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
  const lines = safe.split('\n');
  let out = '';
  let inList = false;
  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith('- ') || t.startsWith('* ')) {
      if (!inList) { out += '<ul>'; inList = true; }
      out += `<li>${t.slice(2)}</li>`;
    } else {
      if (inList) { out += '</ul>'; inList = false; }
      if (t) out += `<p>${t}</p>`;
    }
  }
  if (inList) out += '</ul>';
  return out;
}
