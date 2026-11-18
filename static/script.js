// State
let currentMode = 'bm25';
let topK = 5;
let alpha = 0.50;
let rrfK = 60;
let showScoreDetails = true;
let isSearching = false;

const modeDescriptions = {
  bm25: {
    title: 'BM25 Sparse Retrieval',
    desc: 'Keyword-based lexical matching using the Okapi BM25 algorithm. Excels at exact term matching, medical codes, acronyms, and rare proper nouns.',
    icon: '⚡'
  },
  dense: {
    title: 'Dense Semantic Retrieval',
    desc: 'Dense vector embeddings (all-MiniLM-L6-v2) with Cosine Similarity. Excels at conceptual understanding, synonyms, and paraphrased medical questions.',
    icon: '🧠'
  },
  rrf: {
    title: 'Reciprocal Rank Fusion (RRF)',
    desc: 'Rank-based fusion combining BM25 and Dense orders without score calibration. Formula: <code>1 / (k + rank_bm25) + 1 / (k + rank_dense)</code> with constant k.',
    icon: '🔀'
  },
  weighted: {
    title: 'Weighted Score Combination',
    desc: 'Linear combination of min-max normalized BM25 and Dense similarity scores. Formula: <code>α · Dense_norm + (1 - α) · BM25_norm</code>.',
    icon: '⚖️'
  },
  compare: {
    title: 'Side-by-Side Strategy Comparison',
    desc: 'Simultaneously executes all 4 retrieval strategies (BM25, Dense, RRF, Weighted) side-by-side to compare ranking differences and overlap.',
    icon: '📊'
  }
};

// DOM ready
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadExamples();
  updateModeDisplay();

  const queryInput = document.getElementById('queryInput');
  const clearBtn = document.getElementById('clearBtn');

  queryInput.addEventListener('input', () => {
    clearBtn.style.display = queryInput.value.trim() ? 'block' : 'none';
  });
});

async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) throw new Error('Failed to load stats');
    const data = await res.json();
    document.getElementById('statsText').innerHTML = 
      `<strong>${data.document}</strong> • ${data.total_chunks} Chunks • ${data.total_pages} Pages • <code>${data.embedding_model}</code>`;
  } catch (err) {
    document.getElementById('statsText').textContent = 'Index connected';
  }
}

async function loadExamples() {
  try {
    const res = await fetch('/api/examples');
    const examples = await res.json();
    const container = document.getElementById('exampleChips');
    container.innerHTML = '';

    examples.forEach(ex => {
      const chip = document.createElement('div');
      chip.className = 'example-chip';
      chip.textContent = ex.query;
      chip.title = `Target: ${ex.target}`;
      chip.onclick = () => {
        document.getElementById('queryInput').value = ex.query;
        document.getElementById('clearBtn').style.display = 'block';
        handleSearch();
      };
      container.appendChild(chip);
    });
  } catch (err) {
    console.error('Failed to load example queries:', err);
  }
}

function clearSearch() {
  const queryInput = document.getElementById('queryInput');
  queryInput.value = '';
  document.getElementById('clearBtn').style.display = 'none';
  queryInput.focus();
}

function selectMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-mode') === mode);
  });

  updateModeDisplay();

  // If there's an existing query, re-run search immediately
  const q = document.getElementById('queryInput').value.trim();
  if (q) {
    handleSearch();
  }
}

function updateModeDisplay() {
  const info = modeDescriptions[currentMode];
  document.getElementById('modeInfo').innerHTML = 
    `<strong>${info.icon} ${info.title}:</strong> ${info.desc}`;

  const alphaBox = document.getElementById('alphaControl');
  const rrfBox = document.getElementById('rrfControl');

  if (currentMode === 'weighted' || currentMode === 'compare') {
    alphaBox.style.display = 'block';
  } else {
    alphaBox.style.display = 'none';
  }

  if (currentMode === 'rrf' || currentMode === 'compare') {
    rrfBox.style.display = 'block';
  } else {
    rrfBox.style.display = 'none';
  }
}

function updateTopK(val) {
  topK = parseInt(val, 10);
  document.getElementById('topKVal').textContent = topK;
  const q = document.getElementById('queryInput').value.trim();
  if (q) handleSearch();
}

function updateAlpha(val) {
  alpha = parseFloat(val);
  document.getElementById('alphaVal').textContent = alpha.toFixed(2);
  const q = document.getElementById('queryInput').value.trim();
  if (q && (currentMode === 'weighted' || currentMode === 'compare')) {
    handleSearch();
  }
}

function updateRRFK(val) {
  rrfK = parseInt(val, 10);
  document.getElementById('rrfKVal').textContent = rrfK;
  const q = document.getElementById('queryInput').value.trim();
  if (q && (currentMode === 'rrf' || currentMode === 'compare')) {
    handleSearch();
  }
}

function toggleDetails() {
  showScoreDetails = !showScoreDetails;
  document.querySelectorAll('.breakdown-row').forEach(row => {
    row.style.display = showScoreDetails ? 'flex' : 'none';
  });
}

async function handleSearch() {
  const queryInput = document.getElementById('queryInput');
  const query = queryInput.value.trim();
  if (!query || isSearching) return;

  isSearching = true;
  setLoading(true);

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        mode: currentMode,
        top_k: topK,
        alpha: alpha,
        rrf_k: rrfK,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Search request failed');
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    isSearching = false;
    setLoading(false);
  }
}

function setLoading(isLoading) {
  const btnText = document.getElementById('btnText');
  const btnSpinner = document.getElementById('btnSpinner');
  const searchBtn = document.getElementById('searchBtn');

  if (isLoading) {
    btnText.classList.add('hidden');
    btnSpinner.classList.remove('hidden');
    searchBtn.disabled = true;
  } else {
    btnText.classList.remove('hidden');
    btnSpinner.classList.add('hidden');
    searchBtn.disabled = false;
  }
}

function renderResults(data) {
  const emptyState = document.getElementById('emptyState');
  const resultsHeader = document.getElementById('resultsHeader');
  const resultsContainer = document.getElementById('resultsContainer');
  const compareContainer = document.getElementById('compareContainer');

  emptyState.style.display = 'none';
  resultsHeader.style.display = 'flex';

  document.getElementById('resultsLatency').textContent = `⚡ ${data.latency_ms} ms`;
  document.getElementById('resultsMode').textContent = 
    data.mode === 'compare' ? 'Comparison Mode' : `${data.mode.toUpperCase()} Retrieval`;

  if (data.mode === 'compare') {
    resultsContainer.style.display = 'none';
    compareContainer.style.display = 'grid';
    document.getElementById('resultsCount').textContent = `Top ${data.top_k} across 4 strategies`;
    renderCompareGrid(data.comparison, data.top_k);
  } else {
    compareContainer.style.display = 'none';
    resultsContainer.style.display = 'flex';
    document.getElementById('resultsCount').textContent = `${data.total_results} results retrieved`;
    renderSingleModeList(data.results, data.mode);
  }
}

function getRankClass(rank) {
  if (rank === 1) return 'rank-1';
  if (rank === 2) return 'rank-2';
  if (rank === 3) return 'rank-3';
  return 'rank-other';
}

function renderSingleModeList(results, mode) {
  const container = document.getElementById('resultsContainer');
  container.innerHTML = '';

  if (!results || results.length === 0) {
    container.innerHTML = `<div class="empty-state"><h3>No results found</h3><p>Try different keywords or switch retrieval modes.</p></div>`;
    return;
  }

  results.forEach(item => {
    const card = document.createElement('div');
    card.className = 'result-card';

    let scoreBadgeHtml = '';
    let breakdownHtml = '';

    if (mode === 'bm25') {
      scoreBadgeHtml = `<span class="score-badge-main badge-bm25">BM25 Score: ${item.score.toFixed(4)}</span>`;
    } else if (mode === 'dense') {
      scoreBadgeHtml = `<span class="score-badge-main badge-dense">Cosine Similarity: ${item.score.toFixed(4)}</span>`;
    } else if (mode === 'rrf') {
      scoreBadgeHtml = `<span class="score-badge-main badge-rrf">RRF Score: ${item.score.toFixed(5)}</span>`;
      breakdownHtml = `
        <div class="breakdown-row" style="display: ${showScoreDetails ? 'flex' : 'none'};">
          <span class="breakdown-pill">BM25 Rank: #${item.bm25_rank} (Score: ${item.bm25_score})</span>
          <span class="breakdown-pill">Dense Rank: #${item.dense_rank} (Sim: ${item.dense_score})</span>
          <span class="breakdown-pill">Constant k = ${item.rrf_k}</span>
        </div>
      `;
    } else if (mode === 'weighted') {
      scoreBadgeHtml = `<span class="score-badge-main badge-weighted">Weighted Score: ${item.score.toFixed(4)}</span>`;
      breakdownHtml = `
        <div class="breakdown-row" style="display: ${showScoreDetails ? 'flex' : 'none'};">
          <span class="breakdown-pill">Dense: ${item.dense_score} (Norm: ${item.dense_norm})</span>
          <span class="breakdown-pill">BM25: ${item.bm25_score} (Norm: ${item.bm25_norm})</span>
          <span class="breakdown-pill">Weight α = ${item.alpha}</span>
        </div>
      `;
    }

    card.innerHTML = `
      <div class="card-top">
        <div class="meta-badges">
          <span class="rank-badge ${getRankClass(item.rank)}">#${item.rank}</span>
          <span class="page-badge">📄 Page ${item.page}</span>
          <span class="page-badge">🏷️ ${item.chunk_id}</span>
          <span class="section-badge" title="${item.section || ''}">📁 ${item.section || 'General'}</span>
        </div>
        ${scoreBadgeHtml}
      </div>
      <div class="chunk-text">${escapeHtml(item.content)}</div>
      ${breakdownHtml}
    `;

    container.appendChild(card);
  });
}

function renderCompareGrid(comparison, topK) {
  const container = document.getElementById('compareContainer');
  container.innerHTML = '';

  const columns = [
    { key: 'bm25', title: 'BM25 (Sparse)', headerClass: 'col-header-bm25', scoreKey: 'bm25_score', label: 'BM25' },
    { key: 'dense', title: 'Dense (Semantic)', headerClass: 'col-header-dense', scoreKey: 'dense_score', label: 'Cosine' },
    { key: 'rrf', title: 'Hybrid: RRF', headerClass: 'col-header-rrf', scoreKey: 'rrf_score', label: 'RRF' },
    { key: 'weighted', title: `Hybrid: Weighted (α=${alpha})`, headerClass: 'col-header-weighted', scoreKey: 'weighted_score', label: 'Weight' },
  ];

  columns.forEach(col => {
    const list = comparison[col.key] || [];
    const colDiv = document.createElement('div');
    colDiv.className = 'compare-column';

    colDiv.innerHTML = `
      <div class="compare-col-header ${col.headerClass}">
        <span>${col.title}</span>
        <span>${list.length} hits</span>
      </div>
      <div class="compare-col-body" id="col-${col.key}"></div>
    `;

    const bodyDiv = colDiv.querySelector(`#col-${col.key}`);
    list.forEach(item => {
      const card = document.createElement('div');
      card.className = 'mini-card';
      const scoreVal = item.score !== undefined ? item.score : item[col.scoreKey];
      card.innerHTML = `
        <div class="mini-card-header">
          <span class="rank-badge ${getRankClass(item.rank)}" style="font-size:0.75rem; padding: 2px 6px;">#${item.rank}</span>
          <span class="page-badge" style="font-size:0.72rem;">Pg ${item.page}</span>
          <span class="mini-score">${col.label}: ${typeof scoreVal === 'number' ? scoreVal.toFixed(4) : scoreVal}</span>
        </div>
        <div class="mini-content" title="${escapeHtml(item.content)}">${escapeHtml(item.content)}</div>
      `;
      bodyDiv.appendChild(card);
    });

    container.appendChild(colDiv);
  });
}

function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
