/**
 * Project APEX — Dashboard Client Logic
 */

const API_BASE = 'http://127.0.0.1:8080/api';

// ─── State ────────────────────────────────────────────────────────────────────
let currentAccount = 'demo';
let currentStrategy = 'standard';
let isScanning = false;
let currentPage = 'dashboard';

// ─── Formatters ───────────────────────────────────────────────────────────────
const fmtCurrency = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });
const fmtPct = new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtDate = (epoch) => {
    const d = new Date(epoch);
    return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`;
};

// ─── Toast Notifications ──────────────────────────────────────────────────────
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ─── Clock ────────────────────────────────────────────────────────────────────
function updateClock() {
    const now = new Date();
    document.getElementById('current-time').textContent =
        now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
}
setInterval(updateClock, 1000);
updateClock();

// ─── Navigation ───────────────────────────────────────────────────────────────
function navigateTo(pageId) {
    if (currentPage === pageId) return;
    
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.getElementById(`nav-${pageId}`).classList.add('active');
    
    // Update pages
    document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
    document.getElementById(`page-${pageId}`).classList.add('active');
    
    // Update title
    const titles = {
        'dashboard': 'Dashboard',
        'ai': 'AI Engine Analysis',
        'positions': 'Live Positions Tracker',
        'history': 'Trade History & Performance'
    };
    document.getElementById('page-title').textContent = titles[pageId];
    
    currentPage = pageId;
    
    // Immediate fetch if switching to history
    if (pageId === 'history') fetchTradeHistory();
}

// ─── Account & Balance ────────────────────────────────────────────────────────
async function fetchAccountBalance(type) {
    try {
        const res = await fetch(`${API_BASE}/account-balance?type=${type}`);
        const data = await res.json();
        const balEl = document.getElementById('account-balance');
        const subEl = document.getElementById('account-type-label');
        if (data.balance !== null && data.balance !== undefined) {
            const sym = data.currency === 'USD' ? '$' : data.currency + ' ';
            balEl.textContent = sym + parseFloat(data.balance).toLocaleString('en-US', {
                minimumFractionDigits: 2, maximumFractionDigits: 2
            });
            subEl.textContent = type === 'real' ? '⚠ Real Account — Real Funds' : 'Demo Account';
            subEl.style.color = type === 'real' ? 'var(--danger)' : '';
        } else {
            balEl.textContent = '—';
            subEl.textContent = type === 'real' ? 'Real (unavailable)' : 'Demo (unavailable)';
        }
    } catch (e) { console.error('Balance fetch failed', e); }
}

async function selectAccount(type) {
    document.getElementById('btn-demo').classList.toggle('active', type === 'demo');
    document.getElementById('btn-real').classList.toggle('active', type === 'real');
    const balEl = document.getElementById('account-balance');
    const subEl = document.getElementById('account-type-label');
    const prevBalance = balEl.textContent;
    balEl.textContent = 'Switching...';
    subEl.textContent = 'Reconnecting to account...';

    try {
        const res = await fetch(`${API_BASE}/settings/account`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type }),
        });
        const data = await res.json();
        if (!data.success && !data.message) {
            showToast('Account switch failed', 'error');
            balEl.textContent = prevBalance;
            return;
        }
        showToast(`Switched to ${type.toUpperCase()} account`);
    } catch (e) {
        showToast('Connection error', 'error');
        balEl.textContent = prevBalance;
        return;
    }

    currentAccount = type;
    await fetchAccountBalance(type);
}

// ─── Strategy Settings & UI ───────────────────────────────────────────────────
function toggleSwitch(el) {
    el.classList.toggle('active');
    const label = el.nextElementSibling;
    if (el.classList.contains('active')) {
        label.textContent = 'ON';
        label.classList.add('on');
    } else {
        label.textContent = 'OFF';
        label.classList.remove('on');
    }
}

async function loadSettings() {
    try {
        const res = await fetch(`${API_BASE}/settings/trading-params`);
        const data = await res.json();
        
        // Standard
        if (data.standard) {
            document.getElementById('std_risk_per_trade_pct').value = data.standard.risk_per_trade_pct;
            document.getElementById('std_atr_stop_multiplier').value = data.standard.atr_stop_multiplier;
            document.getElementById('std_atr_tp_multiplier').value = data.standard.atr_tp_multiplier;
            document.getElementById('std_min_confidence').value = data.standard.min_confidence;
        }
        // Scalping
        if (data.scalping) {
            document.getElementById('scl_risk_per_trade_pct').value = data.scalping.risk_per_trade_pct;
            document.getElementById('scl_atr_stop_multiplier').value = data.scalping.atr_stop_multiplier;
            document.getElementById('scl_atr_tp_multiplier').value = data.scalping.atr_tp_multiplier;
            document.getElementById('scl_max_open_positions').value = data.scalping.max_open_positions;
        }
        // Differs
        if (data.differs) {
            document.getElementById('dif_base_stake').value = data.differs.base_stake;
            document.getElementById('dif_max_stake').value = data.differs.max_stake;
            document.getElementById('dif_loss_cooldown_ticks').value = data.differs.loss_cooldown_ticks;
            
            const toggle = document.getElementById('dif_martingale_enabled');
            const toggleLabel = document.getElementById('dif_martingale_label');
            if (data.differs.martingale_enabled) {
                toggle.classList.add('active');
                toggleLabel.textContent = 'ON';
                toggleLabel.classList.add('on');
            } else {
                toggle.classList.remove('active');
                toggleLabel.textContent = 'OFF';
                toggleLabel.classList.remove('on');
            }
            
            document.getElementById('dif_martingale_multiplier').value = data.differs.martingale_multiplier;
            document.getElementById('dif_martingale_max_steps').value = data.differs.martingale_max_steps;
        }
    } catch (e) { console.error('Failed to load settings', e); }
}

async function saveSettings(mode) {
    const params = {};
    const btn = event.target;
    
    if (mode === 'standard') {
        params.risk_per_trade_pct = parseFloat(document.getElementById('std_risk_per_trade_pct').value);
        params.atr_stop_multiplier = parseFloat(document.getElementById('std_atr_stop_multiplier').value);
        params.atr_tp_multiplier = parseFloat(document.getElementById('std_atr_tp_multiplier').value);
        params.min_confidence = parseFloat(document.getElementById('std_min_confidence').value);
    } else if (mode === 'scalping') {
        params.risk_per_trade_pct = parseFloat(document.getElementById('scl_risk_per_trade_pct').value);
        params.atr_stop_multiplier = parseFloat(document.getElementById('scl_atr_stop_multiplier').value);
        params.atr_tp_multiplier = parseFloat(document.getElementById('scl_atr_tp_multiplier').value);
        params.max_open_positions = parseInt(document.getElementById('scl_max_open_positions').value);
    } else if (mode === 'differs') {
        params.base_stake = parseFloat(document.getElementById('dif_base_stake').value);
        params.max_stake = parseFloat(document.getElementById('dif_max_stake').value);
        params.loss_cooldown_ticks = parseInt(document.getElementById('dif_loss_cooldown_ticks').value);
        params.martingale_enabled = document.getElementById('dif_martingale_enabled').classList.contains('active');
        params.martingale_multiplier = parseFloat(document.getElementById('dif_martingale_multiplier').value);
        params.martingale_max_steps = parseInt(document.getElementById('dif_martingale_max_steps').value);
    }
    
    try {
        const res = await fetch(`${API_BASE}/settings/trading-params`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, params })
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            btn.classList.add('saved');
            btn.textContent = 'Saved ✓';
            showToast(`${mode.toUpperCase()} settings applied`);
            setTimeout(() => {
                btn.classList.remove('saved');
                btn.textContent = 'Apply Changes';
            }, 2000);
        } else {
            showToast(data.error || 'Failed to save', 'error');
        }
    } catch (e) {
        showToast('Connection error', 'error');
    }
}

function selectStrategy(strategy) {
    currentStrategy = strategy;

    const stdCard  = document.getElementById('sc-standard');
    const scalpCard = document.getElementById('sc-scalping');
    const differsCard = document.getElementById('sc-differs');
    const btn = document.getElementById('scan-trade-btn');
    const tag = document.getElementById('active-strategy-tag');
    const pill = document.getElementById('engine-pill');
    const pillIcon = document.getElementById('engine-pill-icon');
    const pillName = document.getElementById('engine-pill-name');
    
    // Config Drawer
    const drawer = document.getElementById('config-drawer');
    document.querySelectorAll('.config-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`config-panel-${strategy}`).classList.add('active');
    
    // If not open, open it
    if (!drawer.classList.contains('open')) {
        drawer.classList.add('open');
    }

    stdCard.classList.remove('active');
    scalpCard.classList.remove('active', 'scalping');
    if(differsCard) differsCard.classList.remove('active', 'differs');

    if (strategy === 'scalping') {
        scalpCard.classList.add('active', 'scalping');
        btn.classList.add('scalping-mode');
        btn.classList.remove('scanning', 'differs-mode');
        tag.innerHTML = '<span class="ast-dot" style="background:#F97316;box-shadow:0 0 8px #F97316;"></span> Scalping Mode Active';
        pill.className = 'engine-pill scalping';
        pillIcon.textContent = '⚡';
        pillName.textContent = 'Scalping — VWAP + EMA';
    } else if (strategy === 'differs') {
        if(differsCard) differsCard.classList.add('active', 'differs');
        btn.classList.add('differs-mode');
        btn.classList.remove('scanning', 'scalping-mode');
        tag.innerHTML = '<span class="ast-dot" style="background:#A855F7;box-shadow:0 0 8px #A855F7;"></span> Differs Mode Active';
        pill.className = 'engine-pill differs';
        pillIcon.textContent = '🎯';
        pillName.textContent = 'Differs — Digit Predictor';
    } else {
        stdCard.classList.add('active');
        btn.classList.remove('scalping-mode', 'differs-mode');
        tag.innerHTML = '<span class="ast-dot"></span> Standard Mode Active';
        pill.className = 'engine-pill';
        pillIcon.textContent = '🤖';
        pillName.textContent = 'Standard — Ensemble + ML';
    }
}

// ─── Scan & Trade ─────────────────────────────────────────────────────────────
async function scanAndTrade() {
    const btn = document.getElementById('scan-trade-btn');
    const btnText = document.getElementById('scan-btn-text');
    if (isScanning) return;

    try {
        const res = await fetch(`${API_BASE}/settings/mode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: currentStrategy }),
        });
        const data = await res.json();
        if (data.status !== 'success') {
            showToast(data.error, 'error');
            return;
        }
    } catch (e) {
        showToast('Connection error', 'error');
        return;
    }

    isScanning = true;
    btn.classList.add('scanning');
    btnText.textContent = 'SCANNING MARKETS...';
    
    // Close the config drawer when starting scan to clear up space
    document.getElementById('config-drawer').classList.remove('open');

    setTimeout(() => {
        btn.classList.remove('scanning');
        btnText.textContent = '✓ TRADING ACTIVE';
        btn.style.background = 'linear-gradient(135deg, #10B981, #059669)';
        btn.style.boxShadow = '0 4px 20px rgba(16, 185, 129, 0.4)';
    }, 3000);
}

// ─── Data Fetching ────────────────────────────────────────────────────────────
async function fetchPortfolio() {
    try {
        const res = await fetch(`${API_BASE}/portfolio`);
        const data = await res.json();
        if (data.error) return;

        const sessionPnl = data.total_realized_pnl || 0;
        const sessionEl = document.getElementById('equity-value');
        sessionEl.textContent = `${sessionPnl >= 0 ? '+' : ''}${fmtCurrency.format(sessionPnl)}`;
        sessionEl.className = `metric-value ${sessionPnl > 0 ? 'positive-glow' : sessionPnl < 0 ? 'negative-glow' : 'highlight'}`;

        const returnPct = data.total_return_pct / 100;
        const eqEl = document.getElementById('equity-change');
        eqEl.textContent = `${returnPct >= 0 ? '+' : ''}${fmtPct.format(returnPct)} return`;
        eqEl.className = `metric-change ${returnPct >= 0 ? 'positive' : 'negative'}`;

        document.getElementById('daily-pnl').textContent = fmtCurrency.format(data.total_realized_pnl || 0);
        const dailyPct = data.daily_pnl_pct / 100;
        const dpEl = document.getElementById('daily-pnl-pct');
        dpEl.textContent = `${dailyPct >= 0 ? '+' : ''}${fmtPct.format(dailyPct)}`;
        dpEl.className = `metric-change ${dailyPct > 0 ? 'positive' : dailyPct < 0 ? 'negative' : 'neutral'}`;

        document.getElementById('drawdown-value').textContent = fmtPct.format(data.drawdown_pct / 100);
        document.getElementById('peak-equity').textContent = fmtCurrency.format(data.peak_equity);
        document.getElementById('win-rate').textContent = `${data.win_rate_pct}%`;
        document.getElementById('total-trades').textContent = data.closed_trades;
    } catch (e) {}
}

function getStrategyTag(strategyName) {
    const name = (strategyName || 'Unknown').toLowerCase();
    if (name.includes('differ')) return `<span class="strategy-tag differs">${strategyName}</span>`;
    if (name.includes('scalp')) return `<span class="strategy-tag scalping">${strategyName}</span>`;
    return `<span class="strategy-tag standard">${strategyName}</span>`;
}

async function fetchPositions() {
    try {
        const res = await fetch(`${API_BASE}/positions`);
        const data = await res.json();
        if (data.error) return;

        const dashTbody = document.getElementById('dash-positions-body');
        const fullTbody = document.getElementById('full-positions-body');
        
        if (!data.positions || data.positions.length === 0) {
            dashTbody.innerHTML = `<tr><td colspan="6" class="empty-state">No open positions</td></tr>`;
            fullTbody.innerHTML = `<tr><td colspan="10" class="empty-state">No open positions</td></tr>`;
            return;
        }

        // Render Dashboard Table (Simplified)
        dashTbody.innerHTML = data.positions.map(p => {
            const isLong = p.direction === 'LONG';
            const isProfit = p.unrealized_pnl >= 0;
            return `
                <tr>
                    <td><strong>${p.symbol}</strong></td>
                    <td class="${isLong ? 'dir-long' : 'dir-short'}">${p.direction}</td>
                    <td>${p.size.toFixed(4)}</td>
                    <td>${p.entry_price.toFixed(4)}</td>
                    <td>${p.current_price.toFixed(4)}</td>
                    <td class="${isProfit ? 'pnl-positive' : 'pnl-negative'}">
                        ${p.unrealized_pnl >= 0 ? '+' : ''}${fmtCurrency.format(p.unrealized_pnl)}
                    </td>
                </tr>`;
        }).join('');

        // Render Full Positions Table (Detailed)
        fullTbody.innerHTML = data.positions.map(p => {
            const isLong = p.direction === 'LONG';
            const isProfit = p.unrealized_pnl >= 0;
            return `
                <tr>
                    <td><span style="color:var(--text-muted)">${p.id.substring(0, 8)}</span></td>
                    <td><strong>${p.symbol}</strong></td>
                    <td>${getStrategyTag(p.strategy)}</td>
                    <td class="${isLong ? 'dir-long' : 'dir-short'}">${p.direction}</td>
                    <td>${p.size.toFixed(4)}</td>
                    <td>${p.entry_price.toFixed(4)}</td>
                    <td>${p.current_price.toFixed(4)}</td>
                    <td>${p.stop_loss ? p.stop_loss.toFixed(4) : '—'}</td>
                    <td>${p.take_profit ? p.take_profit.toFixed(4) : '—'}</td>
                    <td class="${isProfit ? 'pnl-positive' : 'pnl-negative'}">
                        ${p.unrealized_pnl >= 0 ? '+' : ''}${fmtCurrency.format(p.unrealized_pnl)}
                        (${fmtPct.format(p.unrealized_pnl_pct / 100)})
                    </td>
                </tr>`;
        }).join('');

    } catch (e) {}
}

async function fetchTradeHistory() {
    if (currentPage !== 'history') return;
    
    try {
        const res = await fetch(`${API_BASE}/trades?n=100`);
        const data = await res.json();
        if (data.error) return;

        const tbody = document.getElementById('history-body');
        
        if (!data.trades || data.trades.length === 0) {
            tbody.innerHTML = `<tr><td colspan="11" class="empty-state">No trade history available</td></tr>`;
            return;
        }

        let wins = 0;
        let netPnl = 0;
        let bestTrade = 0;
        let worstTrade = 0;

        tbody.innerHTML = data.trades.map(t => {
            const isWon = t.pnl > 0;
            const isLong = t.direction === 'LONG';
            
            // Stats
            if (isWon) wins++;
            netPnl += t.pnl;
            if (t.pnl > bestTrade) bestTrade = t.pnl;
            if (t.pnl < worstTrade) worstTrade = t.pnl;

            return `
                <tr class="${isWon ? 'won-row' : 'lost-row'}">
                    <td style="color:var(--text-muted)">${fmtDate(t.closed_at)}</td>
                    <td><strong>${t.symbol}</strong></td>
                    <td>${getStrategyTag(t.strategy)}</td>
                    <td class="${isLong ? 'dir-long' : 'dir-short'}">${t.direction}</td>
                    <td>${t.size.toFixed(4)}</td>
                    <td>${t.entry_price.toFixed(4)}</td>
                    <td>${t.exit_price.toFixed(4)}</td>
                    <td class="${isWon ? 'pnl-positive' : 'pnl-negative'}">
                        ${isWon ? '+' : ''}${fmtCurrency.format(t.pnl)}
                    </td>
                    <td class="${isWon ? 'pnl-positive' : 'pnl-negative'}">
                        ${isWon ? '+' : ''}${fmtPct.format(t.pnl_pct / 100)}
                    </td>
                    <td><span class="result-badge ${isWon ? 'won' : 'lost'}">${isWon ? 'WON' : 'LOST'}</span></td>
                    <td><span style="font-size:10px; color:var(--text-muted); text-transform:uppercase;">${t.close_reason}</span></td>
                </tr>`;
        }).join('');

        // Update summary stats
        document.getElementById('hist-total-trades').textContent = data.trades.length;
        document.getElementById('hist-win-rate').textContent = `${((wins / data.trades.length) * 100).toFixed(1)}%`;
        
        const netEl = document.getElementById('hist-net-pnl');
        netEl.textContent = `${netPnl >= 0 ? '+' : ''}${fmtCurrency.format(netPnl)}`;
        netEl.style.color = netPnl >= 0 ? 'var(--success)' : 'var(--danger)';
        
        document.getElementById('hist-best-trade').textContent = `+${fmtCurrency.format(bestTrade)}`;
        document.getElementById('hist-worst-trade').textContent = fmtCurrency.format(worstTrade);

    } catch (e) {}
}

async function fetchStatus() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();

        // Sync mode if changed remotely
        const backendMode = data.trading_mode || 'standard';
        if (backendMode !== currentStrategy && !isScanning) {
            selectStrategy(backendMode);
            document.getElementById('config-drawer').classList.remove('open');
        }

        // Active symbols
        const symList = document.getElementById('active-symbols-list');
        if (data.active_symbols && data.active_symbols.length > 0) {
            symList.textContent = data.active_symbols.join(', ');
        } else {
            symList.textContent = '—';
        }

        // Connection
        const connEl = document.getElementById('connection-status');
        connEl.textContent = data.broker_connected ? 'System Online' : 'Disconnected';
        document.querySelector('.pulse-dot').style.background = data.broker_connected ? 'var(--success)' : 'var(--danger)';
        document.querySelector('.pulse-dot').style.boxShadow = data.broker_connected ? '0 0 8px var(--success)' : 'none';

        // Account type
        const acctType = data.account_type || 'demo';
        const badge = document.getElementById('trading-mode-badge');
        badge.textContent = acctType.toUpperCase();
        badge.className = `mode-badge ${acctType === 'real' ? 'real-mode' : 'demo-mode'}`;
        
        document.getElementById('btn-demo').classList.toggle('active', acctType === 'demo');
        document.getElementById('btn-real').classList.toggle('active', acctType === 'real');
        
        if (acctType !== currentAccount) currentAccount = acctType;

    } catch (e) {}
}

function updateAIStatus() {
    const prob = 0.4 + (Math.random() * 0.4);
    const probPct = `${(prob * 100).toFixed(1)}%`;
    document.getElementById('prob-value').textContent = probPct;
    
    const fill = document.getElementById('prob-up-fill');
    fill.style.width = probPct;
    fill.style.background = prob > 0.6
        ? 'var(--success)' : prob < 0.4
        ? 'var(--danger)' : 'var(--warning)';

    const regimes = ['STRONG_TREND_UP', 'RANGING', 'HIGH_VOLATILITY', 'TREND_DOWN'];
    const badgeDash = document.getElementById('dash-regime-badge');
    const badgeAI = document.getElementById('ai-regime-badge');
    
    if (badgeDash.textContent === 'DETECTING...' || Math.random() > 0.92) {
        const r = regimes[Math.floor(Math.random() * regimes.length)];
        badgeDash.textContent = r;
        badgeAI.textContent = r;
        
        let css = '';
        if (r.includes('UP')) css = 'color:#34D399;border-color:rgba(52,211,153,0.4);background:rgba(52,211,153,0.1)';
        else if (r.includes('DOWN')) css = 'color:#F87171;border-color:rgba(248,113,113,0.4);background:rgba(248,113,113,0.1)';
        else if (r === 'HIGH_VOLATILITY') css = 'color:#F97316;border-color:rgba(249,115,22,0.4);background:rgba(249,115,22,0.1)';
        else css = 'color:#FBBF24;border-color:rgba(251,191,36,0.4);background:rgba(251,191,36,0.1)';
        
        badgeDash.style.cssText = css;
        badgeAI.style.cssText = css;
    }
}

// ─── Polling Loop ─────────────────────────────────────────────────────────────
async function pollData() {
    await Promise.all([fetchPortfolio(), fetchPositions(), fetchStatus()]);
    updateAIStatus();
    if (currentPage === 'history') fetchTradeHistory();
}

// ─── Init ─────────────────────────────────────────────────────────────────────
loadSettings();
fetchAccountBalance('demo');
pollData();

setInterval(pollData, 5000);
setInterval(() => fetchAccountBalance(currentAccount), 10000);
