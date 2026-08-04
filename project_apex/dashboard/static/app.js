/**
 * Project APEX — Dashboard Client Logic
 */

const API_BASE = 'http://127.0.0.1:8080/api';

// Formatters
const fmtCurrency = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
});

const fmtPct = new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
});

// Update UTC Clock
function updateClock() {
    const now = new Date();
    document.getElementById('current-time').textContent = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
}
setInterval(updateClock, 1000);

// Fetch Portfolio Data
async function fetchPortfolio() {
    try {
        const res = await fetch(`${API_BASE}/portfolio`);
        const data = await res.json();
        
        if (data.error) return;

        document.getElementById('equity-value').textContent = fmtCurrency.format(data.equity);
        
        const returnPct = data.total_return_pct / 100;
        const eqChangeEl = document.getElementById('equity-change');
        eqChangeEl.textContent = `${returnPct >= 0 ? '+' : ''}${fmtPct.format(returnPct)}`;
        eqChangeEl.className = `metric-change ${returnPct >= 0 ? 'positive' : 'negative'}`;

        document.getElementById('daily-pnl').textContent = fmtCurrency.format(data.total_realized_pnl || 0); // Simplified for demo
        
        const dailyPct = data.daily_pnl_pct / 100;
        const dailyChangeEl = document.getElementById('daily-pnl-pct');
        dailyChangeEl.textContent = `${dailyPct >= 0 ? '+' : ''}${fmtPct.format(dailyPct)}`;
        dailyChangeEl.className = `metric-change ${dailyPct > 0 ? 'positive' : (dailyPct < 0 ? 'negative' : 'neutral')}`;

        document.getElementById('drawdown-value').textContent = fmtPct.format(data.drawdown_pct / 100);
        document.getElementById('peak-equity').textContent = fmtCurrency.format(data.peak_equity);
        
        document.getElementById('win-rate').textContent = `${data.win_rate_pct}%`;
        document.getElementById('total-trades').textContent = data.closed_trades;

    } catch (e) {
        console.error('Failed to fetch portfolio', e);
    }
}

// Fetch Active Positions
async function fetchPositions() {
    try {
        const res = await fetch(`${API_BASE}/positions`);
        const data = await res.json();
        
        if (data.error) return;

        const tbody = document.getElementById('positions-body');
        
        if (!data.positions || data.positions.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" class="empty-state">No open positions</td></tr>`;
            return;
        }

        tbody.innerHTML = data.positions.map(p => {
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
                        (${fmtPct.format(p.unrealized_pnl_pct)})
                    </td>
                </tr>
            `;
        }).join('');

    } catch (e) {
        console.error('Failed to fetch positions', e);
    }
}

// Mocked ML & Regime updates (Waiting for WS integration in future)
function updateAIStatus() {
    // In reality, this would come from a websocket stream
    // Simulating ML prediction flip for UI demonstration
    const prob = 0.4 + (Math.random() * 0.4); // Random between 40-80%
    const probPct = `${(prob * 100).toFixed(1)}%`;
    
    document.getElementById('prob-value').textContent = probPct;
    const fill = document.getElementById('prob-up-fill');
    fill.style.width = probPct;
    
    // Change fill color based on prob
    if (prob > 0.6) {
        fill.style.background = 'var(--success)';
    } else if (prob < 0.4) {
        fill.style.background = 'var(--danger)';
    } else {
        fill.style.background = 'var(--warning)';
    }
    
    // Simulate Regime detection
    const regimes = ['STRONG_TREND_UP', 'RANGING', 'HIGH_VOLATILITY'];
    const currentRegime = document.getElementById('regime-badge').textContent;
    if (currentRegime === 'DETECTING...' || Math.random() > 0.9) {
        const newRegime = regimes[Math.floor(Math.random() * regimes.length)];
        document.getElementById('regime-badge').textContent = newRegime;
        if (newRegime.includes('TREND')) {
            document.getElementById('regime-badge').style.color = '#34D399'; // Greenish
            document.getElementById('regime-badge').style.borderColor = 'rgba(52, 211, 153, 0.4)';
            document.getElementById('regime-badge').style.background = 'rgba(52, 211, 153, 0.1)';
        } else if (newRegime === 'HIGH_VOLATILITY') {
            document.getElementById('regime-badge').style.color = '#F87171'; // Reddish
            document.getElementById('regime-badge').style.borderColor = 'rgba(248, 113, 113, 0.4)';
            document.getElementById('regime-badge').style.background = 'rgba(248, 113, 113, 0.1)';
        } else {
            document.getElementById('regime-badge').style.color = '#FBBF24'; // Yellowish
            document.getElementById('regime-badge').style.borderColor = 'rgba(251, 191, 36, 0.4)';
            document.getElementById('regime-badge').style.background = 'rgba(251, 191, 36, 0.1)';
        }
    }
}

// Polling Loop
async function pollData() {
    await fetchPortfolio();
    await fetchPositions();
    updateAIStatus();
}

// Initial Load
updateClock();
pollData();
setInterval(pollData, 5000);
