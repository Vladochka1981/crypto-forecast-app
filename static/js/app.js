// static/js/app.js
const HORIZONS_HOURS = { 1: 4, 3: 12, 6: 24, 12: 48, 18: 72 };

let chart = null;
let candleSeries = null;
let macdChart = null;
let macdLineSeries = null;
let macdSignalSeries = null;
let macdHistSeries = null;
let volumeChart = null;
let volumeSeries = null;

const forecastSeries = {
    gru:    { line: null, lower: null, upper: null },
    arimax: { line: null, lower: null, upper: null },
};

function toTimestamp(dateStr) {
    return Math.floor(new Date(dateStr).getTime() / 1000);
}

function initChart() {
    const el = document.getElementById('chart');
    if (!el) return;
    chart = LightweightCharts.createChart(el, {
        width: el.clientWidth,
        height: 380,
        layout: { background: { color: '#fff' }, textColor: '#333' },
        grid: { vertLines: { color: '#eee' }, horzLines: { color: '#eee' } },
        rightPriceScale: { borderColor: '#ddd' },
        timeScale: { borderColor: '#ddd', timeVisible: true },
    });

    // === MACD chart (отдельная панель) ===
    const macdEl = document.getElementById('macd-chart');
    if (macdEl) {
        macdChart = LightweightCharts.createChart(macdEl, {
            width: macdEl.clientWidth,
            height: 90,
            layout: { background: { color: '#fff' }, textColor: '#333' },
            grid: { vertLines: { color: '#eee' }, horzLines: { color: '#eee' } },
            rightPriceScale: { borderColor: '#ddd' },
            timeScale: { visible: false },  // синхронизация с основным
            crosshair: { mode: 0 },
        });
        macdLineSeries = macdChart.addLineSeries({ color: '#3b82f6', lineWidth: 1, title: 'MACD' });
        macdSignalSeries = macdChart.addLineSeries({ color: '#f59e0b', lineWidth: 1, title: 'Signal' });
        macdHistSeries = macdChart.addHistogramSeries({
            color: '#16a34a', priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
            priceLineVisible: false, lastValueVisible: false,
        });
        // Синхронизация timeScale с основным графиком
        chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
            if (range) macdChart.timeScale().setVisibleLogicalRange(range);
        });
        new ResizeObserver(() => macdChart.applyOptions({ width: macdEl.clientWidth })).observe(macdEl);
    }

    // === Volume chart (отдельная панель) ===
    const volEl = document.getElementById('volume-chart');
    if (volEl) {
        volumeChart = LightweightCharts.createChart(volEl, {
            width: volEl.clientWidth,
            height: 70,
            layout: { background: { color: '#fff' }, textColor: '#333' },
            grid: { vertLines: { color: '#eee' }, horzLines: { color: '#eee' } },
            rightPriceScale: { borderColor: '#ddd' },
            timeScale: { visible: false },
        });
        volumeSeries = volumeChart.addHistogramSeries({
            color: '#9ca3af',
            priceFormat: { type: 'volume' },
            priceLineVisible: false, lastValueVisible: false,
        });
        chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
            if (range) volumeChart.timeScale().setVisibleLogicalRange(range);
        });
        new ResizeObserver(() => volumeChart.applyOptions({ width: volEl.clientWidth })).observe(volEl);
    }

    candleSeries = chart.addCandlestickSeries({
        upColor: '#16a34a', downColor: '#dc2626',
        borderVisible: false, wickUpColor: '#16a34a', wickDownColor: '#dc2626',
    });
    
    forecastSeries.gru.line = chart.addLineSeries({ color: '#8b5cf6', lineWidth: 2, title: 'GRU' });
    forecastSeries.gru.lower = chart.addLineSeries({ color: 'rgba(87, 15, 105, 0.2)', lineWidth: 1 });
    forecastSeries.gru.upper = chart.addLineSeries({ color: 'rgba(87, 15, 105, 0.2)', lineWidth: 1 });

    forecastSeries.arimax.line = chart.addLineSeries({ color: '#2563eb', lineWidth: 2, title: 'ARIMAX' });
    forecastSeries.arimax.lower = chart.addLineSeries({ color: 'rgba(37,99,235,0.2)', lineWidth: 1 });
    forecastSeries.arimax.upper = chart.addLineSeries({ color: 'rgba(37,99,235,0.2)', lineWidth: 1 });

    new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth })).observe(el);
}

function renderChart(chartData, predictions) {
    const candles = chartData.map(d => ({
        time: toTimestamp(d.date),   // ← unix seconds, не строка
        open: d.open, high: d.high, low: d.low, close: d.close,
    }));
    candleSeries.setData(candles);

    
    // === MACD ===
    const closes = chartData.map(d => d.close);
    const times = chartData.map(d => toTimestamp(d.date));
    const m = macd(closes, 12, 26, 9);

    if (macdLineSeries) {
        macdLineSeries.setData(
            m.macdLine.map((v, i) => v !== null ? { time: times[i], value: v } : null).filter(Boolean)
        );
        macdSignalSeries.setData(
            m.signalLine.map((v, i) => v !== null ? { time: times[i], value: v } : null).filter(Boolean)
        );
        macdHistSeries.setData(
            m.hist.map((v, i) => v !== null ? {
                time: times[i], value: v,
                color: v >= 0 ? 'rgba(22,163,74,0.6)' : 'rgba(220,38,38,0.6)',
            } : null).filter(Boolean)
        );
    }

    // === Volume ===
    if (volumeSeries) {
        volumeSeries.setData(
            chartData.map(d => ({
                time: toTimestamp(d.date),
                value: d.volume || 0,
                color: d.close >= d.open ? 'rgba(22,163,74,0.6)' : 'rgba(220,38,38,0.6)',
            }))
        );
    }

    const showAll = document.getElementById('toggle-all-h')?.checked;
    const showGru = document.getElementById('toggle-gru')?.checked;
    const showArimax = document.getElementById('toggle-arimax')?.checked;

    const lastDate = chartData[chartData.length - 1].date;
    const lastAnchor = chartData[chartData.length - 1].close;
    const lastTs = toTimestamp(lastDate);

    function renderForecast(key, pred, series) {
        if (!pred || !series.line) return;
        let points = [];
        if (showAll) {
            // 5 точек: anchor + h=1,3,6,12,18
            points = [{ time: lastTs, value: lastAnchor }];
            for (const h of [1, 3, 6, 12, 18]) {
                const p = pred[`point_h${h}`] ?? null;
                if (p != null) {
                    points.push({ time: lastTs + HORIZONS_HOURS[h] * 3600, value: p });
                }
            }
        } else {
            const h = pred.horizon || 6;
            points = [
                { time: lastTs, value: lastAnchor },
                { time: lastTs + HORIZONS_HOURS[h] * 3600, value: pred.point },
            ];
        }
        series.line.setData(points);

        // Интервалы (если есть lower/upper по всем h)
        if (showAll && pred.lower_h1 != null) {
            const lo = [{ time: lastTs, value: lastAnchor }];
            const hi = [{ time: lastTs, value: lastAnchor }];
            for (const h of [1, 3, 6, 12, 18]) {
                const loV = pred[`lower_h${h}`];
                const hiV = pred[`upper_h${h}`];
                if (loV != null && hiV != null) {
                    lo.push({ time: lastTs + HORIZONS_HOURS[h] * 3600, value: loV });
                    hi.push({ time: lastTs + HORIZONS_HOURS[h] * 3600, value: hiV });
                }
            }
            series.lower.setData(lo);
            series.upper.setData(hi);
        } else {
            series.lower.setData([]);
            series.upper.setData([]);
        }
    }

    // GRU
    if (showGru && predictions.gru) renderForecast('gru', predictions.gru, forecastSeries.gru);
    else {
        forecastSeries.gru.line.setData([]);
        forecastSeries.gru.lower.setData([]);
        forecastSeries.gru.upper.setData([]);
    }

    // ARIMAX
    if (showArimax && predictions.arimax) renderForecast('arimax', predictions.arimax, forecastSeries.arimax);
    else {
        forecastSeries.arimax.line.setData([]);
        forecastSeries.arimax.lower.setData([]);
        forecastSeries.arimax.upper.setData([]);
    }
}

async function runForecast() {
    const modelSel = document.getElementById('model').value;
    const horizonSel = document.getElementById('horizon').value;
    const versionSel = document.getElementById('model-version')?.value || '';
    const btn = document.getElementById('btn-forecast');
    const summary = document.getElementById('summary');
    const resultEl = document.getElementById('forecast-result');

    btn.disabled = true;
    summary.style.display = 'block';
    summary.innerText = 'Загрузка...';
    resultEl.innerHTML = '<div class="fc-loading">Расчёт прогноза...</div>';

    try {
        const modelsToRun = (modelSel === 'auto')
            ? ['attention_gru', 'arimax_lstm']
            : [modelSel];

        // Версию применяем только если выбрана ОДНА модель
        const vParam = (modelSel !== 'auto' && versionSel) ? `&version=${versionSel}` : '';

        const results = await Promise.all(
            modelsToRun.map(m =>
                fetch(`/predict_all?model=${m}${vParam}`).then(r => r.json())
            )
        );

        const KEY_MAP = { attention_gru: 'gru', arimax_lstm: 'arimax' };
        const predictions = {};
        let firstChart = null;
        const errors = [];

        for (let i = 0; i < modelsToRun.length; i++) {
            const modelName = modelsToRun[i];
            const data = results[i];

            if (data && data.error) {
                errors.push(`${modelName}: ${data.error}`);
                continue;
            }
            if (!data || !data.predictions) {
                errors.push(`${modelName}: нет predictions`);
                continue;
            }
            predictions[KEY_MAP[modelName]] = data.predictions;
            if (!firstChart && data.chart) firstChart = data.chart;
        }

        if (errors.length > 0) {
            summary.innerText = 'Ошибка: ' + errors.join('; ');
            resultEl.innerHTML = '';
            return;
        }
        if (!firstChart) {
            summary.innerText = 'Ошибка: нет данных для графика';
            resultEl.innerHTML = '';
            return;
        }

        // Фильтр по горизонту
        let filtered = {};
        for (const [key, preds] of Object.entries(predictions)) {
            filtered[key] = (horizonSel === 'all')
                ? preds
                : preds.filter(p => p.horizon === parseInt(horizonSel, 10));
        }

        // Карточки
        let cardsHtml = '';
        for (const [key, preds] of Object.entries(filtered)) {
            if (!preds.length) continue;
            const label = key === 'gru' ? 'Attention-GRU' : 'ARIMAX-LSTM';
            cardsHtml += `<div class="fc-group"><h4>${label}</h4><div class="forecast-cards">`;
            for (const p of preds) {
                const diff = p.point - p.anchor;
                const cls = diff > 0 ? 'up' : (diff < 0 ? 'down' : 'flat');
                cardsHtml += `
                    <div class="fc-card">
                        <div class="fc-h">h=${p.horizon} (${p.horizon * 4}ч)</div>
                        <div class="fc-price">$${p.point.toFixed(2)}</div>
                        <div class="fc-range">[$${p.lower.toFixed(2)}, $${p.upper.toFixed(2)}]</div>
                        <div class="fc-diff ${cls}">Δ ${diff.toFixed(2)} (${(diff / p.anchor * 100).toFixed(2)}%)</div>
                    </div>`;
            }
            cardsHtml += `</div></div>`;
        }
        resultEl.innerHTML = cardsHtml;

        // Сводка: активная модель + версия
        const activeModel = modelSel === 'auto' ? 'Обе модели' : modelSel;
        const vLabel = (modelSel !== 'auto' && versionSel) ? versionSel : 'Текущая (deploy)';
        summary.innerText = `Активная модель: ${activeModel} (${vLabel})`;

        renderChartAllHorizons(firstChart, filtered);
    } catch (e) {
        console.error('runForecast:', e);
        summary.innerText = 'Ошибка: ' + e;
        resultEl.innerHTML = '';
    } finally {
        btn.disabled = false;
        loadSignals();
        loadBacktest(); 
    }
}

async function loadSignals() {
    const el = document.getElementById('signal-body');
    if (!el) return;
    const modelSel = document.getElementById('model')?.value || 'attention_gru';
    const versionSel = document.getElementById('model-version')?.value || '';

    // Если "auto" — показываем обе модели
    const models = modelSel === 'auto' ? ['attention_gru', 'arimax_lstm'] : [modelSel];

    try {
        let html = '';
        for (const m of models) {
            const label = m === 'attention_gru' ? 'GRU' : 'ARIMAX';
            const vParam = versionSel && modelSel !== 'auto' ? `&version=${versionSel}` : '';
            const r = await fetch(`/predict_all?model=${m}${vParam}`);
            const d = await r.json();
            if (d.error || !d.predictions) {
                html += `<div class="dash-muted">${label}: нет данных</div>`;
                continue;
            }
            html += `<div class="signal-model"><b>${label}</b></div>`;
            for (const p of d.predictions) {
                const sig = p.signal || 'HOLD';
                const cls = sig === 'BUY' ? 'buy' : (sig === 'SELL' ? 'sell' : 'hold');
                const icon = sig === 'BUY' ? '🟢' : (sig === 'SELL' ? '🔴' : '⚪');
                const expPct = p.exp_ret != null ? (p.exp_ret * 100).toFixed(2) : '—';
                html += `<div class="signal-row">
                    <span class="signal-h">h=${p.horizon}</span>
                    <span class="signal-badge ${cls}">${icon} ${sig}</span>
                    <span class="signal-exp">${expPct}%</span>
                </div>`;
            }
        }
        el.innerHTML = html;
    } catch (e) {
        el.innerHTML = `<span class="warn">Ошибка: ${e}</span>`;
    }
}

function renderChartAllHorizons(chartData, predictions) {
    const candles = chartData.map(d => ({
        time: toTimestamp(d.date),
        open: d.open, high: d.high, low: d.low, close: d.close,
    }));
    candleSeries.setData(candles);

    
    // === MACD ===
    const closes = chartData.map(d => d.close);
    const times = chartData.map(d => toTimestamp(d.date));
    const m = macd(closes, 12, 26, 9);

    if (macdLineSeries) {
        macdLineSeries.setData(
            m.macdLine.map((v, i) => v !== null ? { time: times[i], value: v } : null).filter(Boolean)
        );
        macdSignalSeries.setData(
            m.signalLine.map((v, i) => v !== null ? { time: times[i], value: v } : null).filter(Boolean)
        );
        macdHistSeries.setData(
            m.hist.map((v, i) => v !== null ? {
                time: times[i], value: v,
                color: v >= 0 ? 'rgba(22,163,74,0.6)' : 'rgba(220,38,38,0.6)',
            } : null).filter(Boolean)
        );
    }

    // === Volume ===
    if (volumeSeries) {
        volumeSeries.setData(
            chartData.map(d => ({
                time: toTimestamp(d.date),
                value: d.volume || 0,
                color: d.close >= d.open ? 'rgba(22,163,74,0.6)' : 'rgba(220,38,38,0.6)',
            }))
        );
    }

    const lastDate   = chartData[chartData.length - 1].date;
    const lastAnchor = chartData[chartData.length - 1].close;
    const lastTs     = toTimestamp(lastDate);

    // Собираем максимальный горизонт (для расширения оси X)
    let maxHorizon = 0;

    function renderModel(key, series) {
        const preds = predictions[key] || [];
        if (!preds.length) {
            series.line.setData([]);
            series.lower.setData([]);
            series.upper.setData([]);
            return;
        }
        const line = [{ time: lastTs, value: lastAnchor }];
        const lo   = [{ time: lastTs, value: lastAnchor }];
        const hi   = [{ time: lastTs, value: lastAnchor }];
        for (const p of preds) {
            const t = lastTs + p.horizon * 4 * 3600;
            line.push({ time: t, value: p.point });
            lo.push({   time: t, value: p.lower });
            hi.push({   time: t, value: p.upper });
            if (p.horizon > maxHorizon) maxHorizon = p.horizon;
        }
        series.line.setData(line);
        series.lower.setData(lo);
        series.upper.setData(hi);
    }

    renderModel('gru',    forecastSeries.gru);
    renderModel('arimax', forecastSeries.arimax);

    // ★ Расширяем видимый диапазон оси X, чтобы прогноз (и его даты) были видны
    if (maxHorizon > 0) {
        const startTs = toTimestamp(chartData[0].date);
        const endTs   = lastTs + maxHorizon * 4 * 3600 + 4 * 3600; // +1 бар запаса
        chart.timeScale().setVisibleRange({ from: startTs, to: endTs });
    }
}

function ema(values, period) {
    const k = 2 / (period + 1);
    const result = [];
    let prev = null;
    for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v === null || v === undefined) { result.push(null); continue; }
        if (prev === null) { prev = v; result.push(v); }
        else { prev = v * k + prev * (1 - k); result.push(prev); }
    }
    return result;
}

function macd(closes, fast = 12, slow = 26, signal = 9) {
    const emaFast = ema(closes, fast);
    const emaSlow = ema(closes, slow);
    const macdLine = emaFast.map((f, i) =>
        (f !== null && emaSlow[i] !== null) ? f - emaSlow[i] : null
    );
    const signalLine = ema(macdLine, signal);
    const hist = macdLine.map((m, i) =>
        (m !== null && signalLine[i] !== null) ? m - signalLine[i] : null
    );
    return { macdLine, signalLine, hist };
}

function renderFearGreedGauge(value, label, date) {
    const size = 200;
    const stroke = 14;
    const radius = (size - stroke) / 2;
    const cx = size / 2;
    const cy = size / 2;
    const v = Math.max(0, Math.min(100, value));
    const angle = Math.PI - (v / 100) * Math.PI;
    const needleLen = radius - 6;
    const nx = cx + needleLen * Math.cos(angle);
    const ny = cy - needleLen * Math.sin(angle);

    const arc = (from, to) => {
        const a0 = Math.PI - (from / 100) * Math.PI;
        const a1 = Math.PI - (to / 100) * Math.PI;
        const x0 = cx + radius * Math.cos(a0);
        const y0 = cy - radius * Math.sin(a0);
        const x1 = cx + radius * Math.cos(a1);
        const y1 = cy - radius * Math.sin(a1);
        return `M ${x0} ${y0} A ${radius} ${radius} 0 0 1 ${x1} ${y1}`;
    };

    const segments = [
        { from: 0, to: 25, color: '#dc2626' },   // Extreme Fear
        { from: 25, to: 45, color: '#f59e0b' },  // Fear
        { from: 45, to: 55, color: '#9ca3af' },  // Neutral
        { from: 55, to: 75, color: '#84cc16' },  // Greed
        { from: 75, to: 100, color: '#16a34a' }, // Extreme Greed
    ];

    const segs = segments.map(s =>
        `<path d="${arc(s.from, s.to)}" fill="none" stroke="${s.color}" stroke-width="${stroke}" stroke-linecap="round" opacity="0.9"/>`
    ).join('');

    return `
        <svg width="${size}" height="${size / 2 + stroke}" viewBox="0 0 ${size} ${size / 2 + stroke}" style="display:block;margin:0 auto;">
            ${segs}
            <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="#222" stroke-width="3" stroke-linecap="round"/>
            <circle cx="${cx}" cy="${cy}" r="6" fill="#222"/>
            <circle cx="${cx}" cy="${cy}" r="2.5" fill="#fff"/>
        </svg>
        <div style="margin-top:-6px;">
            <div class="fng-value">${v}</div>
            <div class="fng-label">${label || '—'}</div>
            <div class="fng-date">${date || ''}</div>
        </div>`;
}

// ===== DASHBOARD =====
async function loadDashboard() {
    try {
        const r = await fetch('/dashboard');
        const d = await r.json();
        if (d.error) {
            document.getElementById('fng-body').innerHTML = `<span class="warn">${d.error}</span>`;
            return;
        }

        // Fear & Greed
        const fng = d.fear_greed || {};
        const fngVal = fng.value != null ? fng.value : null;
        if (fngVal != null) {
            document.getElementById('fng-body').innerHTML = renderFearGreedGauge(fngVal, fng.label, fng.date);
        } else {
            document.getElementById('fng-body').innerHTML = `<span class="warn">Нет данных</span>`;
        }

        // BTC price
        const p = d.price || {};
        const dir = p.direction || 'flat';
        const arrow = dir === 'up' ? '▲' : (dir === 'down' ? '▼' : '—');
        const cls = dir === 'up' ? 'up' : (dir === 'down' ? 'down' : 'flat');
        const last = p.last != null ? p.last.toLocaleString('en-US', {maximumFractionDigits: 2}) : '—';
        const dAbs = p.delta_abs != null ? p.delta_abs.toFixed(2) : '—';
        const dPct = p.delta_pct != null ? p.delta_pct.toFixed(2) : '—';
        document.getElementById('price-body').innerHTML = `
            <div class="price-last">$${last}</div>
            <div class="price-delta ${cls}">${arrow} ${dAbs} (${dPct}%)</div>
            <div class="price-date">${p.date || ''}</div>`;

        // Data status
        const st = d.data_status || {};
        const fngSt = st.fng || {};
        const priceSt = st.price || {};
        const onchainSt = st.onchain || {};
        const macroSt = st.macro || {};

        const badge = (s) => {
            if (s === 'live') return '<span class="badge live">live</span>';
            if (s === 'cache') return '<span class="badge cache">cache</span>';
            if (s === 'stale') return '<span class="badge stale">stale</span>';
            return '<span class="badge unknown">—</span>';
        };

        const fmtShort = (iso) => {
            if (!iso) return '';
            try {
                const dt = new Date(iso);
                if (isNaN(dt.getTime())) return iso;
                return dt.toLocaleString('ru-RU', {
                    day: '2-digit', month: '2-digit',
                    hour: '2-digit', minute: '2-digit'
                });
            } catch {
                return iso;
            }
        };

       document.getElementById('status-body').innerHTML = `
            <div class="status-cell"><span class="status-label">F&amp;G:</span> ${badge(fngSt.status)} <span class="dash-muted">${fngSt.age_seconds != null ? Math.round(fngSt.age_seconds) + 's' : ''}</span></div>
            <div class="status-cell"><span class="status-label">Price:</span> ${badge(priceSt.status)} <span class="dash-muted">${priceSt.age_seconds != null ? Math.round(priceSt.age_seconds) + 's' : ''}</span></div>
            <div class="status-cell"><span class="status-label">Onchain:</span> ${badge(onchainSt.status)} <span class="dash-muted">${fmtShort(onchainSt.last_date)}</span></div>
            <div class="status-cell"><span class="status-label">Macro:</span> ${badge(macroSt.status)} <span class="dash-muted">${fmtShort(macroSt.last_date)}</span></div>`;
        
        const retrainEl = document.getElementById('last-retrain-date');
        if (retrainEl) {
            retrainEl.textContent = fmtShort(d.last_retrain) || '—';
        }

        // Warnings (плашки)
        const warns = d.warnings || [];
        const warnEl = document.getElementById('warnings-body');
        if (warnEl) {
            warnEl.innerHTML = warns.length
                ? warns.map(w => `<div class="warn-badge">🚨 ${w}</div>`).join('')
                : '';
        }
    } catch (e) {
        console.error('loadDashboard:', e);
        document.getElementById('fng-body').innerHTML = `<span class="warn">Ошибка: ${e}</span>`;
    }
}

async function loadMetrics() {
    const el = document.getElementById('model-quality-body');
    if (!el) return;

    const modelSel = document.getElementById('model')?.value || 'attention_gru';
    const versionSel = document.getElementById('model-version')?.value || '';

    const fetchModel = modelSel === 'auto' ? 'attention_gru' : modelSel;
    const vParam = versionSel ? `&version=${versionSel}` : '';

    try {
        const r = await fetch(`/metrics?model=${fetchModel}${vParam}`);
        const data = await r.json();
        if (data.error) {
            el.innerHTML = `<span class="warn">Ошибка: ${data.error}</span>`;
            return;
        }
        const models = data.models || {};
        let html = '';
        for (const [name, m] of Object.entries(models)) {
            const label = name === 'attention_gru' ? 'Attention-GRU' : 'ARIMAX-LSTM';
            if (!m) {
                html += `<div class="mq-model"><b>${label}</b>: <span class="warn">нет данных</span></div>`;
                continue;
            }
            const metrics = m.metrics || {};
            html += `<div class="mq-model">
                <div class="mq-title"><b>${label}</b> <span class="mq-meta">${m.version} · ${(m.trained_at || '').slice(0, 16)}</span></div>
                <table class="mq-table">
                    <tr><th>h</th><th>MAE</th><th>R²</th><th>DA</th></tr>`;
            for (const h of [1, 3, 6, 12, 18]) {
                const mm = metrics[`h${h}`];
                if (!mm) continue;
                const mae = mm.mae != null ? mm.mae.toFixed(0) : '—';
                const r2 = mm.r2 != null ? mm.r2.toFixed(3) : '—';
                const da = mm.da != null ? mm.da.toFixed(1) + '%' : '—';
                html += `<tr><td>${h}</td><td>${mae}</td><td>${r2}</td><td>${da}</td></tr>`;
            }
            html += `</table></div>`;
        }
        el.innerHTML = html;
    } catch (e) {
        el.innerHTML = `<span class="warn">Ошибка: ${e}</span>`;
    }
}

async function loadBacktest() {
    const el = document.getElementById('backtest-body');
    if (!el) return;

    const modelSel = document.getElementById('model')?.value || 'attention_gru';
    const versionSel = document.getElementById('model-version')?.value || '';

    // Если "auto" — показываем обе модели; иначе — только выбранную
    const models = modelSel === 'auto'
        ? ['attention_gru', 'arimax_lstm']
        : [modelSel];

    try {
        let html = '';
        for (const m of models) {
            const label = m === 'attention_gru' ? 'Attention-GRU' : 'ARIMAX-LSTM';
            const vParam = (modelSel !== 'auto' && versionSel) ? `&version=${versionSel}` : '';
            const r = await fetch(`/backtest?model=${m}${vParam}`);
            const d = await r.json();
            if (d.error) {
                html += `<div class="bt-title">Backtest (${label}): нет данных</div>`;
                continue;
            }
            html += `<div class="bt-title">Backtest — ${label}${vParam ? ` (${versionSel})` : ''}</div>`;
            html += `<table class="bt-table"><tr><th>h</th><th>Sharpe</th><th>Return</th><th>MaxDD</th><th>Trades</th></tr>`;
            for (const h of [1, 3, 6, 12, 18]) {
                const hb = d.horizons[h];
                if (!hb) continue;
                const sharpeCls = hb.sharpe > 0 ? 'up' : 'down';
                html += `<tr>
                    <td>${h}</td>
                    <td class="${sharpeCls}">${hb.sharpe.toFixed(2)}</td>
                    <td>${(hb.total_return * 100).toFixed(1)}%</td>
                    <td>${(hb.max_drawdown * 100).toFixed(1)}%</td>
                    <td>${hb.n_trades}</td>
                </tr>`;
            }
            html += `</table>`;
            // Buy&Hold — общий для обеих моделей (на test)
            if (d.horizons["1"]) {
                html += `<div class="bt-bh">B&H: Sharpe ${d.horizons["1"].buy_hold_sharpe.toFixed(2)}, Return ${(d.horizons["1"].buy_hold_return * 100).toFixed(1)}%</div>`;
            }
        }
        el.innerHTML = html;
    } catch (e) {
        el.innerHTML = `<span class="warn">Ошибка: ${e}</span>`;
    }
}

async function loadComparison() {
    const el = document.getElementById('comparison-body');
    if (!el) return;

    const modelSel = document.getElementById('model')?.value || 'attention_gru';
    const fetchModel = modelSel === 'auto' ? 'attention_gru' : modelSel;

    try {
        const r = await fetch(`/comparison?model=${fetchModel}`);
        const d = await r.json();
        if (d.error) {
            el.innerHTML = `<div class="dash-muted">Переобучения ещё не было</div>`;
            return;
        }
        const dec = d.decision === 'promote' ? '✅ promote' : '⏸ keep';
        const delta = Object.entries(d.delta_mae || {})
            .map(([h, v]) => `h${h}: ${(v * 100).toFixed(2)}%`)
            .join(', ');
        el.innerHTML = `
            <div class="comp-title">Последнее сравнение: <b>${d.version}</b></div>
            <div class="comp-row">${dec} — <span class="dash-muted">${d.reason || ''}</span></div>
            <div class="comp-row dash-muted">Δ MAE: ${delta || '—'}</div>
            <div class="comp-row dash-muted">Дата: ${(d.trained_at || '').slice(0, 16)}</div>
        `;
    } catch (e) {
        el.innerHTML = `<span class="warn">Ошибка: ${e}</span>`;
    }
}

function renderForecastCards(data) {
    const p = data.prediction;
    const html = `
        <div class="forecast-cards">
            <div class="fc-card">
                <div class="fc-h">h=${p.horizon}</div>
                <div class="fc-price">$${p.point.toFixed(2)}</div>
                <div class="fc-range">[$${p.lower.toFixed(2)}, $${p.upper.toFixed(2)}]</div>
                <div class="fc-diff">Δ ${(p.point - p.anchor).toFixed(2)}</div>
            </div>
        </div>`;
    document.getElementById('forecast-result').innerHTML = html;
}

async function loadInitialChart() {
    try {
        const r = await fetch('/predict_all?model=attention_gru');
        const d = await r.json();
        if (d && d.chart) {
            renderChartAllHorizons(d.chart, {});
        }
    } catch (e) {
        console.error('loadInitialChart:', e);
    }
}

async function loadVersions() {
    const modelSel = document.getElementById('model').value;
    const versionSel = document.getElementById('model-version');
    if (!versionSel) return;

    // Если "auto" — скрываем выбор версии
    if (modelSel === 'auto') {
        versionSel.style.display = 'none';
        versionSel.innerHTML = '<option value="">—</option>';
        return;
    } else {
        versionSel.style.display = 'block';
    }

    try {
        const r = await fetch('/versions');
        const d = await r.json();
        const list = d[modelSel] || [];
        
        versionSel.innerHTML = '<option value="">Текущая (deploy)</option>';
        for (const v of list) {
            const h1 = v.metrics?.h1?.mae;
            const maeStr = h1 ? `MAE: ${h1.toFixed(0)}$` : 'нет метрик';
            versionSel.innerHTML += `<option value="${v.version}">${v.version} (${maeStr})</option>`;
        }
    } catch (e) {
        console.error('loadVersions:', e);
    }
}

async function triggerRetrain() {
    const btn = document.getElementById('btn-retrain');
    const status = document.getElementById('retrain-status');
    btn.disabled = true;
    status.textContent = 'Запуск...';
    status.style.color = '#666';
    
    try {
        const resp = await fetch('/retrain', { method: 'POST' });
        const data = await resp.json();
        if (resp.ok) {
            status.textContent = '⏳ Переобучение запущено (смотрите логи)';
            status.style.color = 'orange';
        } else if (resp.status === 409) {
            status.textContent = '⏳ Переобучение уже идёт';
            status.style.color = 'orange';
        } else {
            status.textContent = '❌ Ошибка: ' + (data.error || 'неизвестно');
            status.style.color = 'red';
        }
    } catch (e) {
        status.textContent = '❌ Ошибка сети';
        status.style.color = 'red';
    } finally {
        btn.disabled = false;
    }
}

function exportCsv() {
    window.location.href = '/export_predictions';
}

window.addEventListener('load', () => {
    initChart();
    loadDashboard();
    loadMetrics();
    loadComparison();
    loadVersions();
    loadInitialChart();
    loadSignals();
    loadBacktest();           // ← NEW

    document.getElementById('model').addEventListener('change', () => {
        loadVersions();
        loadMetrics();
        loadComparison();
        loadSignals();
        loadBacktest();      // ← NEW
    });
    document.getElementById('model-version').addEventListener('change', () => {
        loadMetrics();
        loadSignals(); 
        loadBacktest();     // ← NEW
    });
});