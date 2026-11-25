// 서버 없이 브라우저만으로 동작하는 버전 (GitHub Pages · index.html 더블클릭).
// service.py 와 같은 결과 형태를 만들어 화면 코드가 그대로 쓰도록 합니다.
// 기상청 API·네이버 쇼핑·학생 제보 통계는 서버(비밀 키·저장소)가 필요해 여기선 빠집니다.
const Standalone = (() => {
  const M = window.ONULCHUPNA;          // model.js (python web_export.py 로 생성)
  const C = M.config;
  const WEEKDAYS = "일월화수목금토";
  const FB_KEY = "onulchupna-feedback";
  let fc = null;

  // ---------- 시간 (항상 한국 시간 기준) ----------
  function kstNow() {
    const s = new Date().toLocaleString("sv-SE", { timeZone: "Asia/Seoul" }); // 2026-11-15 18:37:00
    const [d, t] = s.split(" ");
    const [hh, mm] = t.split(":").map(Number);
    return { date: d, hour: hh, minute: mm };
  }
  function addDays(dateStr, n) {
    const [y, m, d] = dateStr.split("-").map(Number);
    return new Date(Date.UTC(y, m - 1, d + n)).toISOString().slice(0, 10);
  }
  function dateLabel(dateStr) {
    const [y, m, d] = dateStr.split("-").map(Number);
    return `${m}월 ${d}일 (${WEEKDAYS[new Date(Date.UTC(y, m - 1, d)).getUTCDay()]})`;
  }

  // ---------- 체감온도 (campus_model.py 와 같은 기상청 식) ----------
  function windChill(t, w) {   // 10~4월: 기온 10°C 이하, 풍속 1.3 m/s 이상일 때만
    if (t > 10 || w < 1.3) return t;
    const v16 = (w * 3.6) ** 0.16;
    return 13.12 + 0.6215 * t - 11.37 * v16 + 0.3965 * v16 * t;
  }
  function summerFeels(t, rh) {
    const tw = t * Math.atan(0.151977 * Math.sqrt(rh + 8.313659)) + Math.atan(t + rh) - Math.atan(rh - 1.67633)
      + 0.00391838 * rh ** 1.5 * Math.atan(0.023101 * rh) - 4.686035;
    return -0.2442 + 0.55399 * tw + 0.45535 * t - 0.0022 * tw * tw + 0.00278 * tw * t + 3.0;
  }
  function feelsLike(t, w, rh, month) {   // 5~9월 습도 공식, 나머지 바람 공식
    return month >= 5 && month <= 9 ? summerFeels(t, rh) : windChill(t, w);
  }

  // ---------- 캠퍼스 AI 모델 (2024년 검증 1위 체크포인트) ----------
  const CM = M.campus_model;
  const CM_KIND = CM.kind || "rf";
  const CM_FALLBACK = CM.fallback || CM;
  const TARGETS = ["uni_ta", "uni_hm", "uni_ws"];
  function rfDelta(model, x, target) {
    let sum = 0;
    for (const tr of model.trees[target]) {
      let n = 0;
      while (tr.f[n] !== -1) n = x[tr.f[n]] <= tr.t[n] ? tr.l[n] : tr.r[n];
      sum += tr.v[n];
    }
    return sum / model.trees[target].length;
  }
  function lgbValue(node, x) {
    while (node.v === undefined) {
      const value = x[node.f];
      node = !Number.isFinite(value) ? (node.d ? node.l : node.r) : (value <= node.t ? node.l : node.r);
    }
    return node.v;
  }
  function modelDelta(model, x, target) {
    if (model.kind === "lightgbm") return model.trees[target].reduce((sum, tree) => sum + lgbValue(tree, x), 0);
    return rfDelta(model, x, target);
  }
  function erf(x) {
    const sign = x < 0 ? -1 : 1, value = Math.abs(x), p = 0.3275911;
    const t = 1 / (1 + p * value);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t * Math.exp(-value * value);
    return sign * y;
  }
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.sqrt(2)));
  const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-80, Math.min(80, x))));
  function linearOne(state, prefix, x) {
    const weight = state[`${prefix}.weight`], bias = state[`${prefix}.bias`];
    return weight.map((row, i) => row.reduce((sum, value, j) => sum + value * x[j], bias[i]));
  }
  function linear(state, prefix, x) {
    return Array.isArray(x[0]) ? x.map((row) => linearOne(state, prefix, row)) : linearOne(state, prefix, x);
  }
  function layerNorm(state, prefix, rows) {
    const weight = state[`${prefix}.weight`], bias = state[`${prefix}.bias`];
    return rows.map((row) => {
      const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
      const variance = row.reduce((sum, value) => sum + (value - mean) ** 2, 0) / row.length;
      const denominator = Math.sqrt(variance + 1e-5);
      return row.map((value, i) => (value - mean) / denominator * weight[i] + bias[i]);
    });
  }
  function transpose(matrix) { return matrix[0].map((_, column) => matrix.map((row) => row[column])); }
  function lstmScaled(model, sequence) {
    const state = model.state;
    let layerInput = sequence;
    for (let layer = 0; layer < (model.layers || 2); layer++) {
      const wih = state[`lstm.weight_ih_l${layer}`], whh = state[`lstm.weight_hh_l${layer}`];
      const bih = state[`lstm.bias_ih_l${layer}`], bhh = state[`lstm.bias_hh_l${layer}`];
      const hiddenSize = whh[0].length;
      let h = new Array(hiddenSize).fill(0), c = new Array(hiddenSize).fill(0);
      const output = [];
      for (const x of layerInput) {
        const gates = wih.map((row, gate) => row.reduce((sum, value, j) => sum + value * x[j], bih[gate] + bhh[gate])
          + whh[gate].reduce((sum, value, j) => sum + value * h[j], 0));
        const nextH = new Array(hiddenSize), nextC = new Array(hiddenSize);
        for (let j = 0; j < hiddenSize; j++) {
          const inputGate = sigmoid(gates[j]), forgetGate = sigmoid(gates[hiddenSize + j]);
          const candidate = Math.tanh(gates[2 * hiddenSize + j]), outputGate = sigmoid(gates[3 * hiddenSize + j]);
          nextC[j] = forgetGate * c[j] + inputGate * candidate;
          nextH[j] = outputGate * Math.tanh(nextC[j]);
        }
        h = nextH; c = nextC; output.push(h.slice());
      }
      layerInput = output;
    }
    const joined = layerInput[layerInput.length - 1].concat(sequence[sequence.length - 1]);
    return linearOne(state, "head.2", linearOne(state, "head.0", joined).map(gelu));
  }
  function mixerScaled(model, sequence) {
    const state = model.state;
    let z = linear(state, "proj", sequence);
    for (let block = 0; block < (model.blocks || 3); block++) {
      const prefix = `blocks.${block}`;
      const temporal = transpose(linear(state, `${prefix}.time.0`, transpose(layerNorm(state, `${prefix}.norm_t`, z)))
        .map((row) => row.map(gelu)));
      z = z.map((row, i) => row.map((value, j) => value + temporal[i][j]));
      let feature = linear(state, `${prefix}.feat.0`, layerNorm(state, `${prefix}.norm_f`, z))
        .map((row) => row.map(gelu));
      feature = linear(state, `${prefix}.feat.3`, feature);
      z = z.map((row, i) => row.map((value, j) => value + feature[i][j]));
    }
    const joined = z.flat().concat(sequence[sequence.length - 1]);
    return linearOne(state, "head.2", linearOne(state, "head.0", joined).map(gelu));
  }
  function neuralDelta(model, rawSequence) {
    const sequence = rawSequence.map((row) => row.map((value, i) => (value - model.input_mean[i]) / model.input_std[i]));
    const scaled = model.kind === "lstm" ? lstmScaled(model, sequence) : mixerScaled(model, sequence);
    return scaled.map((value, i) => value * model.target_std[i] + model.target_mean[i]);
  }
  function currentFeatures(t, w, rh, month, hour) {
    return [t, rh, w, Math.sin(2 * Math.PI * hour / 24), Math.cos(2 * Math.PI * hour / 24),
            Math.sin(2 * Math.PI * month / 12), Math.cos(2 * Math.PI * month / 12)];
  }
  function corrected(t, w, rh, delta) {
    return { temp: t + delta[0], humidity: Math.min(100, Math.max(0, rh + delta[1])),
             wind: Math.max(0, w + delta[2]) };
  }
  function fallbackPredict(t, w, rh, month, hour) {
    const model = CM_KIND === "rf" ? CM : CM_FALLBACK;
    const x = currentFeatures(t, w, rh, month, hour);
    return corrected(t, w, rh, TARGETS.map((target) => modelDelta(model, x, target)));
  }
  function campusPredict(t, w, rh, month, hour) {
    if (CM_KIND !== "rf") return fallbackPredict(t, w, rh, month, hour);
    const x = currentFeatures(t, w, rh, month, hour);
    return corrected(t, w, rh, TARGETS.map((target) => modelDelta(CM, x, target)));
  }
  const rowTime = (row) => Date.parse(`${row.date}T${String(row.hour).padStart(2, "0")}:00:00+09:00`);
  function historyRow(rows, i, lag) {
    const j = i - lag;
    return j >= 0 && rowTime(rows[i]) - rowTime(rows[j]) === lag * 3600000 ? rows[j] : null;
  }
  function windowRows(rows, i, size) {
    const first = historyRow(rows, i, size - 1);
    return first ? rows.slice(i - size + 1, i + 1) : null;
  }
  function lagFeatureMap(rows, i) {
    const r = rows[i], month = +r.date.slice(5, 7), hour = r.hour;
    const values = { aws_ta: r.temp, aws_hm: r.humidity, aws_ws: r.wind,
      hour_sin: Math.sin(2 * Math.PI * hour / 24), hour_cos: Math.cos(2 * Math.PI * hour / 24),
      month_sin: Math.sin(2 * Math.PI * month / 12), month_cos: Math.cos(2 * Math.PI * month / 12) };
    const source = { aws_ta: "temp", aws_hm: "humidity", aws_ws: "wind" };
    for (const [column, key] of Object.entries(source)) {
      for (const lag of [1, 2, 3, 6, 12, 24]) values[`${column}_lag${lag}`] = historyRow(rows, i, lag)?.[key] ?? NaN;
      values[`${column}_d1`] = r[key] - values[`${column}_lag1`];
      values[`${column}_d3`] = r[key] - values[`${column}_lag3`];
      for (const size of [3, 6, 24]) {
        const window = windowRows(rows, i, size);
        values[`${column}_mean${size}`] = window ? window.reduce((sum, row) => sum + row[key], 0) / size : NaN;
      }
      const six = windowRows(rows, i, 6);
      if (six) {
        const mean = six.reduce((sum, row) => sum + row[key], 0) / six.length;
        values[`${column}_std6`] = Math.sqrt(six.reduce((sum, row) => sum + (row[key] - mean) ** 2, 0) / (six.length - 1));
      } else values[`${column}_std6`] = NaN;
    }
    const day = windowRows(rows, i, 24), sixWind = windowRows(rows, i, 6);
    values.ta_range24 = day ? Math.max(...day.map((row) => row.temp)) - Math.min(...day.map((row) => row.temp)) : NaN;
    values.ws_max6 = sixWind ? Math.max(...sixWind.map((row) => row.wind)) : NaN;
    return values;
  }
  function sequenceFeatures(row) {
    const month = +row.date.slice(5, 7), hour = row.hour;
    return [row.temp, row.humidity, row.wind,
      Math.sin(2 * Math.PI * hour / 24), Math.cos(2 * Math.PI * hour / 24),
      Math.sin(2 * Math.PI * month / 12), Math.cos(2 * Math.PI * month / 12)];
  }
  function applyCampusSeries(rows) {
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i], month = +r.date.slice(5, 7);
      if (CM_KIND === "rf") {
        r._campus = campusPredict(r.temp, r.wind, r.humidity, month, r.hour);
        continue;
      }
      if (CM_KIND === "lstm" || CM_KIND === "tsmixer") {
        const window = windowRows(rows, i, CM.window);
        r._campus = window
          ? corrected(r.temp, r.wind, r.humidity, neuralDelta(CM, window.map(sequenceFeatures)))
          : fallbackPredict(r.temp, r.wind, r.humidity, month, r.hour);
        continue;
      }
      const featureMap = lagFeatureMap(rows, i);
      const x = CM.features.map((name) => featureMap[name]);
      r._campus = x.every(Number.isFinite)
        ? corrected(r.temp, r.wind, r.humidity, TARGETS.map((target) => modelDelta(CM, x, target)))
        : fallbackPredict(r.temp, r.wind, r.humidity, month, r.hour);
    }
  }
  // 오늘 실측 반영: 최근 정시 실측 − (그 시각 예보 → AI 예측). 예보가 빗나간 만큼과 AI가 빗나간 만큼이
  // 함께 잡히고, 몇 시간 동안 같은 방향으로 이어지므로 exp(−경과시간/4시간)으로 줄여서 더한다.
  // (2025년 검증: ±1°C 정답률 1시간 뒤 72.0 → 78.9%, 3시간 뒤 74.6%, 6시간 뒤 거의 원래대로)
  const NOWCAST_TAU = 4;
  function nowcastError(nc) {
    if (!nc || !nc.obs || !nc.fcst || hoursAgo(nc.time) > 12) return null;
    const date = nc.time.slice(0, 10), hour = +nc.time.slice(11, 13);
    const p = campusPredict(nc.fcst.temp, nc.fcst.wind, nc.fcst.humidity, +date.slice(5, 7), hour);
    return { date, hour, temp: nc.obs.temp - p.temp, humidity: nc.obs.humidity - p.humidity,
             wind: nc.obs.wind - p.wind };
  }
  function nowcastWeight(err, date, hour) {
    if (!err) return 0;
    const lead = (Date.parse(date + "T00:00:00Z") - Date.parse(err.date + "T00:00:00Z")) / 3600000 + hour - err.hour;
    return Math.exp(-Math.max(0, lead) / NOWCAST_TAU);
  }
  function campusFeel(t, w, rh, month, hour, sens, err = null, date = null, predicted = null) {
    const b = predicted || campusPredict(t, w, rh, month, hour); // 반영 전 AI 예측
    const k = err && date ? nowcastWeight(err, date, hour) : 0;   // 오늘 실측 반영 가중치
    const c = k ? { temp: b.temp + k * err.temp,
                    humidity: Math.min(100, Math.max(0, b.humidity + k * err.humidity)),
                    wind: Math.max(0, b.wind + k * err.wind) } : b;
    const appFeels = feelsLike(t, w, rh, month);
    const f = feelsLike(c.temp, c.wind, c.humidity, month);
    const base = feelsLike(b.temp, b.wind, b.humidity, month);
    const mid = feelsLike(b.temp, w, b.humidity, month);
    let low = f, high = f;   // 겨울에 풍속이 1.3 m/s 기준선 근처면 범위로
    if (!(month >= 5 && month <= 9) && c.temp <= 10 && Math.abs(c.wind - 1.3) <= 0.3) {
      const a = feelsLike(c.temp, 1.6, c.humidity, month), bb = feelsLike(c.temp, 1.0, c.humidity, month);
      low = Math.min(a, bb, f); high = Math.max(a, bb, f);
    }
    const s = C.sensitivity[sens] ?? 0;
    return { app: t, appFeels, campusTemp: c.temp, campusHum: c.humidity, campusWind: c.wind,
             feels: f, personal: f + s, low: low + s, high: high + s,
             climate: mid - appFeels, wind: base - mid, nowcast: f - base, nowcastW: k, sens: s };
  }

  // ---------- AI 모델 (Random Forest 추론) ----------
  function predict(x) {
    const proba = new Array(M.classes.length).fill(0);
    for (const tr of M.trees) {
      let n = 0;
      while (tr.f[n] !== -1) n = x[tr.f[n]] <= tr.t[n] ? tr.l[n] : tr.r[n];
      tr.v[n].forEach((p, i) => (proba[i] += p));
    }
    return M.classes[proba.indexOf(Math.max(...proba))];
  }

  // ---------- 날씨 ----------
  function demoForecast() {
    const mean = [-2.4, 0.4, 5.7, 12.5, 17.8, 22.2, 25.0, 25.7, 21.2, 14.8, 7.2, 0.4];
    const today = kstNow().date, rows = [];
    for (let d = -1; d < 3; d++) {
      const date = addDays(today, d), m = +date.slice(5, 7);
      for (let h = 0; h < 24; h++) rows.push({
        date, hour: h, temp: +(mean[m - 1] + 4.5 * Math.sin((h - 9) / 24 * 2 * Math.PI)).toFixed(1),
        wind: +(2.5 + 1.5 * Math.sin(h / 24 * 2 * Math.PI)).toFixed(1), humidity: 55, pop: 10 });
    }
    return { hourly: rows, source: "데모 데이터(오프라인)" };
  }
  // 한국 시간 "YYYY-MM-DD HH:MM" → 지금부터 몇 시간 전인지
  const hoursAgo = (s) => (Date.now() - new Date(s.replace(" ", "T") + ":00+09:00")) / 3600000;

  async function forecast() {
    if (fc && Date.now() - fc.at < 600000) return fc;
    // 1) GitHub Actions가 매시간 받은 기상청 만안구 실황·단기예보 (인증키는 GitHub 비밀값에만 있음)
    try {
      const js = await (await fetch("forecast.json?t=" + Date.now(), { cache: "no-store" })).json();
      if (Array.isArray(js.hourly) && js.hourly.length && hoursAgo(js.updated_at) < 6) {
        fc = { hourly: js.hourly, source: js.source || "기상청 단기예보" };
        if (js.now && hoursAgo(js.now.time) < 3) fc.now = js.now;
        applyCampusSeries(fc.hourly);
        fc.at = Date.now();
        return fc;
      }
    } catch (e) { /* 파일 없음(더블클릭 실행 등) → Open-Meteo */ }
    // 2) Open-Meteo (무료, 키 불필요) → 3) 데모
    try {
      const q = new URLSearchParams({
        latitude: C.app_ref[0], longitude: C.app_ref[1],
        hourly: "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation_probability",
        wind_speed_unit: "ms", timezone: "Asia/Seoul", past_days: 1, forecast_days: 3 });
      const js = await (await fetch("https://api.open-meteo.com/v1/forecast?" + q)).json();
      const h = js.hourly;
      fc = { hourly: h.time.map((t, i) => ({
              date: t.slice(0, 10), hour: +t.slice(11, 13), temp: h.temperature_2m[i] ?? 0,
              wind: h.wind_speed_10m[i] ?? 0, humidity: h.relative_humidity_2m[i] ?? 50,
              pop: h.precipitation_probability[i] ?? 0 })),
             source: "Open-Meteo 예보" };
    } catch (e) {
      fc = demoForecast();
    }
    applyCampusSeries(fc.hourly);
    fc.at = Date.now();
    return fc;
  }

  // ---------- 체감 제보 (이 기기 브라우저에만 저장) ----------
  function readFb() {
    try { return JSON.parse(localStorage.getItem(FB_KEY) || "[]"); } catch { return []; }
  }
  function stats() {
    const today = kstNow().date;
    const fb = readFb().filter((r) => r.date === today);
    const total = fb.length, choices = ["추워요", "적당해요", "더워요"];
    const counts = Object.fromEntries(choices.map((c) => [c, fb.filter((r) => r.feedback === c).length]));
    const recent = fb.slice(-3).reverse().map((r) => {
      const mins = Math.floor((Date.now() - r.at) / 60000);
      const top = C.levels[r.worn_level].categories["상의"][0];
      return { text: `${top} 입었는데 ${r.feedback}`,
               ago: mins < 1 ? "방금" : mins < 60 ? `${mins}분 전` : `${Math.floor(mins / 60)}시간 전` };
    });
    return { total, counts, recent, local: true,
             pct: Object.fromEntries(choices.map((c) => [c, total ? Math.round(100 * counts[c] / total) : 0])) };
  }

  // ---------- 지금 캠퍼스 체감온도 ----------
  // GitHub Actions가 매시간 만든 campus_now.json(안양대 500m 격자 실측, 인증키는 GitHub 비밀값에만 있음).
  // 파일이 없거나(더블클릭 실행 등) 6시간보다 오래됐으면 지금 시각 예보를 AI 모델에 넣어 예측.
  let live = null;   // campus_now.json (10분 캐시)
  async function liveCampus() {
    if (live && Date.now() - live.at < 600000) return live.js;
    let js = null;
    try { js = await (await fetch("campus_now.json?t=" + Date.now(), { cache: "no-store" })).json(); }
    catch (e) { /* 파일 없음(더블클릭 실행 등) */ }
    live = { js, at: Date.now() };
    return js;
  }
  async function campusNow(cur) {
    const js = await liveCampus();
    if (js && Number.isFinite(js.feels) && hoursAgo(js.time) < 6) return {
      feels: +js.feels.toFixed(1), temp: js.temp, wind: js.wind, humidity: Math.round(js.humidity),
      time: js.time.slice(11, 16), kind: "실측" };
    const cm = campusFeel(cur.temp, cur.wind, cur.humidity, +cur.date.slice(5, 7), cur.hour, "보통", null, null, cur._campus);
    return { feels: +cm.feels.toFixed(1), temp: +cm.campusTemp.toFixed(1), wind: +cm.campusWind.toFixed(1),
             humidity: Math.round(cm.campusHum), time: `${String(cur.hour).padStart(2, "0")}:00`, kind: "AI 예측" };
  }

  // ---------- 오늘 AI 예측 vs 캠퍼스 실측 (history.json, GitHub Actions가 매시간 기록) ----------
  async function todayCheck() {
    try {
      const js = await (await fetch("history.json?t=" + Date.now(), { cache: "no-store" })).json();
      const today = kstNow().date;
      const LEAD_KEY = "h2";   // 2시간 전 예측 (update_live_weather.py 의 LEAD_HOURS 와 맞출 것)
      const rows = Object.keys(js.obs || {}).filter((k) => k.startsWith(today) && js.pred?.[k]?.last).sort()
        .map((k) => {
          const o = js.obs[k], p = js.pred[k];
          const lead = p[LEAD_KEY] || p.last;     // 2시간 전 기록이 없던 시각(도입 전)은 직전 값으로 대신
          return { hour: +k.slice(11, 13), actual: o.feels, temp: o.temp, wind: o.wind,
                   first: p.first.feels, firstMade: p.first.made.slice(11, 16),
                   last: lead.feels, lastMade: lead.made.slice(11, 16), approx: !p[LEAD_KEY],
                   diff: +(lead.feels - o.feels).toFixed(1) };
        });
      const hit = rows.filter((r) => Math.abs(r.diff) <= 1).length;
      const mae = rows.length ? rows.reduce((a, r) => a + Math.abs(r.diff), 0) / rows.length : null;
      return { rows, hit, mae: mae === null ? null : +mae.toFixed(1) };
    } catch (e) { return null; }
  }

  // ---------- service.py 와 같은 응답 ----------
  async function home() {
    const f = await forecast(), now = kstNow();
    const cur = f.hourly.find((r) => r.date === now.date && r.hour === now.hour) || f.hourly[0];
    const appNow = f.now ? f.now.temp : cur.temp;          // 초단기실황이 있으면 지금 관측 기온
    const appLabel = f.now ? `${f.now.source || "기상청 초단기실황"} ${f.now.time.slice(11, 16)}` : f.source;
    const ampm = now.hour < 12 ? "오전" : "오후";
    const campus = await campusNow(cur);
    const check = await todayCheck();
    return {
      today_check: check,
      campus: C.campus,
      now_label: `${dateLabel(now.date)} ${ampm} ${now.hour % 12 || 12}:${String(now.minute).padStart(2, "0")}`,
      app_temp: Math.round(appNow), app_label: appLabel, source: f.source,
      app_ref: C.app_ref_name, campus_now: campus,
      errors: [f.source.startsWith("기상청") ? "웹 체험판이에요. 기상청 값은 매시간 갱신되고, 학생 제보 모으기는 로컬 실행 버전에서 지원돼요."
                                             : `웹 체험판이에요. 기상청 예보 파일이 없어 ${f.source}로 계산하고 있어요.`],
      schedules: Object.keys(C.schedules), sensitivities: Object.keys(C.sensitivity),
      levels: C.levels.map((l) => ({ level: l.level, name: l.name, items: Object.values(l.categories).flat() })),
      stats: stats(),
      data: { campus_model: `${CM.name || "캠퍼스 AI"} · ${CM.validation_year || "-"}년 검증 선택`,
              campus_obs: campus.kind === "실측" ? "안양대 500m 격자 실측 (기상청, 매시간 갱신)"
                                                 : "실측 파일이 없어 지금 시각 AI 예측" },
    };
  }

  async function analyze({ day, schedule, sensitivity }) {
    const f = await forecast(), now = kstNow();
    const target = day === "오늘" ? now.date : addDays(now.date, 1);
    const [sh, eh] = C.schedules[schedule] || [9, 18];
    let sel = f.hourly.filter((r) => r.date === target && r.hour >= sh && r.hour <= eh);
    if (day === "오늘") sel = sel.filter((r) => r.hour >= now.hour);
    if (!sel.length) return { error: "선택한 시간대의 예보가 없어요. 오늘 일정이 이미 끝났다면 '내일'을 선택하세요." };

    const sensCode = Object.keys(C.sensitivity).indexOf(sensitivity);
    const month = +target.slice(5, 7);
    const err = f.source.startsWith("기상청") ? nowcastError((await liveCampus())?.nowcast) : null;
    const hours = sel.map((r) => {
      const cf = campusFeel(r.temp, r.wind, r.humidity, month, r.hour, sensitivity, err, target, r._campus);
      const x = [cf.campusTemp, cf.campusWind, cf.campusHum, cf.personal, r.hour, month, sensCode, r.pop];
      return { hour: r.hour, app: r.temp, wind: r.wind, pop: r.pop, feels: cf.personal, level: predict(x), cf };
    });

    const worst = hours.reduce((a, b) => (b.level > a.level || (b.level === a.level && b.feels < a.feels) ? b : a));
    const cf = worst.cf, lv = C.levels[worst.level];
    const levels = hours.map((h) => h.level);
    const layering = Math.max(...levels) - Math.min(...levels) >= 2;
    const rainMax = Math.max(...hours.map((h) => h.pop));

    const cats = {};
    for (const [k, v] of Object.entries(lv.categories)) cats[k] = [...v];
    if (rainMax >= 60) for (const [k, extra] of Object.entries(C.rain_extra))
      cats[k] = [...extra, ...cats[k].filter((i) => i !== "샌들" && i !== "슬리퍼")];

    let tip;
    if (rainMax >= 60) tip = `강수확률이 최대 ${Math.round(rainMax)}%예요. 우산을 꼭 챙기세요.`;
    else if (layering) tip = "일교차가 커요. 벗기 쉬운 겉옷으로 겹쳐 입고 낮에는 한 겹 벗으세요.";
    else if (cf.wind <= -1) tip = "건물 사이 바람이 강해요. 목을 감싸는 옷이 체감온도를 크게 올려줘요.";
    else tip = "실내 강의실은 따뜻하니 안에 너무 두껍게 입기보다 겉옷으로 조절하세요.";

    const diff = Math.round(cf.appFeels - cf.personal);
    const head = diff >= 1 ? `날씨앱만 보고 얇게 입으면 후회해요! 캠퍼스는 날씨앱보다 ${diff}° 더 춥게 느껴져요. `
      : diff <= -1 ? `오늘 캠퍼스는 날씨앱보다 ${-diff}° 덜 춥게 느껴져요. ` : "오늘 캠퍼스 체감은 날씨앱과 비슷해요. ";
    const comment = `${head}${worst.hour}시가 체감 ${Math.round(cf.personal)}°로 가장 추우니 ${cats["상의"][0]}에 ${cats["의류 잡화"][0]}까지 챙기세요.`;

    const chart = hours.length <= 6 ? hours : hours.filter((_, i) => i % 3 === 0 || i === hours.length - 1);
    const minFeels = Math.min(...hours.map((h) => h.feels));
    const last = hours[hours.length - 1];
    const warning = hours.length >= 2 && last.feels <= minFeels && hours[0].feels - last.feels >= 3
      ? `${last.hour}시 이후 기온 급하강 주의` : null;
    const st = stats();
    const sgn = (v) => (v >= 0 ? "+" : "") + v.toFixed(1) + "°";

    return {
      source: f.source, errors: [], date_label: dateLabel(target), schedule, end_hour: eh,
      steps: {
        weather: `${Math.round(hours[0].app)}° · ${hours[0].wind.toFixed(1)}m/s`,
        campus: sgn(cf.climate), wind: sgn(cf.wind),
        nowcast: err ? `${err.hour}시 실측 ${sgn(cf.nowcast)}` : "실측 없음",
        feedback: `${st.total}건 반영`, sensitivity: sgn(cf.sens),
      },
      result: {
        hour: worst.hour, feels: +cf.personal.toFixed(1), app: +cf.appFeels.toFixed(1), diff,
        campus: +cf.climate.toFixed(1), wind: +cf.wind.toFixed(1), nowcast: +cf.nowcast.toFixed(1), sensitivity: +cf.sens.toFixed(1),
        range: Math.round(cf.low) !== Math.round(cf.high) ? [Math.round(cf.low), Math.round(cf.high)] : null,
        nowcast_hour: err && nowcastWeight(err, target, sel[0].hour) >= 0.05 ? err.hour : null,
        basis: cf.nowcastW >= 0.05 ? "검증 최고 모델 + 오늘 실측 반영" : "검증 최고 모델",
        level: lv.level, level_name: lv.name,
        categories: Object.entries(cats).map(([name, items]) => ({
          name, emoji: C.category_emoji[name],
          items: items.map((i) => ({ name: i, emoji: C.item_emoji[i] || "👕" })) })),
        comment, tip, model: `${CM.name || "캠퍼스 AI"} (2024 검증 선택)`, feedback_total: st.total,
      },
      hourly: chart.map((h) => ({ hour: h.hour, app: +h.app.toFixed(1), feels: +h.feels.toFixed(1) })),
      warning, stats: st,
    };
  }

  async function feedback({ worn_level, feedback }) {
    const fb = readFb();
    fb.push({ worn_level, feedback, date: kstNow().date, at: Date.now() });
    try { localStorage.setItem(FB_KEY, JSON.stringify(fb.slice(-200))); } catch {}
    return stats();
  }

  async function products(item) {
    const links = {};
    for (const [site, url] of Object.entries(C.search_links)) links[site] = url.replace("{q}", encodeURIComponent(item));
    return { products: [], links };
  }

  return { home, analyze, feedback, products, _neuralDelta: neuralDelta };
})();
