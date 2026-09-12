/* Review Analyzer — UI controller. Depends on sentiment.js and chart.js */
(function () {
  "use strict";
  const SA = window.SA;
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const pct = (v) => (v * 100).toFixed(1) + "%";
  const NAMES = { perceptron: "Perceptron", average: "Average perceptron", pegasos: "Pegasos" };
  const COLORS = { perceptron: "#c96a0e", average: "#6b3fa0", pegasos: "#2f6fd6" };
  const EXAMPLES = [
    "Absolutely delicious — the best dark chocolate I have ever ordered online, will buy again!",
    "Arrived stale and the bag was half empty. Very disappointed for the price.",
    "The tea is ok. Not bad, not great. Shipping was fast though.",
    "I was worried it would be too sweet but it's perfect for my morning coffee.",
  ];

  // ------------------------------------------------------------ data + models
  let D = null, dict = null, X = null, XV = null, XT = null, y = null, yv = null, yt = null;
  const models = {};      // kind -> {theta, theta0, cfg}
  let featOpts = {};      // current feature config
  async function load() {
    D = await (await fetch("data/reviews.json")).json();
    y = D.train.map((r) => r.y); yv = D.val.map((r) => r.y); yt = D.test.map((r) => r.y);
    buildFeatures({});
    for (const k of ["perceptron", "average", "pegasos"]) models[k] = SA.train(k, X, y, dict.size, { T: 25, L: 0.01, order: D.order4000 });
    toast("Three classifiers trained on 4,000 reviews in your browser.");
    $("reviewText").value = EXAMPLES[0];
    analyze(); renderWords(); renderReviews(); renderTrainStats();
  }
  function buildFeatures(opts) {
    featOpts = opts;
    dict = SA.bagOfWords(D.train.map((r) => r.t), opts);
    X = SA.featurize(D.train.map((r) => r.t), dict, opts); XV = SA.featurize(D.val.map((r) => r.t), dict, opts); XT = SA.featurize(D.test.map((r) => r.t), dict, opts);
  }

  // ------------------------------------------------------------ analyze
  function analyze() {
    const text = $("reviewText").value;
    const kind = $("anModel").value; const m = models[kind]; if (!m) return;
    const f = SA.features(text, dict, featOpts);
    const s = SA.score(f, m);
    const v = $("verdict");
    if (!text.trim()) { v.innerHTML = `<span class="big none">Type something…</span>`; $("highlight").innerHTML = ""; $("contrib").innerHTML = ""; return; }
    const scale = kind === "pegasos" ? 3 : 12; // typical magnitudes differ
    const w = Math.min(50, (Math.abs(s) / scale) * 50);
    v.innerHTML = `<span class="big ${s > 1e-5 ? "pos" : "neg"}">${s > 1e-5 ? "👍 Positive" : "👎 Negative"}</span><span class="meter"><i class="${s > 0 ? "pos" : "neg"}" style="width:${w}%"></i></span><span class="score">θ·x + θ₀ = ${s.toFixed(2)}</span>`;
    const parts = SA.explain(text, dict, m, featOpts);
    let maxAbs = 0.01; parts.forEach((p) => { maxAbs = Math.max(maxAbs, Math.abs(p.weight)); });
    $("highlight").innerHTML = parts.map((p) => { const t = p.weight / maxAbs; const bg = t > 0 ? `rgba(22,163,74,${(t * 0.55).toFixed(2)})` : `rgba(220,38,38,${(-t * 0.55).toFixed(2)})`; return `<span style="background:${p.weight ? bg : "transparent"}" title="${p.weight ? p.weight.toFixed(3) : "not in vocabulary / repeated"}">${escapeHtml(p.word)}</span>`; }).join(" ");
    const contrib = parts.filter((p) => p.weight).sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, 12);
    $("contrib").innerHTML = contrib.map((p) => `<div class="cbar"><span class="w">${escapeHtml(p.word)}</span><span class="track"><i class="${p.weight > 0 ? "pos" : "neg"}" style="width:${(Math.abs(p.weight) / maxAbs) * 50}%"></i></span><span class="v">${p.weight > 0 ? "+" : ""}${p.weight.toFixed(2)}</span></div>`).join("") + `<div class="cbar"><span class="w">θ₀ (bias)</span><span class="track"><i class="${m.theta0 > 0 ? "pos" : "neg"}" style="width:${Math.min(50, (Math.abs(m.theta0) / maxAbs) * 50)}%"></i></span><span class="v">${m.theta0 > 0 ? "+" : ""}${m.theta0.toFixed(2)}</span></div>`;
    const unknown = parts.filter((p) => !dict.has(p.word)).length;
    const tile = (l, v, sub) => `<div class="tile"><div class="tile-label">${l}</div><div class="tile-value">${v}</div>${sub ? `<div class="tile-sub">${sub}</div>` : ""}</div>`;
    $("anStats").innerHTML = tile("Words", parts.length, `${unknown} not in the 13k vocabulary`) + tile("Model", NAMES[kind], `test accuracy ${pct(SA.accuracy(XT, yt, m))}`) + tile("Vocabulary", dict.size.toLocaleString(), "features");
  }
  const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ------------------------------------------------------------ train & compare
  const chart = window.LineChart.create($("chart"), $("tip"), $("legend"), { yMin: 0.5, yMax: 1, yFormat: (v) => Math.round(v * 100) + "%", xFormat: (v) => String(v), xLabel: "epochs" });
  const curves = { perceptron: [], average: [], pegasos: [] };
  let training = false;
  function currentOpts() { return { stopwords: $("trStop").checked ? new Set(D.stopwords) : null, binarize: !$("trCounts").checked, bigrams: $("trBigrams").checked }; }
  async function trainAll() {
    if (training) return; training = true; $("btnTrain").disabled = true; $("btnSweep").disabled = true;
    const opts = currentOpts(); const T = Number($("trT").value), L = Number($("trL").value);
    const t0 = performance.now();
    buildFeatures(opts);
    for (const k of Object.keys(curves)) curves[k] = [];
    renderCurves();
    for (const kind of ["perceptron", "average", "pegasos"]) {
      await sleep(0);
      models[kind] = SA.train(kind, X, y, dict.size, { T, L, order: D.order4000, onEpoch: (t, m) => { curves[kind].push([t, SA.accuracy(XV, yv, m)]); } });
      renderCurves(); renderTrainStats(); await sleep(0);
    }
    toast(`Trained all three in ${((performance.now() - t0) / 1000).toFixed(1)} s on ${dict.size.toLocaleString()} features.`);
    training = false; $("btnTrain").disabled = false; $("btnSweep").disabled = false;
    analyze(); renderWords(); renderReviews();
  }
  function renderCurves() { chart.draw(Object.keys(curves).filter((k) => curves[k].length).map((k) => ({ name: NAMES[k], color: COLORS[k], points: curves[k] }))); }
  function renderTrainStats() {
    const tile = (l, v, sub, color) => `<div class="tile"><div class="tile-label">${l}</div><div class="tile-value" ${color ? `style="color:${color}"` : ""}>${v}</div>${sub ? `<div class="tile-sub">${sub}</div>` : ""}</div>`;
    $("trStats").innerHTML = tile("Features", dict.size.toLocaleString(), "vocabulary size") + ["perceptron", "average", "pegasos"].map((k) => models[k] ? tile(NAMES[k], pct(SA.accuracy(XT, yt, models[k])), `train ${pct(SA.accuracy(X, y, models[k]))} · val ${pct(SA.accuracy(XV, yv, models[k]))} · test`, COLORS[k]) : "").join("");
  }
  async function sweep() {
    if (training) return; training = true; $("btnTrain").disabled = true; $("btnSweep").disabled = true;
    const opts = currentOpts(); buildFeatures(opts);
    const Ts = [1, 5, 10, 15, 25, 50], Ls = [0.001, 0.01, 0.1, 1];
    const rows = []; const box = $("sweep"); box.innerHTML = `<p class="hint">Sweeping…</p>`;
    for (const T of Ts) { await sleep(0); const r = { T }; for (const k of ["perceptron", "average"]) { const m = SA.train(k, X, y, dict.size, { T, order: D.order4000 }); r[k] = SA.accuracy(XV, yv, m); } r.pegasos = SA.accuracy(XV, yv, SA.train("pegasos", X, y, dict.size, { T, L: 0.01, order: D.order4000 })); rows.push(r); }
    const lrows = []; for (const L of Ls) { await sleep(0); lrows.push({ L, acc: SA.accuracy(XV, yv, SA.train("pegasos", X, y, dict.size, { T: 25, L, order: D.order4000 })) }); }
    const best = (k) => Math.max(...rows.map((r) => r[k]));
    box.innerHTML = `<div class="sweep"><h3>Validation accuracy vs T</h3><table><thead><tr><th>T</th><th>Perceptron</th><th>Average perceptron</th><th>Pegasos (λ=0.01)</th></tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${r.T}</td>${["perceptron", "average", "pegasos"].map((k) => `<td class="${r[k] === best(k) ? "best" : ""}">${pct(r[k])}</td>`).join("")}</tr>`).join("") +
      `</tbody></table><h3>Pegasos: validation accuracy vs λ (T = 25)</h3><table><thead><tr><th>λ</th><th>Accuracy</th></tr></thead><tbody>` +
      lrows.map((r) => `<tr><td>${r.L}</td><td class="${r.acc === Math.max(...lrows.map((x) => x.acc)) ? "best" : ""}">${pct(r.acc)}</td></tr>`).join("") + `</tbody></table></div>`;
    training = false; $("btnTrain").disabled = false; $("btnSweep").disabled = false;
  }

  // ------------------------------------------------------------ words
  function renderWords() {
    const m = models[$("wModel").value]; if (!m) return;
    const ex = SA.explanatoryWords(m, dict, 25);
    const maxAbs = Math.max(Math.abs(ex.positive[0][1]), Math.abs(ex.negative[0][1]));
    const bars = (list) => list.map(([w, v]) => `<div class="cbar"><span class="w">${escapeHtml(w)}</span><span class="track"><i class="${v > 0 ? "pos" : "neg"}" style="width:${(Math.abs(v) / maxAbs) * 50}%"></i></span><span class="v">${v > 0 ? "+" : ""}${v.toFixed(2)}</span></div>`).join("");
    $("posWords").innerHTML = bars(ex.positive); $("negWords").innerHTML = bars(ex.negative);
  }

  // ------------------------------------------------------------ reviews
  function renderReviews() {
    const m = models.pegasos; if (!m) return;
    const filter = $("rvFilter").value, q = $("rvSearch").value.trim().toLowerCase();
    const items = D.test.map((r, i) => ({ r, i, pred: SA.classify(XT[i], m), score: SA.score(XT[i], m) }))
      .filter((it) => (filter === "wrong" ? it.pred !== it.r.y : filter === "pos" ? it.pred === 1 : filter === "neg" ? it.pred === -1 : true))
      .filter((it) => !q || it.r.t.toLowerCase().includes(q) || it.r.s.toLowerCase().includes(q));
    const wrong = D.test.filter((r, i) => SA.classify(XT[i], m) !== r.y).length;
    $("rvNote").textContent = `Pegasos gets ${500 - wrong} of 500 test reviews right (${pct((500 - wrong) / 500)}). Showing ${items.length}. Click a review to load it into Analyze.`;
    $("reviewList").innerHTML = items.slice(0, 200).map((it) => `<div class="review ${it.pred !== it.r.y ? "wrong" : ""}" data-i="${it.i}"><div class="meta"><span class="pill ${it.pred === 1 ? "pos" : "neg"}">predicted ${it.pred === 1 ? "positive" : "negative"}</span><span>true ${it.r.y === 1 ? "positive" : "negative"}</span><span>score ${it.score.toFixed(2)}</span></div><b>${escapeHtml(it.r.s)}</b> — ${escapeHtml(it.r.t)}</div>`).join("");
    $("reviewList").querySelectorAll(".review").forEach((el) => el.addEventListener("click", () => { $("reviewText").value = D.test[Number(el.dataset.i)].t; switchTab("analyze"); analyze(); }));
  }

  // ------------------------------------------------------------ misc
  function toast(html, ms) { const t = document.createElement("div"); t.className = "toast"; t.innerHTML = html; $("toasts").appendChild(t); setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 300); }, ms || 3200); }
  function switchTab(name) { document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name)); document.querySelectorAll(".tab-body").forEach((b) => b.classList.toggle("active", b.dataset.tab === name)); }
  document.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => b.blur()));
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("reviewText").addEventListener("input", analyze);
  $("anModel").addEventListener("change", analyze);
  $("examples").innerHTML = EXAMPLES.map((e, i) => `<button data-i="${i}">${e.slice(0, 34)}…</button>`).join("");
  $("examples").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { $("reviewText").value = EXAMPLES[Number(b.dataset.i)]; analyze(); }));
  $("btnTrain").addEventListener("click", trainAll);
  $("btnSweep").addEventListener("click", sweep);
  $("wModel").addEventListener("change", renderWords);
  $("rvFilter").addEventListener("change", renderReviews);
  $("rvSearch").addEventListener("input", renderReviews);
  load().catch((e) => toast("Could not load the review data."));
  window.__ra = { models, analyze, trainAll, sweep, get dict() { return dict; } };
})();
