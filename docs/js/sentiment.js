/*
 * Sentiment classifiers from scratch — port of project1.py.
 * Perceptron, average perceptron and Pegasos on sparse bag-of-words features.
 * No DOM. Browser global (window.SA) + CommonJS module.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.SA = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789";

  /** extract_words(): lowercase words, punctuation and digits as their own tokens. */
  function extractWords(text) {
    let out = "";
    for (const c of text) out += PUNCT.includes(c) ? " " + c + " " : c;
    return out.toLowerCase().split(/\s+/).filter(Boolean);
  }

  /** bag_of_words(): word → index over the training texts. Options: stopwords (Set), bigrams. */
  function bagOfWords(texts, o) {
    o = o || {};
    const stop = o.stopwords || new Set();
    const dict = new Map();
    for (const t of texts) for (const w of tokens(t, o)) if (!dict.has(w) && !stop.has(w)) dict.set(w, dict.size);
    return dict;
  }
  function tokens(text, o) {
    const words = extractWords(text);
    if (!o || !o.bigrams) return words;
    const out = words.slice();
    for (let i = 0; i + 1 < words.length; i++) out.push(words[i] + "_" + words[i + 1]);
    return out;
  }

  /** Sparse feature vector: Map(index → count or 1). */
  function features(text, dict, o) {
    o = o || {};
    const f = new Map();
    for (const w of tokens(text, o)) { const i = dict.get(w); if (i === undefined) continue; f.set(i, o.binarize === false ? (f.get(i) || 0) + 1 : 1); }
    return f;
  }
  function featurize(texts, dict, o) { return texts.map((t) => features(t, dict, o)); }

  const dot = (theta, f) => { let s = 0; for (const [i, v] of f) s += theta[i] * v; return s; };

  // ------------------------------------------------------------ the three algorithms (single steps as in project1.py)
  function perceptronStep(f, y, theta, theta0) {
    if (y * (dot(theta, f) + theta0) < 1e-4) { for (const [i, v] of f) theta[i] += y * v; theta0 += y; }
    return theta0;
  }
  function pegasosStep(f, y, L, eta, theta, theta0, scaleState) {
    // theta ← (1 − ηλ)θ (+ ηy x if margin ≤ 1). The shrink is applied lazily through a global scale for speed.
    const z = y * (dot(theta, f) * scaleState.s + theta0);
    scaleState.s *= 1 - eta * L;
    if (z <= 1) { for (const [i, v] of f) theta[i] += (eta * y * v) / scaleState.s; theta0 += eta * y; }
    return theta0;
  }

  /**
   * Train. kind: "perceptron" | "average" | "pegasos". Returns { theta: Float64Array, theta0 }.
   * opts: { T, L, order (array of indices, or null for a seeded shuffle), onEpoch(t, theta, theta0) }
   */
  function train(kind, X, y, dim, opts) {
    const o = Object.assign({ T: 10, L: 0.01 }, opts);
    const n = X.length;
    const order = o.order || shuffled(n, 1);
    let theta = new Float64Array(dim), theta0 = 0;
    // average perceptron via the lazy trick: θ̄ = θ − u / N where u = Σ (k−1)·Δ_k over updates made at step k
    const u = new Float64Array(dim); let u0 = 0, count = 0;
    const scale = { s: 1 }; let step = 1;
    for (let t = 0; t < o.T; t++) {
      for (const i of order) {
        if (kind === "pegasos") { theta0 = pegasosStep(X[i], y[i], o.L, 1 / Math.sqrt(step), theta, theta0, scale); step++; }
        else if (kind === "average") {
          count++;
          if (y[i] * (dot(theta, X[i]) + theta0) < 1e-4) { for (const [k, v] of X[i]) { theta[k] += y[i] * v; u[k] += (count - 1) * y[i] * v; } theta0 += y[i]; u0 += (count - 1) * y[i]; }
        } else theta0 = perceptronStep(X[i], y[i], theta, theta0);
      }
      if (o.onEpoch) o.onEpoch(t + 1, current());
    }
    function current() {
      if (kind === "average") { const th = new Float64Array(dim); for (let k = 0; k < dim; k++) th[k] = theta[k] - u[k] / count; return { theta: th, theta0: theta0 - u0 / count }; }
      if (kind === "pegasos") { const th = new Float64Array(dim); for (let k = 0; k < dim; k++) th[k] = theta[k] * scale.s; return { theta: th, theta0 }; }
      return { theta: Float64Array.from(theta), theta0 };
    }
    return current();
  }

  function shuffled(n, seed) { let s = seed >>> 0; const rng = () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; const a = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

  const score = (f, model) => dot(model.theta, f) + model.theta0;
  const classify = (f, model) => (score(f, model) > 1e-5 ? 1 : -1);
  function accuracy(X, y, model) { let c = 0; for (let i = 0; i < X.length; i++) if (classify(X[i], model) === y[i]) c++; return X.length ? c / X.length : 0; }
  function hingeLoss(X, y, model) { let s = 0; for (let i = 0; i < X.length; i++) s += Math.max(0, 1 - y[i] * score(X[i], model)); return X.length ? s / X.length : 0; }

  /** most_explanatory_word(): words sorted by weight (descending). */
  function explanatoryWords(model, dict, n) {
    const words = new Array(dict.size); dict.forEach((i, w) => { words[i] = w; });
    const idx = Array.from(words.keys()).sort((a, b) => model.theta[b] - model.theta[a]);
    return { positive: idx.slice(0, n).map((i) => [words[i], model.theta[i]]), negative: idx.slice(-n).reverse().map((i) => [words[i], model.theta[i]]) };
  }

  /** Per-token contribution for highlighting a review. */
  function explain(text, dict, model, o) {
    const words = extractWords(text);
    const seen = new Set();
    return words.map((w) => {
      const i = dict.get(w);
      let c = 0;
      if (i !== undefined) { c = model.theta[i]; if ((!o || o.binarize !== false) && seen.has(w)) c = 0; seen.add(w); }
      return { word: w, weight: c };
    });
  }

  return { extractWords, tokens, bagOfWords, features, featurize, train, score, classify, accuracy, hingeLoss, explanatoryWords, explain, shuffled };
});
