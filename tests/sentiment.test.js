// Run with:  node --test tests/*.test.js
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs"), path = require("path");
const SA = require("../docs/js/sentiment.js");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "docs", "data", "reviews.json"), "utf8"));
const tr = D.train.map((r) => r.t), y = D.train.map((r) => r.y), va = D.val.map((r) => r.t), yv = D.val.map((r) => r.y);

test("extract_words splits punctuation and digits like the Python version", () => {
  assert.deepEqual(SA.extractWords("Good, but 2 stars!"), ["good", ",", "but", "2", "stars", "!"]);
});
test("bag of words indexes every word once; stopwords removable; bigrams added", () => {
  const d = SA.bagOfWords(["a b a", "b c"]); assert.equal(d.size, 3); assert.equal(d.get("a"), 0);
  assert.equal(SA.bagOfWords(["a b a", "b c"], { stopwords: new Set(["a"]) }).size, 2);
  assert.ok(SA.bagOfWords(["a b"], { bigrams: true }).has("a_b"));
});
test("features are binary by default, counts on request", () => {
  const d = SA.bagOfWords(["a b a"]); assert.equal(SA.features("a a a", d).get(0), 1); assert.equal(SA.features("a a a", d, { binarize: false }).get(0), 3);
});
test("perceptron converges to 100% on a separable toy set", () => {
  const d = new Map([["good", 0], ["bad", 1]]);
  const X = [new Map([[0, 1]]), new Map([[1, 1]]), new Map([[0, 1]]), new Map([[1, 1]])], yy = [1, -1, 1, -1];
  for (const kind of ["perceptron", "average", "pegasos"]) assert.equal(SA.accuracy(X, yy, SA.train(kind, X, yy, 2, { T: 10, L: 0.01 })), 1, kind);
});
test("all three reach ~80% validation accuracy on the review data (course result)", () => {
  const dict = SA.bagOfWords(tr); const X = SA.featurize(tr, dict), XV = SA.featurize(va, dict);
  for (const kind of ["perceptron", "average", "pegasos"]) {
    const m = SA.train(kind, X, y, dict.size, { T: 25, L: 0.01, order: D.order4000 });
    const acc = SA.accuracy(XV, yv, m); assert.ok(acc > 0.77 && acc < 0.9, `${kind}: ${acc}`);
  }
});
test("lazy average perceptron equals the explicit running average", () => {
  const dict = SA.bagOfWords(tr.slice(0, 300)); const X = SA.featurize(tr.slice(0, 300), dict); const yy = y.slice(0, 300);
  const lazy = SA.train("average", X, yy, dict.size, { T: 3, order: Array.from({ length: 300 }, (_, i) => i) });
  // explicit
  const theta = new Float64Array(dict.size); let theta0 = 0; const sum = new Float64Array(dict.size); let sum0 = 0, n = 0;
  for (let t = 0; t < 3; t++) for (let i = 0; i < 300; i++) { let s = theta0; for (const [k, v] of X[i]) s += theta[k] * v; if (yy[i] * s < 1e-4) { for (const [k, v] of X[i]) theta[k] += yy[i] * v; theta0 += yy[i]; } for (let k = 0; k < dict.size; k++) sum[k] += theta[k]; sum0 += theta0; n++; }
  for (let k = 0; k < dict.size; k++) assert.ok(Math.abs(sum[k] / n - lazy.theta[k]) < 1e-9);
  assert.ok(Math.abs(sum0 / n - lazy.theta0) < 1e-9);
});
test("most explanatory words make sense", () => {
  const dict = SA.bagOfWords(tr); const X = SA.featurize(tr, dict); const m = SA.train("pegasos", X, y, dict.size, { T: 25, L: 0.01, order: D.order4000 });
  const ex = SA.explanatoryWords(m, dict, 10);
  assert.ok(ex.positive.map((w) => w[0]).includes("delicious")); assert.ok(ex.negative.map((w) => w[0]).includes("disappointed"));
});
