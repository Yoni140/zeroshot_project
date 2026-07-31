const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.title = 'Zero-Shot vs Fine-Tuned LLMs for Misinformation Detection (3-Class)';

// ── Color Palette (matches existing project deck) ──────────────────────────────
const C = {
  darkBg: '0D1B2A', navy: '1A3A5C', teal: '00B4D8', tealDark: '0077A8',
  tealLight: 'E0F7FF', lightBg: 'F4F6FA', white: 'FFFFFF',
  textDark: '1A2B4A', textBody: '2D3748', orange: 'F4A261',
  green: '2DC653', red: 'E63946', purple: '6A0572', subtle: '90A4AE',
  rob: '2196F3', zs: 'FF9800',
};
const BASE = 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/results/figures';
const W = 10, H = 5.625;

// ── Helpers ───────────────────────────────────────────────────────────────────
function addHeaderBar(slide, title, subtitle) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 1.15, fill: { color: C.darkBg } });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 1.15, fill: { color: C.teal } });
  slide.addText(title, { x: 0.28, y: 0.1, w: 9.5, h: 0.6, fontSize: 23, bold: true, color: C.white, fontFace: 'Calibri', margin: 0, valign: 'middle' });
  if (subtitle) slide.addText(subtitle, { x: 0.28, y: 0.74, w: 9.5, h: 0.34, fontSize: 13, color: C.teal, fontFace: 'Calibri', margin: 0, italic: true });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.15, w: W, h: 0.04, fill: { color: C.teal } });
}
function addCard(slide, x, y, w, h, fillColor) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fillColor || C.white },
    line: { color: 'E2E8F0', width: 0.75 },
    shadow: { type: 'outer', color: '000000', blur: 4, offset: 1, angle: 135, opacity: 0.08 } });
}
// place an image fit (contain) inside a box, centered, preserving aspect ratio
function fitImg(slide, path, bx, by, bw, bh, ratio) {
  let w = bw, h = w / ratio;
  if (h > bh) { h = bh; w = h * ratio; }
  slide.addImage({ path, x: bx + (bw - w) / 2, y: by + (bh - h) / 2, w, h });
}
function caption(slide, text, x, y, w, color) {
  slide.addText(text, { x, y, w, h: 0.28, fontSize: 11, bold: true, color: color || C.textDark, fontFace: 'Calibri', margin: 0, align: 'center' });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: H, fill: { color: C.darkBg } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: H, fill: { color: C.teal } });
  s.addShape(pres.shapes.OVAL, { x: 7.0, y: -0.9, w: 4.6, h: 4.6, fill: { color: C.tealDark, transparency: 78 }, line: { color: C.tealDark, transparency: 78 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.78, w: W, h: 0.85, fill: { color: C.navy } });

  s.addText([
    { text: 'Detecting Misinformation on Twitter', options: { breakLine: true } },
    { text: 'Fine-Tuned RoBERTa vs Zero-Shot LLM', options: {} },
  ], { x: 0.5, y: 0.7, w: 9, h: 1.5, fontSize: 33, bold: true, color: C.white, fontFace: 'Calibri', margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.35, w: 5.2, h: 0.06, fill: { color: C.teal } });
  s.addText('A 3-Class Study:  reliable  ·  misinformation  ·  unrelated', {
    x: 0.5, y: 2.55, w: 9, h: 0.5, fontSize: 17, color: 'A8C8E8', fontFace: 'Calibri', margin: 0 });

  const badges = [
    { label: 'Manchester Arena 2017', x: 0.5 },
    { label: 'Monkeypox 2022', x: 3.5 },
    { label: 'PHEME 2015', x: 6.0 },
  ];
  badges.forEach(b => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: b.x, y: 3.55, w: 2.7, h: 0.45, fill: { color: C.tealDark, transparency: 20 }, rectRadius: 0.08, line: { color: C.teal, width: 1 } });
    s.addText(b.label, { x: b.x, y: 3.55, w: 2.7, h: 0.45, fontSize: 12, color: C.white, fontFace: 'Calibri', align: 'center', valign: 'middle', margin: 0 });
  });
  s.addText('Final Project  ·  Yoni  ·  2026', { x: 0.5, y: 4.95, w: 9, h: 0.4, fontSize: 12, color: C.subtle, fontFace: 'Calibri', margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Problem & the 3-class scheme
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'The Problem & Why Three Classes', 'From a binary task to a realistic three-way decision');

  addCard(s, 0.4, 1.35, 9.2, 1.15, 'EBF5FB');
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.35, w: 0.12, h: 1.15, fill: { color: C.teal } });
  s.addText('Can a large language model with NO training examples detect misinformation as well as a RoBERTa model fine-tuned on thousands of labeled tweets — now across three classes?', {
    x: 0.65, y: 1.4, w: 8.8, h: 1.05, fontSize: 15, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0, valign: 'middle' });

  const cards = [
    { t: 'reliable', d: 'Factually accurate, verified, or plausible information about the event.', col: C.green },
    { t: 'misinformation', d: 'False, unverified, or misleading claims — rumours, conspiracy theories, fabrications.', col: C.red },
    { t: 'unrelated', d: 'The tweet matched the search keywords but is not actually about the event — noise.', col: C.subtle },
  ];
  cards.forEach((c, i) => {
    const x = 0.4 + i * 3.1;
    addCard(s, x, 2.7, 2.9, 2.55);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 2.7, w: 2.9, h: 0.5, fill: { color: c.col } });
    s.addText(c.t, { x: x + 0.12, y: 2.7, w: 2.66, h: 0.5, fontSize: 15, bold: true, color: C.white, fontFace: 'Calibri', margin: 0, valign: 'middle' });
    s.addText(c.d, { x: x + 0.14, y: 3.35, w: 2.62, h: 1.4, fontSize: 12.5, color: C.textBody, fontFace: 'Calibri', margin: 0 });
    if (c.t === 'unrelated') {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 1.95, y: 2.78, w: 0.82, h: 0.34, fill: { color: C.white }, rectRadius: 0.04, line: { color: C.white } });
      s.addText('NEW', { x: x + 1.95, y: 2.78, w: 0.82, h: 0.34, fontSize: 10, bold: true, color: C.subtle, align: 'center', valign: 'middle', margin: 0 });
    }
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Datasets
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'Three Datasets', 'Real-world misinformation across three domains');

  const ds = [
    { name: 'Manchester Arena', col: C.tealDark, rows: [['Total tweets', '89,147'], ['Gold standard', '3,427'], ['Classes', 'reliable / misinfo / unrelated'], ['Domain', 'Terror attack (2017)']] },
    { name: 'Monkeypox', col: C.navy, rows: [['Total tweets', '~6,287'], ['Gold standard', '4,043'], ['Classes', 'reliable / misinfo / unrelated'], ['Domain', 'Health crisis (2022)']] },
    { name: 'PHEME', col: C.purple, rows: [['Total tweets', '62,445'], ['Gold standard', '4,500'], ['Classes', 'not_rumour / rumour / unrelated'], ['Domain', 'Breaking news (2015)']] },
  ];
  ds.forEach((d, i) => {
    const x = 0.28 + i * 3.25;
    addCard(s, x, 1.35, 3.1, 3.0);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.35, w: 3.1, h: 0.5, fill: { color: d.col } });
    s.addText(d.name, { x: x + 0.1, y: 1.35, w: 2.9, h: 0.5, fontSize: 15, bold: true, color: C.white, fontFace: 'Calibri', margin: 0, valign: 'middle' });
    d.rows.forEach((r, ri) => {
      const ry = 2.0 + ri * 0.56;
      s.addText(r[0], { x: x + 0.14, y: ry, w: 1.2, h: 0.5, fontSize: 10.5, color: C.subtle, fontFace: 'Calibri', margin: 0, valign: 'middle' });
      s.addText(r[1], { x: x + 1.25, y: ry, w: 1.75, h: 0.5, fontSize: 11, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0, valign: 'middle' });
    });
  });
  addCard(s, 0.28, 4.55, 9.42, 0.78, 'EBF5FB');
  s.addText([
    { text: 'Balanced gold standard:  ', options: { bold: true, color: C.tealDark } },
    { text: 'up to 1,500 tweets per class, stratified 70 / 15 / 15 into train / validation / test. Test sets are near-balanced across the three classes.', options: { color: C.textBody } },
  ], { x: 0.45, y: 4.6, w: 9.1, h: 0.68, fontSize: 11.5, fontFace: 'Calibri', margin: 0, valign: 'middle' });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Where the 'unrelated' class comes from
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, "Building the 'unrelated' Class", 'A relevance filter, tailored to each dataset');

  const cards = [
    { name: 'Manchester', icon: '🔎', method: 'Keyword relevance filter', body: 'A "reliable" tweet that mentions NONE of the event keywords (arena, bomb, Ariana, Abedi, …) is re-labelled unrelated — plus the 3 native "Not related" tweets.', n: '≈ 2,539 unrelated', col: C.tealDark },
    { name: 'Monkeypox', icon: '🏷️', method: 'Native ternary label', body: 'The dataset already ships a 3-way annotation (ternary_class). Class 9 — generic / off-topic tweets — is mapped directly to unrelated.', n: '≈ 2,647 unrelated', col: C.navy },
    { name: 'PHEME', icon: '🗂️', method: 'Topic-based filter', body: "Tweets whose topic is 'unknown' (not tied to any tracked event, Charlie Hebdo / Ferguson) become unrelated, regardless of rumour status.", n: '≈ 9,589 unrelated', col: C.purple },
  ];
  cards.forEach((c, i) => {
    const x = 0.28 + i * 3.25;
    addCard(s, x, 1.35, 3.1, 3.95);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.35, w: 3.1, h: 0.52, fill: { color: c.col } });
    s.addText(c.icon + '  ' + c.name, { x: x + 0.12, y: 1.35, w: 2.86, h: 0.52, fontSize: 14, bold: true, color: C.white, fontFace: 'Calibri', margin: 0, valign: 'middle' });
    s.addText(c.method, { x: x + 0.14, y: 2.0, w: 2.82, h: 0.5, fontSize: 12.5, bold: true, color: c.col, fontFace: 'Calibri', margin: 0 });
    s.addText(c.body, { x: x + 0.14, y: 2.52, w: 2.82, h: 2.0, fontSize: 11.5, color: C.textBody, fontFace: 'Calibri', margin: 0 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.14, y: 4.78, w: 2.82, h: 0.4, fill: { color: 'EBF5FB' }, rectRadius: 0.05, line: { color: c.col, width: 1 } });
    s.addText(c.n, { x: x + 0.14, y: 4.78, w: 2.82, h: 0.4, fontSize: 11, bold: true, color: c.col, fontFace: 'Calibri', align: 'center', valign: 'middle', margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Methodology (two approaches)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'Two Approaches', 'Supervised fine-tuning vs zero-shot prompting');

  // RoBERTa column
  addCard(s, 0.35, 1.35, 4.5, 3.95);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.35, y: 1.35, w: 4.5, h: 0.55, fill: { color: C.rob } });
  s.addText('RoBERTa  ·  Fine-Tuned', { x: 0.5, y: 1.35, w: 4.2, h: 0.55, fontSize: 15, bold: true, color: C.white, fontFace: 'Calibri', margin: 0, valign: 'middle' });
  [
    ['Model', 'roberta-base (125M params)'],
    ['Training', '5-fold stratified cross-validation'],
    ['Imbalance', 'Weighted cross-entropy loss'],
    ['Hyper-params', 'LR 2e-5 · batch 16 · ≤4 epochs'],
    ['Regularization', 'Early stopping (patience 2)'],
    ['Hardware', 'Local GPU (RTX 4070 SUPER), fp16'],
  ].forEach((r, i) => {
    const y = 2.05 + i * 0.53;
    s.addText(r[0], { x: 0.5, y, w: 1.5, h: 0.5, fontSize: 11, bold: true, color: C.rob, fontFace: 'Calibri', margin: 0, valign: 'middle' });
    s.addText(r[1], { x: 2.0, y, w: 2.75, h: 0.5, fontSize: 11, color: C.textBody, fontFace: 'Calibri', margin: 0, valign: 'middle' });
  });

  // Zero-shot column
  addCard(s, 5.15, 1.35, 4.5, 3.95);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: 1.35, w: 4.5, h: 0.55, fill: { color: C.zs } });
  s.addText('Llama 3.1 8B  ·  Zero-Shot', { x: 5.3, y: 1.35, w: 4.2, h: 0.55, fontSize: 15, bold: true, color: C.white, fontFace: 'Calibri', margin: 0, valign: 'middle' });
  [
    ['Runtime', 'Ollama — fully local, no API / internet'],
    ['Training', 'None — zero labeled examples'],
    ['Prompt', 'Chain-of-Thought, 3 class definitions'],
    ['Output', 'Structured JSON: label + reasoning'],
    ['Decoding', 'temperature 0 (deterministic)'],
    ['Robustness', 'Retry + keyword-fallback parser'],
  ].forEach((r, i) => {
    const y = 2.05 + i * 0.53;
    s.addText(r[0], { x: 5.3, y, w: 1.5, h: 0.5, fontSize: 11, bold: true, color: 'C77B22', fontFace: 'Calibri', margin: 0, valign: 'middle' });
    s.addText(r[1], { x: 6.8, y, w: 2.75, h: 0.5, fontSize: 11, color: C.textBody, fontFace: 'Calibri', margin: 0, valign: 'middle' });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — RoBERTa confusion matrices (3 datasets)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'RoBERTa — Test Confusion Matrices', 'Strong diagonal: the fine-tuned model rarely confuses classes');
  const items = [
    { p: `${BASE}/manchester/manchester_roberta_test_cm.png`, t: 'Manchester  (F1 0.965)' },
    { p: `${BASE}/monkeypox/monkeypox_roberta_test_cm.png`, t: 'Monkeypox  (F1 0.869)' },
    { p: `${BASE}/pheme/pheme_roberta_test_cm.png`, t: 'PHEME  (F1 0.658)' },
  ];
  items.forEach((it, i) => {
    const x = 0.3 + i * 3.25;
    caption(s, it.t, x, 1.35, 3.1, C.rob);
    fitImg(s, it.p, x, 1.7, 3.1, 3.0, 1.161);
  });
  addCard(s, 0.3, 4.85, 9.4, 0.55, 'EBF5FB');
  s.addText('Manchester & Monkeypox are cleanly separated; PHEME stays hardest — rumour vs not_rumour needs event context a single tweet lacks.', {
    x: 0.45, y: 4.88, w: 9.1, h: 0.5, fontSize: 11, italic: true, color: C.tealDark, fontFace: 'Calibri', margin: 0, valign: 'middle' });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — RoBERTa cross-validation (all datasets)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'RoBERTa — 5-Fold Cross-Validation', 'OOF confusion matrix + per-fold F1 (low variance = stable model)');
  const rows = [
    { p: `${BASE}/manchester/manchester_roberta_cv_results.png`, t: 'Manchester', col: C.tealDark },
    { p: `${BASE}/monkeypox/monkeypox_roberta_cv_results.png`, t: 'Monkeypox', col: C.navy },
    { p: `${BASE}/pheme/pheme_roberta_cv_results.png`, t: 'PHEME', col: C.purple },
  ];
  rows.forEach((r, i) => {
    const y = 1.32 + i * 1.38;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: y + 0.45, w: 1.25, h: 0.45, fill: { color: r.col } });
    s.addText(r.t, { x: 0.3, y: y + 0.45, w: 1.25, h: 0.45, fontSize: 11, bold: true, color: C.white, fontFace: 'Calibri', align: 'center', valign: 'middle', margin: 0 });
    fitImg(s, r.p, 1.7, y, 8.0, 1.34, 2.85);
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — RoBERTa ROC / PR + heatmap (Manchester)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'RoBERTa — ROC / PR & Per-Fold Metrics (Manchester)', 'One-vs-rest curves for the 3-class problem');
  caption(s, 'One-vs-Rest ROC & Precision–Recall  (macro ROC-AUC 0.996, PR-AUC 0.990)', 0.3, 1.32, 9.4, C.rob);
  fitImg(s, `${BASE}/manchester/manchester_roberta_roc_pr_curves.png`, 0.3, 1.62, 9.4, 2.0, 2.851);
  caption(s, 'Per-Fold Metrics Heatmap (5-fold CV)', 0.3, 3.72, 9.4, C.rob);
  fitImg(s, `${BASE}/manchester/manchester_roberta_cv_heatmap.png`, 1.5, 4.0, 7.0, 1.45, 2.408);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — RoBERTa results table (all datasets)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'RoBERTa — Test Results (all datasets)', 'Supervised upper bound for the 3-class task');
  const rows = [
    ['Dataset', 'CV F1 (mean ± std)', 'Test F1 Macro', 'Accuracy', 'Precision', 'Recall', 'F1 Weighted'],
    ['Manchester', '0.965 ± 0.007', '0.965', '0.977', '0.970', '0.960', '0.977'],
    ['Monkeypox', '0.859 ± 0.005', '0.869', '0.873', '0.868', '0.870', '0.872'],
    ['PHEME', '0.633 ± 0.016', '0.658', '0.656', '0.660', '0.656', '0.658'],
  ];
  const cw = [1.55, 2.05, 1.3, 1.15, 1.15, 1.1, 1.2];
  let ry = 1.75;
  rows.forEach((row, ri) => {
    let cx = 0.35; const isH = ri === 0;
    row.forEach((cell, ci) => {
      s.addShape(pres.shapes.RECTANGLE, { x: cx, y: ry, w: cw[ci], h: 0.5,
        fill: { color: isH ? C.darkBg : (ri % 2 === 0 ? 'F0F4F8' : C.white) }, line: { color: 'E2E8F0', width: 0.5 } });
      s.addText(cell, { x: cx + 0.04, y: ry + 0.05, w: cw[ci] - 0.08, h: 0.4,
        fontSize: isH ? 10.5 : 12, bold: isH || ci === 2, color: isH ? C.white : (ci === 2 ? C.rob : C.textDark),
        fontFace: 'Calibri', align: 'center', valign: 'middle', margin: 0 });
      cx += cw[ci];
    });
    ry += 0.5;
  });
  addCard(s, 0.35, 3.9, 9.3, 1.4, 'EBF5FB');
  s.addText('Why it works', { x: 0.5, y: 3.97, w: 9.0, h: 0.3, fontSize: 13, bold: true, color: C.tealDark, fontFace: 'Calibri', margin: 0 });
  s.addText([
    { text: '•  CV mean ≈ Test score for every dataset → no overfitting (e.g. Manchester 0.965 CV vs 0.965 Test).\n', options: { breakLine: true } },
    { text: '•  Weighted cross-entropy handles the smaller misinformation class (Manchester has only 64 test cases).\n', options: { breakLine: true } },
    { text: '•  PHEME is intrinsically hard: rumour detection on isolated tweets caps accuracy near 0.66.', options: {} },
  ], { x: 0.5, y: 4.27, w: 9.0, h: 0.95, fontSize: 11, color: C.textBody, fontFace: 'Calibri', margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Zero-Shot: how it works + prompt
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'Zero-Shot — Llama 3.1 8B via Ollama', 'No training examples — only a well-engineered prompt');

  addCard(s, 0.35, 1.32, 4.5, 4.05);
  s.addText('How it works', { x: 0.5, y: 1.42, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: 'C77B22', fontFace: 'Calibri', margin: 0 });
  const steps = [
    'Ollama runs Llama 3.1 8B locally (no key, no internet)',
    'Build a Chain-of-Thought prompt with the 3 class definitions',
    'Model first checks: is the tweet on-topic at all?',
    'Reasons step-by-step, then emits structured JSON',
    'Parse JSON → label; keyword-fallback if malformed',
    'Checkpoint every 50 tweets (resume-safe)',
  ];
  steps.forEach((t, i) => {
    const y = 1.85 + i * 0.56;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y, w: 0.4, h: 0.4, fill: { color: C.zs }, line: { color: C.zs } });
    s.addText(String(i + 1), { x: 0.5, y, w: 0.4, h: 0.4, fontSize: 12, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0 });
    s.addText(t, { x: 1.0, y: y - 0.04, w: 3.75, h: 0.48, fontSize: 11, color: C.textBody, fontFace: 'Calibri', margin: 0, valign: 'middle' });
  });

  addCard(s, 5.05, 1.32, 4.6, 4.05, '1A1A2E');
  s.addText('Prompt (Chain-of-Thought, 3-class)', { x: 5.18, y: 1.4, w: 4.4, h: 0.3, fontSize: 11, bold: true, color: C.teal, fontFace: 'Calibri', margin: 0 });
  const pl = [
    { text: 'You are an expert fact-checker...\n', color: C.orange },
    { text: 'Classify this tweet about <topic>.\n\n', color: C.teal },
    { text: 'CLASSES:\n', color: 'FFFFFF' },
    { text: '  "reliable": accurate info...\n', color: '90EE90' },
    { text: '  "misinformation": false claims...\n', color: 'FF8080' },
    { text: '  "unrelated": not about the event...\n\n', color: 'C0C8D0' },
    { text: 'TWEET: """<text>"""\n\n', color: 'FFD700' },
    { text: 'Think step-by-step:\n', color: 'FFFFFF' },
    { text: ' 1. Is it on-topic, or unrelated?\n', color: C.subtle },
    { text: ' 2. What claim is made?\n', color: C.subtle },
    { text: ' 3. Verifiable, or conspiracy signals?\n\n', color: C.subtle },
    { text: 'Output JSON:\n', color: 'FFFFFF' },
    { text: '{"reasoning":"...",\n "label":"unrelated",\n "confidence":0.83}', color: '90EE90' },
  ];
  s.addText(pl.map(l => ({ text: l.text, options: { color: l.color } })),
    { x: 5.16, y: 1.74, w: 4.4, h: 3.5, fontSize: 9.5, fontFace: 'Consolas', margin: 0.05, valign: 'top' });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 11 — Zero-Shot confusion matrices (3 datasets)
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'Zero-Shot — Confusion Matrices', 'The LLM uses all three labels, but confuses them far more');
  const items = [
    { p: `${BASE}/manchester/manchester_zeroshot_cm.png`, t: 'Manchester  (F1 0.651)' },
    { p: `${BASE}/monkeypox/monkeypox_zeroshot_cm.png`, t: 'Monkeypox  (F1 0.644)' },
    { p: `${BASE}/pheme/pheme_zeroshot_cm.png`, t: 'PHEME  (F1 0.320)' },
  ];
  items.forEach((it, i) => {
    const y = 1.3 + i * 1.36;
    caption(s, it.t, 0.3, y, 2.2, C.zs);
    fitImg(s, it.p, 2.4, y, 7.4, 1.32, 2.76);
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 12 — Zero-Shot confidence analysis
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'Zero-Shot — Confidence & Parse Quality', 'Self-reported confidence vs correctness (Manchester)');
  fitImg(s, `${BASE}/manchester/manchester_zeroshot_confidence.png`, 0.3, 1.35, 6.2, 3.6, 2.851);
  addCard(s, 6.7, 1.35, 3.0, 3.85);
  s.addText('Output reliability', { x: 6.85, y: 1.45, w: 2.7, h: 0.32, fontSize: 13, bold: true, color: 'C77B22', fontFace: 'Calibri', margin: 0 });
  [
    ['JSON parsed', '513 / 515'],
    ['Null predictions', '1'],
    ['Keyword fallback', '~1'],
    ['Mean confidence', 'high & overconfident'],
  ].forEach((r, i) => {
    const y = 1.95 + i * 0.62;
    s.addText(r[0], { x: 6.85, y, w: 2.7, h: 0.28, fontSize: 11, color: C.subtle, fontFace: 'Calibri', margin: 0 });
    s.addText(r[1], { x: 6.85, y: y + 0.24, w: 2.7, h: 0.3, fontSize: 13, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0 });
  });
  s.addText('The model is confident even when wrong — confidence is not a reliable error signal.', {
    x: 6.85, y: 4.5, w: 2.7, h: 0.65, fontSize: 10.5, italic: true, color: C.tealDark, fontFace: 'Calibri', margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 13 — Comparison grouped bars
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'Head-to-Head — F1 Macro by Dataset', 'RoBERTa fine-tuned vs Llama 3.1 zero-shot');
  fitImg(s, `${BASE}/comparison/comparison_grouped_bars.png`, 0.3, 1.35, 9.4, 3.05, 3.015);
  addCard(s, 0.3, 4.55, 9.4, 0.78, 'EBF5FB');
  s.addText([
    { text: 'RoBERTa wins every dataset:  ', options: { bold: true, color: C.tealDark } },
    { text: 'Δ F1 = +0.314 (Manchester), +0.225 (Monkeypox), +0.338 (PHEME). Error bars = CV std.', options: { color: C.textBody } },
  ], { x: 0.45, y: 4.58, w: 9.1, h: 0.7, fontSize: 12, fontFace: 'Calibri', margin: 0, valign: 'middle' });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 14 — Comparison confusion matrices
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'Confusion Matrices — Side by Side', 'Top: RoBERTa  ·  Bottom: Zero-Shot  (all three datasets)');
  fitImg(s, `${BASE}/comparison/comparison_confusion_matrices.png`, 0.3, 1.3, 9.4, 4.05, 1.666);
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 15 — Comparison heatmap
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'Performance Heatmap — All Metrics', 'Every metric, both models, all datasets');
  fitImg(s, `${BASE}/comparison/comparison_heatmap.png`, 0.3, 1.32, 6.4, 3.95, 1.884);
  addCard(s, 6.85, 1.32, 2.85, 3.95);
  s.addText('Reading it', { x: 7.0, y: 1.42, w: 2.55, h: 0.3, fontSize: 13, bold: true, color: C.tealDark, fontFace: 'Calibri', margin: 0 });
  s.addText([
    { text: 'Darker = higher score.\n\n', options: { breakLine: true } },
    { text: 'The RoBERTa rows stay dark across every metric; the zero-shot rows fade — most sharply on PHEME.\n\n', options: { breakLine: true } },
    { text: 'The gap is consistent: it is not driven by a single metric.', options: {} },
  ], { x: 7.0, y: 1.8, w: 2.55, h: 3.3, fontSize: 11.5, color: C.textBody, fontFace: 'Calibri', margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 16 — Comparison PR curves
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'Precision–Recall Curves', 'Positive class per dataset (misinformation / rumour)');
  fitImg(s, `${BASE}/comparison/comparison_pr_curves.png`, 0.3, 1.35, 9.4, 3.05, 3.019);
  addCard(s, 0.3, 4.55, 9.4, 0.78, 'EBF5FB');
  s.addText('RoBERTa keeps high precision deep into the recall range; the zero-shot curve collapses toward the baseline, confirming the F1 gap.', {
    x: 0.45, y: 4.58, w: 9.1, h: 0.7, fontSize: 11.5, italic: true, color: C.tealDark, fontFace: 'Calibri', margin: 0, valign: 'middle' });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 17 — Summary table + McNemar
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addHeaderBar(s, 'Summary Table & Statistical Significance', 'Final 3-class results — RoBERTa vs zero-shot');
  fitImg(s, `${BASE}/comparison/comparison_final_table.png`, 0.3, 1.32, 9.4, 2.5, 2.929);
  const mc = [
    { d: 'Manchester', p: 'p < 0.0001', b: 'b=160, c=8' },
    { d: 'Monkeypox', p: 'p < 0.0001', b: 'b=164, c=27' },
    { d: 'PHEME', p: 'p < 0.0001', b: 'b=296, c=69' },
  ];
  s.addText("McNemar's test (RoBERTa vs Zero-Shot)", { x: 0.3, y: 3.95, w: 9.4, h: 0.3, fontSize: 13, bold: true, color: C.tealDark, fontFace: 'Calibri', margin: 0 });
  mc.forEach((m, i) => {
    const x = 0.3 + i * 3.2;
    addCard(s, x, 4.3, 3.05, 0.95);
    s.addText(m.d, { x: x + 0.12, y: 4.36, w: 2.8, h: 0.3, fontSize: 12, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0 });
    s.addText([{ text: m.p + '  ', options: { bold: true, color: C.green } }, { text: '· significant', options: { color: C.textBody } }], { x: x + 0.12, y: 4.66, w: 2.8, h: 0.28, fontSize: 11, fontFace: 'Calibri', margin: 0 });
    s.addText(m.b + '  (RoBERTa-only-correct vs ZS-only-correct)', { x: x + 0.12, y: 4.93, w: 2.8, h: 0.28, fontSize: 9, color: C.subtle, fontFace: 'Calibri', margin: 0 });
  });
}

// ════════════════════════════════════════════════════════════════════════════
// SLIDE 18 — Conclusions
// ════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: H, fill: { color: C.darkBg } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: H, fill: { color: C.teal } });
  s.addText('Conclusions', { x: 0.45, y: 0.3, w: 9.2, h: 0.6, fontSize: 30, bold: true, color: C.white, fontFace: 'Calibri', margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y: 0.95, w: 5.5, h: 0.05, fill: { color: C.teal } });

  const points = [
    ['Fine-tuning wins — decisively.', 'RoBERTa beats zero-shot on all 3 datasets (Δ F1 +0.22 to +0.34), significant at p < 0.0001.'],
    ['Zero-shot is still respectable.', 'With zero training, Llama 3.1 8B reaches F1 ≈ 0.65 on Manchester & Monkeypox — usable as a cold-start baseline.'],
    ['The third class is learnable.', 'Both models use all three labels; RoBERTa separates "unrelated" almost perfectly, the LLM only partially.'],
    ['PHEME remains the hard case.', 'Rumour detection on isolated tweets caps both models — context beyond a single tweet is needed.'],
  ];
  points.forEach((p, i) => {
    const y = 1.2 + i * 1.0;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y, w: 0.5, h: 0.5, fill: { color: C.teal }, line: { color: C.teal } });
    s.addText(String(i + 1), { x: 0.5, y, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.darkBg, align: 'center', valign: 'middle', margin: 0 });
    s.addText(p[0], { x: 1.2, y: y - 0.02, w: 8.3, h: 0.4, fontSize: 15, bold: true, color: C.teal, fontFace: 'Calibri', margin: 0 });
    s.addText(p[1], { x: 1.2, y: y + 0.36, w: 8.3, h: 0.55, fontSize: 12, color: 'C7D3E0', fontFace: 'Calibri', margin: 0 });
  });
  s.addText('reliable · misinformation · unrelated   —   3-class misinformation classification', {
    x: 0.5, y: 5.15, w: 9, h: 0.35, fontSize: 11, italic: true, color: C.subtle, fontFace: 'Calibri', margin: 0 });
}

// ── Save ──────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/ZeroShot_3Class_Presentation.pptx' })
  .then(() => console.log('SAVED: ZeroShot_3Class_Presentation.pptx'))
  .catch(err => { console.error('ERROR:', err); process.exit(1); });
