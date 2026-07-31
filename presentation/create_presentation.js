const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.title = 'Zero-Shot vs Fine-Tuned LLMs for Misinformation Detection';

// ── Color Palette ─────────────────────────────────────────────────────────────
const C = {
  darkBg:    '0D1B2A',
  navy:      '1A3A5C',
  teal:      '00B4D8',
  tealDark:  '0077A8',
  tealLight: 'E0F7FF',
  lightBg:   'F4F6FA',
  white:     'FFFFFF',
  textDark:  '1A2B4A',
  textBody:  '2D3748',
  orange:    'F4A261',
  green:     '2DC653',
  red:       'E63946',
  subtle:    '90A4AE',
  cardBg:    'FFFFFF',
};

const BASE = 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/results/figures';
const W = 10, H = 5.625;

// ── Helpers ───────────────────────────────────────────────────────────────────
function addHeaderBar(slide, title, subtitle) {
  // Dark top bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 1.15, fill: { color: C.darkBg }
  });
  // Teal left accent stripe
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 1.15, fill: { color: C.teal }
  });
  slide.addText(title, {
    x: 0.28, y: 0.1, w: 9.5, h: 0.6,
    fontSize: 24, bold: true, color: C.white, fontFace: 'Calibri', margin: 0
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.28, y: 0.72, w: 9.5, h: 0.35,
      fontSize: 13, color: C.teal, fontFace: 'Calibri', margin: 0, italic: true
    });
  }
  // Thin teal underline
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 1.15, w: W, h: 0.04, fill: { color: C.teal }
  });
}

function addCard(slide, x, y, w, h, fillColor) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: fillColor || C.cardBg },
    line: { color: 'E2E8F0', width: 0.75 },
    shadow: { type: 'outer', color: '000000', blur: 4, offset: 1, angle: 135, opacity: 0.08 }
  });
}

function addLabelChip(slide, label, x, y, color, textColor) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w: 1.5, h: 0.32, fill: { color: color }, rectRadius: 0.05,
    line: { color: color }
  });
  slide.addText(label, {
    x, y, w: 1.5, h: 0.32,
    fontSize: 10, bold: true, color: textColor || C.white,
    fontFace: 'Calibri', align: 'center', valign: 'middle', margin: 0
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ══════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();

  // Full dark background
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: H, fill: { color: C.darkBg }
  });

  // Left teal accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.22, h: H, fill: { color: C.teal }
  });

  // Bottom gradient-style stripe
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 4.8, w: W, h: 0.825, fill: { color: C.navy }
  });

  // Decorative teal circle (background element)
  slide.addShape(pres.shapes.OVAL, {
    x: 7.2, y: -0.8, w: 4.5, h: 4.5,
    fill: { color: C.tealDark, transparency: 75 },
    line: { color: C.tealDark, transparency: 75 }
  });

  // Main title
  slide.addText([
    { text: 'Zero-Shot vs Fine-Tuned LLMs', options: { breakLine: true } },
    { text: 'for Misinformation Detection', options: {} }
  ], {
    x: 0.5, y: 0.8, w: 9, h: 1.5,
    fontSize: 36, bold: true, color: C.white, fontFace: 'Calibri', margin: 0
  });

  // Teal divider line
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 2.4, w: 5.5, h: 0.06, fill: { color: C.teal }
  });

  // Subtitle
  slide.addText('Comparing RoBERTa Fine-Tuned with Llama 3.1 8B Zero-Shot\nacross Three Misinformation Datasets', {
    x: 0.5, y: 2.6, w: 8, h: 0.9,
    fontSize: 16, color: 'A8C8E8', fontFace: 'Calibri', margin: 0
  });

  // Dataset badges
  const badges = [
    { label: '🏟️  Manchester Arena', x: 0.5 },
    { label: '🦠  Monkeypox',       x: 3.2 },
    { label: '📰  PHEME',           x: 5.5 },
  ];
  badges.forEach(b => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: b.x, y: 3.7, w: 2.5, h: 0.45,
      fill: { color: C.tealDark, transparency: 20 }, rectRadius: 0.08,
      line: { color: C.teal, width: 1 }
    });
    slide.addText(b.label, {
      x: b.x, y: 3.7, w: 2.5, h: 0.45,
      fontSize: 12, color: C.white, fontFace: 'Calibri',
      align: 'center', valign: 'middle', margin: 0
    });
  });

  // Bottom info
  slide.addText('Yoni  |  Final Project 2025–2026', {
    x: 0.5, y: 4.95, w: 9, h: 0.4,
    fontSize: 11, color: C.subtle, fontFace: 'Calibri', margin: 0
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Research Question
// ══════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeaderBar(slide, 'Research Question', 'What drives this study?');

  // Big question box
  addCard(slide, 0.4, 1.35, 9.2, 1.3, 'EBF5FB');
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.35, w: 0.12, h: 1.3, fill: { color: C.teal }
  });
  slide.addText(
    'Can a large language model, with zero training examples, detect misinformation\nas accurately as a fine-tuned RoBERTa model trained on thousands of labeled tweets?',
    {
      x: 0.65, y: 1.4, w: 8.8, h: 1.2,
      fontSize: 16, bold: true, color: C.textDark, fontFace: 'Calibri',
      margin: 0, valign: 'middle'
    }
  );

  // 3 cards below
  const cards = [
    {
      icon: '🗃️', title: '3 Datasets', body: 'Manchester Arena bombing,\nMonkeypox health claims,\nPHEME breaking news',
      color: C.tealDark
    },
    {
      icon: '🧠', title: '2 Approaches', body: 'Fine-Tuning (supervised)\nvs\nZero-Shot (no training)',
      color: C.navy
    },
    {
      icon: '📊', title: '5 Metrics', body: 'F1 Macro, F1 Weighted,\nAccuracy, Precision,\nRecall',
      color: '6A0572'
    },
  ];

  cards.forEach((c, i) => {
    const x = 0.4 + i * 3.1;
    addCard(slide, x, 2.85, 2.9, 2.5);
    // Color top accent
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: 2.85, w: 2.9, h: 0.35, fill: { color: c.color }
    });
    slide.addText(c.icon + '  ' + c.title, {
      x: x + 0.1, y: 2.87, w: 2.7, h: 0.32,
      fontSize: 13, bold: true, color: C.white, fontFace: 'Calibri', margin: 0
    });
    slide.addText(c.body, {
      x: x + 0.12, y: 3.3, w: 2.65, h: 1.9,
      fontSize: 13, color: C.textBody, fontFace: 'Calibri', margin: 0
    });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Datasets Overview
// ══════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeaderBar(slide, 'Datasets', 'Three domains — real-world misinformation challenges');

  const datasets = [
    {
      name: 'Manchester Arena',
      year: '2017',
      size: '89,147',
      labeled: '~2,500',
      labels: 'reliable / misinformation',
      source: 'Twitter (event-specific)',
      img: `${BASE}/manchester/01_label_distribution.png`,
      color: C.tealDark,
    },
    {
      name: 'Monkeypox',
      year: '2022',
      size: '~6,287',
      labeled: '~2,500',
      labels: 'reliable / misinformation',
      source: 'Twitter (health crisis)',
      img: `${BASE}/monkeypox/mpox_01_label_distribution.png`,
      color: C.navy,
    },
    {
      name: 'PHEME',
      year: '2015',
      size: '62,445',
      labeled: '~2,500',
      labels: 'not_rumour / rumour',
      source: 'Twitter (breaking news)',
      img: `${BASE}/pheme/pheme_01_label_distribution.png`,
      color: '6A0572',
    },
  ];

  datasets.forEach((ds, i) => {
    const x = 0.28 + i * 3.25;
    addCard(slide, x, 1.3, 3.1, 4.05);
    // Colored header
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.3, w: 3.1, h: 0.48, fill: { color: ds.color }
    });
    slide.addText(ds.name, {
      x: x + 0.08, y: 1.33, w: 2.95, h: 0.42,
      fontSize: 14, bold: true, color: C.white, fontFace: 'Calibri', margin: 0, valign: 'middle'
    });
    // Image
    slide.addImage({
      path: ds.img, x: x + 0.05, y: 1.83, w: 3.0, h: 1.85
    });
    // Stats
    const stats = [
      `Total tweets: ${ds.size}`,
      `Gold Standard: ${ds.labeled}`,
      `Labels: ${ds.labels}`,
      `Source: ${ds.source}`,
    ];
    stats.forEach((s, si) => {
      slide.addText([{ text: s }], {
        x: x + 0.1, y: 3.78 + si * 0.22, w: 2.9, h: 0.22,
        fontSize: 11, color: C.textBody, fontFace: 'Calibri', margin: 0
      });
    });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Preprocessing Pipeline
// ══════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeaderBar(slide, 'Data Preprocessing', 'Building a clean, balanced Gold Standard for fair evaluation');

  // Left: pipeline steps
  const steps = [
    { num: '1', label: 'Remove URLs, @mentions,\n#tags (keep text), RT, emojis' },
    { num: '2', label: 'Filter tweets < 5 words\n(noise removal)' },
    { num: '3', label: 'Deduplicate\n(remove exact copies)' },
    { num: '4', label: 'Sample Gold Standard\n~2,500 stratified tweets' },
    { num: '5', label: 'Stratified Split\n70% Train / 15% Val / 15% Test' },
  ];

  steps.forEach((s, i) => {
    const y = 1.35 + i * 0.78;
    // Circle number
    slide.addShape(pres.shapes.OVAL, {
      x: 0.35, y: y, w: 0.48, h: 0.48,
      fill: { color: C.teal }, line: { color: C.teal }
    });
    slide.addText(s.num, {
      x: 0.35, y: y, w: 0.48, h: 0.48,
      fontSize: 14, bold: true, color: C.white, fontFace: 'Calibri',
      align: 'center', valign: 'middle', margin: 0
    });
    // Connector line (except last)
    if (i < steps.length - 1) {
      slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.56, y: y + 0.48, w: 0.06, h: 0.3,
        fill: { color: C.subtle }, line: { color: C.subtle }
      });
    }
    // Step box
    addCard(slide, 0.95, y, 4.2, 0.48);
    slide.addText(s.label, {
      x: 1.05, y: y + 0.02, w: 4.0, h: 0.44,
      fontSize: 11, color: C.textDark, fontFace: 'Calibri', margin: 0, valign: 'middle'
    });
  });

  // Right: before/after cleaning figure
  slide.addText('Before → After Cleaning', {
    x: 5.5, y: 1.28, w: 4.2, h: 0.3,
    fontSize: 12, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0, align: 'center'
  });
  slide.addImage({
    path: `${BASE}/manchester/08_before_after_cleaning.png`,
    x: 5.4, y: 1.6, w: 4.35, h: 2.3
  });

  // Manchester example tweet cleaned
  addCard(slide, 5.4, 4.0, 4.35, 1.4, 'FFFBF0');
  slide.addText('Example Tweet (Manchester):', {
    x: 5.55, y: 4.05, w: 4.0, h: 0.22,
    fontSize: 10, bold: true, color: C.tealDark, fontFace: 'Calibri', margin: 0
  });
  slide.addText([
    { text: 'Before: ', options: { bold: true, color: C.red } },
    { text: '"@user RT: Manchester Arena BOMB — refugees did this! 💥 #Islamists http://t.co/xyz"', options: { italic: true, color: C.textBody } },
    { text: '\nAfter:  ', options: { bold: true, color: C.green, breakLine: true } },
    { text: '"Manchester Arena BOMB — refugees did this Islamists"', options: { italic: true, color: C.textBody } },
  ], {
    x: 5.55, y: 4.3, w: 4.1, h: 1.05,
    fontSize: 9.5, fontFace: 'Calibri', margin: 0
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Gold Standard & Class Imbalance
// ══════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeaderBar(slide, 'Gold Standard & Class Imbalance', 'Why we cannot use the raw data directly');

  // Problem box
  addCard(slide, 0.35, 1.32, 4.4, 1.6, 'FFF3F3');
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 1.32, w: 0.1, h: 1.6, fill: { color: C.red }
  });
  slide.addText('⚠️  The Problem', {
    x: 0.55, y: 1.37, w: 4.1, h: 0.32,
    fontSize: 13, bold: true, color: C.red, fontFace: 'Calibri', margin: 0
  });
  slide.addText([
    { text: 'Manchester raw: 89,147 tweets\n', options: { breakLine: true } },
    { text: '→ Only 490 misinformation (0.55%)\n', options: { breakLine: true, color: C.red } },
    { text: 'A naive model predicting "reliable" always\nwould achieve 99.4% accuracy!', options: {} }
  ], {
    x: 0.55, y: 1.73, w: 4.1, h: 1.1,
    fontSize: 11, color: C.textBody, fontFace: 'Calibri', margin: 0
  });

  // Solution box
  addCard(slide, 0.35, 3.1, 4.4, 2.25, 'F0FFF4');
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 3.1, w: 0.1, h: 2.25, fill: { color: C.green }
  });
  slide.addText('✅  Our Solution', {
    x: 0.55, y: 3.15, w: 4.1, h: 0.32,
    fontSize: 13, bold: true, color: C.green, fontFace: 'Calibri', margin: 0
  });
  slide.addText([
    { text: '• Take ALL minority class examples\n', options: { breakLine: true } },
    { text: '• Random sample from majority class\n', options: { breakLine: true } },
    { text: '• Result: ~2,500 balanced Gold Standard\n', options: { breakLine: true } },
    { text: '• Stratified 70/15/15 split\n', options: { breakLine: true } },
    { text: '• Class weights in training loss', options: {} },
  ], {
    x: 0.55, y: 3.5, w: 4.1, h: 1.75,
    fontSize: 11, color: C.textBody, fontFace: 'Calibri', margin: 0
  });

  // Gold standard split figure
  slide.addText('Manchester Gold Standard Splits', {
    x: 5.1, y: 1.28, w: 4.6, h: 0.28,
    fontSize: 12, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0, align: 'center'
  });
  slide.addImage({
    path: `${BASE}/manchester/09_gold_standard_splits.png`,
    x: 5.05, y: 1.6, w: 4.65, h: 2.4
  });

  // Split summary table
  const rows = [
    ['Dataset', 'Gold Total', 'Train', 'Val', 'Test'],
    ['Manchester', '2,427', '1,698', '364', '365'],
    ['Monkeypox',  '~2,500', '~1,750', '~375', '~375'],
    ['PHEME',      '~2,500', '~1,750', '~375', '~375'],
  ];
  const colW = [1.6, 1.0, 0.85, 0.75, 0.75];
  const startX = 5.05;
  let rowY = 3.95;
  rows.forEach((row, ri) => {
    let cx = startX;
    row.forEach((cell, ci) => {
      const isHeader = ri === 0;
      slide.addShape(pres.shapes.RECTANGLE, {
        x: cx, y: rowY, w: colW[ci], h: 0.36,
        fill: { color: isHeader ? C.darkBg : (ri % 2 === 0 ? 'F0F4F8' : C.white) },
        line: { color: 'E2E8F0', width: 0.5 }
      });
      slide.addText(cell, {
        x: cx + 0.04, y: rowY + 0.04, w: colW[ci] - 0.08, h: 0.28,
        fontSize: 11, bold: isHeader, color: isHeader ? C.white : C.textDark,
        fontFace: 'Calibri', align: 'center', margin: 0
      });
      cx += colW[ci];
    });
    rowY += 0.36;
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 6 — RoBERTa Fine-Tuning
// ══════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeaderBar(slide, 'RoBERTa Fine-Tuning', 'Supervised approach — learning from labeled examples');

  // Architecture diagram (left side)
  // Tweet input → Tokenizer → [CLS] t1 t2 ... → RoBERTa 12 layers → Classifier
  const archItems = [
    { label: 'Raw Tweet', color: C.subtle, textColor: C.white },
    { label: 'RoBERTa\nTokenizer (BPE)', color: C.tealDark, textColor: C.white },
    { label: 'Token Embeddings\n[CLS] t1 t2 ... [SEP]', color: C.navy, textColor: C.white },
    { label: 'RoBERTa-base\n12 Transformer Layers\n(125M params)', color: C.teal, textColor: C.white },
    { label: '[CLS] Representation\n(768-dim)', color: C.tealDark, textColor: C.white },
    { label: 'Classification Head\n768 → 2 (Linear)', color: C.orange, textColor: C.white },
    { label: 'reliable / misinformation', color: C.green, textColor: C.white },
  ];

  archItems.forEach((item, i) => {
    const y = 1.35 + i * 0.56;
    // Box
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.35, y, w: 3.5, h: 0.48,
      fill: { color: item.color }, line: { color: item.color },
      shadow: { type: 'outer', color: '000000', blur: 3, offset: 1, angle: 135, opacity: 0.1 }
    });
    slide.addText(item.label, {
      x: 0.35, y, w: 3.5, h: 0.48,
      fontSize: 10, bold: true, color: item.textColor,
      fontFace: 'Calibri', align: 'center', valign: 'middle', margin: 0
    });
    // Arrow down (except last)
    if (i < archItems.length - 1) {
      slide.addText('▼', {
        x: 1.7, y: y + 0.47, w: 0.8, h: 0.1,
        fontSize: 9, color: C.subtle, align: 'center', margin: 0
      });
    }
  });

  // Right: training details
  addCard(slide, 4.1, 1.32, 5.65, 4.05);

  slide.addText('Training Configuration', {
    x: 4.25, y: 1.42, w: 5.3, h: 0.35,
    fontSize: 14, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0
  });

  // Hyperparams table
  const params = [
    ['Parameter', 'Value', 'Reason'],
    ['Learning Rate', '2e-5', 'Prevent catastrophic forgetting'],
    ['Batch Size', '16', 'GPU memory balance'],
    ['Epochs (max)', '4', 'With early stopping (patience=2)'],
    ['Warmup Ratio', '10%', 'Smooth LR increase at start'],
    ['Loss Function', 'Weighted Cross-Entropy', 'Handles class imbalance'],
    ['Precision', 'fp16 (mixed)', 'Faster GPU training'],
    ['Seed', '42', 'Reproducibility'],
  ];

  const colWs = [1.6, 1.5, 2.35];
  let ry = 1.85;
  params.forEach((row, ri) => {
    let cx = 4.18;
    row.forEach((cell, ci) => {
      const isH = ri === 0;
      slide.addShape(pres.shapes.RECTANGLE, {
        x: cx, y: ry, w: colWs[ci], h: 0.36,
        fill: { color: isH ? C.darkBg : (ri % 2 === 0 ? 'F0F4F8' : C.white) },
        line: { color: 'E2E8F0', width: 0.5 }
      });
      slide.addText(cell, {
        x: cx + 0.05, y: ry + 0.04, w: colWs[ci] - 0.1, h: 0.28,
        fontSize: 10.5, bold: isH, color: isH ? C.white : C.textDark,
        fontFace: 'Calibri', valign: 'middle', margin: 0
      });
      cx += colWs[ci];
    });
    ry += 0.36;
  });

  // CV note
  addCard(slide, 4.1, 4.6, 5.65, 0.75, 'EBF5FB');
  slide.addText('📋  5-Fold Stratified Cross-Validation', {
    x: 4.25, y: 4.65, w: 5.3, h: 0.28,
    fontSize: 11, bold: true, color: C.tealDark, fontFace: 'Calibri', margin: 0
  });
  slide.addText('Gold Standard split into 5 equal parts → train on 4, validate on 1 → rotate → report mean ± std F1', {
    x: 4.25, y: 4.93, w: 5.3, h: 0.35,
    fontSize: 10, color: C.textBody, fontFace: 'Calibri', margin: 0
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 7 — RoBERTa Results
// ══════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeaderBar(slide, 'RoBERTa Results', '5-Fold CV + Final Test Set Evaluation');

  // Manchester CV figure (left)
  slide.addText('Manchester — 5-Fold CV', {
    x: 0.35, y: 1.28, w: 4.5, h: 0.28,
    fontSize: 11, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0, align: 'center'
  });
  slide.addImage({
    path: `${BASE}/manchester/manchester_roberta_cv_results.png`,
    x: 0.35, y: 1.6, w: 4.5, h: 2.3
  });

  // Manchester test CM (right)
  slide.addText('Manchester — Test Set Confusion Matrix', {
    x: 5.1, y: 1.28, w: 4.6, h: 0.28,
    fontSize: 11, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0, align: 'center'
  });
  slide.addImage({
    path: `${BASE}/manchester/manchester_roberta_test_cm.png`,
    x: 5.1, y: 1.6, w: 4.6, h: 2.3
  });

  // Summary table all 3 datasets
  slide.addText('Test Set Results — All Datasets (RoBERTa Fine-Tuned)', {
    x: 0.35, y: 3.98, w: 9.3, h: 0.28,
    fontSize: 12, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0
  });

  const sumRows = [
    ['Dataset', 'CV F1 Macro (mean ± std)', 'Test F1 Macro', 'Test Accuracy', 'Test Precision', 'Test Recall'],
    ['Manchester', '0.961 ± 0.012', '0.986', '0.992', '0.989', '0.983'],
    ['Monkeypox',  '0.893 ± 0.014', '0.913', '0.921', '0.913', '0.913'],
    ['PHEME',      '0.599 ± 0.007', '0.598', '0.822', '0.627', '0.586'],
  ];
  const scol = [1.55, 1.9, 1.2, 1.15, 1.2, 1.2];
  let sy = 4.29;
  sumRows.forEach((row, ri) => {
    let cx = 0.35;
    row.forEach((cell, ci) => {
      const isH = ri === 0;
      slide.addShape(pres.shapes.RECTANGLE, {
        x: cx, y: sy, w: scol[ci], h: 0.31,
        fill: { color: isH ? C.darkBg : (ri % 2 === 0 ? 'F0F4F8' : C.white) },
        line: { color: 'E2E8F0', width: 0.5 }
      });
      slide.addText(cell, {
        x: cx + 0.04, y: sy + 0.03, w: scol[ci] - 0.08, h: 0.25,
        fontSize: 10, bold: isH, color: isH ? C.white : C.textDark,
        fontFace: 'Calibri', align: 'center', margin: 0
      });
      cx += scol[ci];
    });
    sy += 0.31;
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Zero-Shot with Ollama (Llama 3.1)
// ══════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeaderBar(slide, 'Zero-Shot Classification — Ollama + Llama 3.1 8B', 'No training examples required');

  // Left: How it works
  addCard(slide, 0.35, 1.32, 4.5, 4.05);
  slide.addText('🔁  How Zero-Shot Works', {
    x: 0.5, y: 1.42, w: 4.2, h: 0.35,
    fontSize: 13, bold: true, color: C.tealDark, fontFace: 'Calibri', margin: 0
  });

  const steps = [
    { icon: '🏠', text: 'Ollama runs Llama 3.1 8B locally\n(no API key, no internet required)' },
    { icon: '📝', text: 'Build a Chain-of-Thought prompt\nwith dataset-specific class definitions' },
    { icon: '🤖', text: 'Model reasons step-by-step,\nthen outputs structured JSON' },
    { icon: '🔍', text: 'Parse JSON → extract label\nFallback: keyword scan in raw text' },
    { icon: '💾', text: 'Checkpoint every 50 tweets\n(resume if interrupted)' },
  ];

  steps.forEach((s, i) => {
    const y = 1.88 + i * 0.65;
    slide.addText(s.icon, {
      x: 0.48, y: y, w: 0.4, h: 0.5,
      fontSize: 18, margin: 0, valign: 'middle'
    });
    slide.addText(s.text, {
      x: 0.92, y: y + 0.02, w: 3.78, h: 0.5,
      fontSize: 10.5, color: C.textBody, fontFace: 'Calibri', margin: 0, valign: 'middle'
    });
    if (i < steps.length - 1) {
      slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.6, y: y + 0.51, w: 0.06, h: 0.14,
        fill: { color: C.subtle }
      });
    }
  });

  // Right: Prompt example
  addCard(slide, 5.05, 1.32, 4.65, 4.05, '1A1A2E');
  slide.addText('Prompt Structure (Chain-of-Thought)', {
    x: 5.15, y: 1.38, w: 4.45, h: 0.32,
    fontSize: 11, bold: true, color: C.teal, fontFace: 'Calibri', margin: 0
  });

  const promptLines = [
    { text: 'You are an expert fact-checker...\n', color: C.orange },
    { text: 'Topic: the 2017 Manchester Arena bombing\n\n', color: C.teal },
    { text: 'CLASSES:\n', color: 'FFFFFF' },
    { text: '  "reliable": factually accurate...\n', color: '90EE90' },
    { text: '  "misinformation": false claims...\n\n', color: 'FF8080' },
    { text: 'TWEET: """[tweet text]"""\n\n', color: 'FFD700' },
    { text: 'INSTRUCTIONS:\n', color: 'FFFFFF' },
    { text: 'Think step-by-step:\n', color: C.subtle },
    { text: '1. What claim does the tweet make?\n', color: C.subtle },
    { text: '2. Verifiable facts or emotional?\n', color: C.subtle },
    { text: '3. Conspiracy signals?\n\n', color: C.subtle },
    { text: 'Output JSON:\n', color: 'FFFFFF' },
    { text: '{"reasoning": "...",\n "label": "reliable",\n "confidence": 0.87}', color: '90EE90' },
  ];

  slide.addText(
    promptLines.map(l => ({ text: l.text, options: { color: l.color } })),
    {
      x: 5.12, y: 1.76, w: 4.5, h: 3.55,
      fontSize: 9.5, fontFace: 'Consolas', margin: 0.05,
      valign: 'top'
    }
  );

  // temperature note
  slide.addText('temperature=0  (deterministic output)', {
    x: 5.12, y: 5.05, w: 4.45, h: 0.22,
    fontSize: 9, color: C.teal, italic: true, fontFace: 'Calibri', margin: 0
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Comparison Results
// ══════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeaderBar(slide, 'Results Comparison — RoBERTa vs Llama 3.1 Zero-Shot', 'Head-to-head on all three datasets');

  // Manchester comparison bars
  slide.addText('Manchester', {
    x: 0.35, y: 1.28, w: 3.0, h: 0.28,
    fontSize: 11, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0, align: 'center'
  });
  slide.addImage({
    path: `${BASE}/manchester/manchester_comparison_bars.png`,
    x: 0.3, y: 1.58, w: 3.1, h: 2.25
  });

  // Manchester confusion matrices
  slide.addText('Confusion Matrices (Manchester)', {
    x: 3.6, y: 1.28, w: 6.1, h: 0.28,
    fontSize: 11, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0, align: 'center'
  });
  slide.addImage({
    path: `${BASE}/manchester/manchester_comparison_cm.png`,
    x: 3.55, y: 1.58, w: 6.1, h: 2.25
  });

  // Manchester final summary figure
  slide.addText('Manchester — Final Summary', {
    x: 0.35, y: 3.98, w: 3.0, h: 0.28,
    fontSize: 11, bold: true, color: C.textDark, fontFace: 'Calibri', margin: 0, align: 'center'
  });
  slide.addImage({
    path: `${BASE}/manchester/manchester_final_summary.png`,
    x: 0.3, y: 4.28, w: 3.1, h: 1.2
  });

  // Agreement / key insight
  addCard(slide, 3.55, 3.95, 6.1, 1.55, 'EBF5FB');
  slide.addText('💡  Key Finding', {
    x: 3.7, y: 4.0, w: 5.8, h: 0.32,
    fontSize: 12, bold: true, color: C.tealDark, fontFace: 'Calibri', margin: 0
  });
  slide.addText([
    { text: '• RoBERTa Fine-Tuned ', options: { bold: true } },
    { text: 'achieves higher F1 Macro overall (trained on domain-specific data)\n', options: {} },
    { text: '• Llama 3.1 8B Zero-Shot ', options: { bold: true } },
    { text: 'shows competitive performance — ', options: {} },
    { text: 'no training examples used\n', options: { italic: true } },
    { text: '• McNemar\'s Test: ', options: { bold: true } },
    { text: 'checks whether performance differences are statistically significant (p < 0.05)', options: {} },
  ], {
    x: 3.7, y: 4.35, w: 5.8, h: 1.1,
    fontSize: 10.5, color: C.textBody, fontFace: 'Calibri', margin: 0
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Summary & Next Steps
// ══════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();

  // Full dark background
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: H, fill: { color: C.darkBg }
  });
  // Left teal bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.22, h: H, fill: { color: C.teal }
  });

  slide.addText('Summary & Next Steps', {
    x: 0.45, y: 0.25, w: 9.2, h: 0.55,
    fontSize: 28, bold: true, color: C.white, fontFace: 'Calibri', margin: 0
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.45, y: 0.85, w: 6, h: 0.05, fill: { color: C.teal }
  });

  // Phase 1 summary cards
  slide.addText('Phase 1 — Completed', {
    x: 0.45, y: 1.05, w: 4.5, h: 0.35,
    fontSize: 15, bold: true, color: C.teal, fontFace: 'Calibri', margin: 0
  });

  const phase1 = [
    '✅  3 datasets preprocessed & Gold Standards created',
    '✅  RoBERTa-base fine-tuned on each dataset (5-Fold CV)',
    '✅  Zero-Shot with Llama 3.1 8B via Ollama (local)',
    '✅  Full evaluation: F1 Macro, Confusion Matrices, McNemar test',
    '✅  Comparison figures & Master Results CSV generated',
  ];

  phase1.forEach((item, i) => {
    slide.addText(item, {
      x: 0.6, y: 1.5 + i * 0.46, w: 4.2, h: 0.4,
      fontSize: 11.5, color: 'A8E6CF', fontFace: 'Calibri', margin: 0
    });
  });

  // Phase 2 box
  addCard(slide, 5.1, 1.05, 4.6, 4.35, '0A2540');
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.05, w: 4.6, h: 0.42,
    fill: { color: C.orange }
  });
  slide.addText('🚀  Phase 2 — Planned', {
    x: 5.2, y: 1.08, w: 4.4, h: 0.36,
    fontSize: 14, bold: true, color: C.white, fontFace: 'Calibri', margin: 0, valign: 'middle'
  });

  const phase2 = [
    { icon: '🌐', text: 'GPT-OSS 120B  (via Groq API)', sub: 'OpenAI open-source, 120B params' },
    { icon: '🦙', text: 'Llama 3.3 70B  (via Groq API)', sub: 'Meta, 70B dense model' },
    { icon: '🧩', text: 'Qwen3 32B  (via Groq API)', sub: 'Alibaba, 32B with thinking mode' },
  ];

  phase2.forEach((item, i) => {
    const y = 1.62 + i * 1.1;
    slide.addText(item.icon, {
      x: 5.2, y: y, w: 0.55, h: 0.9,
      fontSize: 24, margin: 0, valign: 'middle'
    });
    slide.addText(item.text, {
      x: 5.78, y: y + 0.08, w: 3.8, h: 0.38,
      fontSize: 12, bold: true, color: C.orange, fontFace: 'Calibri', margin: 0
    });
    slide.addText(item.sub, {
      x: 5.78, y: y + 0.48, w: 3.8, h: 0.3,
      fontSize: 10, color: C.subtle, fontFace: 'Calibri', italic: true, margin: 0
    });
    if (i < phase2.length - 1) {
      slide.addShape(pres.shapes.RECTANGLE, {
        x: 5.2, y: y + 0.92, w: 4.35, h: 0.01, fill: { color: '1E3A5F' }
      });
    }
  });

  slide.addText('Goal: determine whether bigger models = better zero-shot performance (scaling law)', {
    x: 5.2, y: 4.98, w: 4.35, h: 0.35,
    fontSize: 10, color: C.teal, italic: true, fontFace: 'Calibri', margin: 0
  });

  // Bottom quote
  slide.addText('"The best model is the one that generalizes — with or without training."', {
    x: 0.5, y: 5.1, w: 4.4, h: 0.38,
    fontSize: 11, italic: true, color: '607D8B', fontFace: 'Calibri', margin: 0
  });
}

// ── Save ──────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/ZeroShot_Presentation.pptx' })
  .then(() => console.log('✅  Presentation saved: ZeroShot_Presentation.pptx'))
  .catch(err => { console.error('❌  Error:', err); process.exit(1); });
