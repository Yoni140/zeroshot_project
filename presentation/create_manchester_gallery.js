const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.title = 'Manchester — Full Figure Gallery';

const C = {
  darkBg: '0D1B2A', navy: '1A3A5C', teal: '00B4D8', tealDark: '0077A8',
  lightBg: 'F4F6FA', white: 'FFFFFF', textDark: '1A2B4A', textBody: '2D3748',
  green: '2DC653', red: 'E63946', subtle: '90A4AE', orange: 'F4A261', purple: '6A0572',
};
const W = 10, H = 5.625;
const B = 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/results/figures/manchester';

// section colors for the little tag chip
function header(slide, title, sub, tagText, tagColor) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 1.0, fill: { color: C.darkBg } });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 1.0, fill: { color: C.teal } });
  slide.addText(title, { x: 0.28, y: 0.08, w: 7.3, h: 0.55, fontSize: 22, bold: true, color: C.white, fontFace: 'Calibri', margin: 0 });
  if (sub) slide.addText(sub, { x: 0.28, y: 0.62, w: 7.3, h: 0.32, fontSize: 12, color: C.teal, italic: true, fontFace: 'Calibri', margin: 0 });
  if (tagText) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.0, y: 0.3, w: 1.75, h: 0.42, fill: { color: tagColor || C.tealDark }, rectRadius: 0.08, line: { color: tagColor || C.tealDark } });
    slide.addText(tagText, { x: 8.0, y: 0.3, w: 1.75, h: 0.42, fontSize: 11, bold: true, color: C.white, fontFace: 'Calibri', align: 'center', valign: 'middle', margin: 0 });
  }
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.0, w: W, h: 0.035, fill: { color: C.teal } });
}

// Fit an image (origW x origH) into box, centered, preserving aspect ratio.
function fitImg(slide, path, ow, oh, bx, by, bw, bh) {
  const ar = ow / oh, bar = bw / bh;
  let w, h;
  if (ar > bar) { w = bw; h = bw / ar; } else { h = bh; w = bh * ar; }
  const x = bx + (bw - w) / 2, y = by + (bh - h) / 2;
  slide.addImage({ path, x, y, w, h });
}

function caption(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.3, fontSize: 11, bold: true, color: C.textDark, fontFace: 'Calibri', align: 'center', margin: 0 });
}

// single big image slide
function single(title, sub, tag, tagC, img, ow, oh, capText) {
  const s = pres.addSlide(); s.background = { color: C.lightBg };
  header(s, title, sub, tag, tagC);
  if (capText) { caption(s, capText, 0.4, 1.12, 9.2); fitImg(s, `${B}/${img}`, ow, oh, 0.5, 1.45, 9.0, 3.95); }
  else fitImg(s, `${B}/${img}`, ow, oh, 0.5, 1.2, 9.0, 4.25);
  return s;
}

// two images side by side
function dual(title, sub, tag, tagC, a, b) {
  const s = pres.addSlide(); s.background = { color: C.lightBg };
  header(s, title, sub, tag, tagC);
  caption(s, a.cap, 0.3, 1.12, 4.7);
  fitImg(s, `${B}/${a.img}`, a.ow, a.oh, 0.3, 1.42, 4.7, 3.95);
  caption(s, b.cap, 5.0, 1.12, 4.7);
  fitImg(s, `${B}/${b.img}`, b.ow, b.oh, 5.0, 1.42, 4.7, 3.95);
  return s;
}

// two images stacked (for wide plots)
function stack(title, sub, tag, tagC, a, b) {
  const s = pres.addSlide(); s.background = { color: C.lightBg };
  header(s, title, sub, tag, tagC);
  caption(s, a.cap, 0.4, 1.1, 9.2);
  fitImg(s, `${B}/${a.img}`, a.ow, a.oh, 0.5, 1.38, 9.0, 1.95);
  caption(s, b.cap, 0.4, 3.42, 9.2);
  fitImg(s, `${B}/${b.img}`, b.ow, b.oh, 0.5, 3.68, 9.0, 1.85);
  return s;
}

const EDA = ['EDA', C.tealDark];
const PREP = ['PREPROCESS', C.purple];
const ROB = ['RoBERTa', C.navy];
const ZS = ['ZERO-SHOT', C.orange];
const CMP = ['COMPARE', C.green];

// ════════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ════════════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: H, fill: { color: C.darkBg } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.22, h: H, fill: { color: C.teal } });
  s.addShape(pres.shapes.OVAL, { x: 7.2, y: -0.8, w: 4.5, h: 4.5, fill: { color: C.tealDark, transparency: 78 }, line: { color: C.tealDark, transparency: 78 } });
  s.addText('Manchester Arena', { x: 0.5, y: 1.5, w: 9, h: 0.8, fontSize: 40, bold: true, color: C.white, fontFace: 'Calibri', margin: 0 });
  s.addText('Full Figure Gallery — Initial Presentation', { x: 0.5, y: 2.35, w: 9, h: 0.6, fontSize: 22, color: C.teal, fontFace: 'Calibri', margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.05, w: 5.5, h: 0.05, fill: { color: C.teal } });
  s.addText('EDA  →  Preprocessing  →  RoBERTa Fine-Tuning  →  Llama 3.1 Zero-Shot  →  Comparison', {
    x: 0.5, y: 3.3, w: 8.8, h: 0.5, fontSize: 13, color: 'A8C8E8', fontFace: 'Calibri', margin: 0 });
  s.addText('reliable vs misinformation  •  Gold Standard 2,427  •  Test 365', {
    x: 0.5, y: 4.9, w: 9, h: 0.4, fontSize: 12, color: C.subtle, fontFace: 'Calibri', margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════════
// EDA SECTION
// ════════════════════════════════════════════════════════════════════════════════
single('Label Distribution', 'Class balance in the raw Manchester data', ...EDA,
  '01_label_distribution.png', 2075, 728, 'reliable vastly outnumbers misinformation — motivates the Gold Standard');

stack('Tweet Length & Sentiment', 'Distribution analysis across classes', ...EDA,
  { img: '02_tweet_length_distribution.png', ow: 2076, oh: 725, cap: 'Tweet Length Distribution' },
  { img: '03_sentiment_analysis.png', ow: 2076, oh: 725, cap: 'Sentiment Analysis (VADER)' });

single('Engagement Analysis', 'Likes, retweets & replies by class', ...EDA,
  '04_engagement_analysis.png', 2076, 1533, null);

single('Temporal Analysis', 'Tweet volume over time', ...EDA,
  '06_temporal_analysis.png', 2076, 1474, null);

dual('Top Words & Embedding Space', 'Lexical and semantic structure', ...EDA,
  { img: '07_top_words.png', ow: 2375, oh: 886, cap: 'Most Frequent Words' },
  { img: '08_tsne.png', ow: 2676, oh: 1064, cap: 't-SNE of Tweet Embeddings' });

single('URL & Media Usage', 'How reliable vs misinformation tweets share links/media', ...EDA,
  '05_url_media_analysis.png', 1776, 724, null);

// ════════════════════════════════════════════════════════════════════════════════
// PREPROCESSING
// ════════════════════════════════════════════════════════════════════════════════
single('Text Cleaning', 'Before vs after preprocessing', ...PREP,
  '08_before_after_cleaning.png', 1785, 581, 'URLs, @mentions, emojis removed; # kept as text');

single('Gold Standard Splits', 'Stratified 70 / 15 / 30 split of the curated sample', ...PREP,
  '09_gold_standard_splits.png', 2084, 741, 'Train 1,698  •  Val 364  •  Test 365 — same label ratio in each');

// ════════════════════════════════════════════════════════════════════════════════
// RoBERTa TRAINING
// ════════════════════════════════════════════════════════════════════════════════
stack('RoBERTa Training Curves', 'Loss & F1 across epochs (5-Fold CV)', ...ROB,
  { img: 'manchester_roberta_loss_curves.png', ow: 2084, oh: 740, cap: 'Training / Validation Loss' },
  { img: 'manchester_roberta_f1_curves.png', ow: 2084, oh: 740, cap: 'Validation F1 Macro' });

dual('Learning Rate & CV Results', 'Schedule and per-fold performance', ...ROB,
  { img: 'manchester_roberta_lr_schedule.png', ow: 1184, oh: 558, cap: 'LR Schedule (warmup + linear decay)' },
  { img: 'manchester_roberta_cv_results.png', ow: 2083, oh: 731, cap: '5-Fold CV — OOF CM & F1 per fold' });

single('Per-Fold Metrics Heatmap', 'Stability of metrics across the 5 folds', ...ROB,
  'manchester_roberta_cv_heatmap.png', 1399, 581, 'Low variance across folds → no dependence on a lucky split');

// ════════════════════════════════════════════════════════════════════════════════
// RoBERTa EVALUATION
// ════════════════════════════════════════════════════════════════════════════════
dual('RoBERTa — Test Evaluation', 'Final held-out test set (n=365)', ...ROB,
  { img: 'manchester_roberta_test_cm.png', ow: 849, oh: 731, cap: 'Confusion Matrix' },
  { img: 'manchester_roberta_roc_pr_curves.png', ow: 2084, oh: 731, cap: 'ROC & Precision-Recall Curves' });

single('RoBERTa — Results Summary', 'CV vs Test metrics table', ...ROB,
  'manchester_roberta_summary_table.png', 1485, 584, 'Test F1 Macro 0.986  •  Accuracy 0.992  •  ROC-AUC near-perfect');

// ════════════════════════════════════════════════════════════════════════════════
// ZERO-SHOT
// ════════════════════════════════════════════════════════════════════════════════
dual('Zero-Shot — Llama 3.1 8B', 'Confusion matrix & confidence distribution', ...ZS,
  { img: 'manchester_zeroshot_cm.png', ow: 2016, oh: 731, cap: 'Confusion Matrix (counts & normalized)' },
  { img: 'manchester_zeroshot_confidence.png', ow: 2084, oh: 731, cap: 'Model Confidence by Outcome' });

single('Zero-Shot — Results Summary', 'Llama 3.1 8B metrics table', ...ZS,
  'manchester_zeroshot_summary_table.png', 1485, 584, 'Test F1 Macro 0.684  •  Accuracy 0.753  •  no training examples');

// ════════════════════════════════════════════════════════════════════════════════
// COMPARISON
// ════════════════════════════════════════════════════════════════════════════════
single('Head-to-Head — Metrics', 'RoBERTa vs Llama 3.1 Zero-Shot', ...CMP,
  'manchester_comparison_bars.png', 1780, 727, 'RoBERTa leads on every metric');

single('Head-to-Head — Confusion Matrices', 'Side-by-side error patterns', ...CMP,
  'manchester_comparison_cm.png', 1877, 771, null);

dual('Agreement & Final Summary', 'Where the models agree, and the verdict', ...CMP,
  { img: 'manchester_agreement.png', ow: 1180, oh: 581, cap: 'Model Agreement' },
  { img: 'manchester_final_summary.png', ow: 1630, oh: 580, cap: 'Final Summary' });

pres.writeFile({ fileName: 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/Manchester_Figure_Gallery.pptx' })
  .then(() => console.log('✅ Saved Manchester_Figure_Gallery.pptx'))
  .catch(e => { console.error('❌', e); process.exit(1); });
