const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.title = 'Training Loss Curves — Manchester';

const C = {
  darkBg: '0D1B2A', navy: '1A3A5C', teal: '00B4D8', tealDark: '0077A8',
  lightBg: 'F4F6FA', white: 'FFFFFF', textDark: '1A2B4A', textBody: '2D3748',
  green: '2DC653', red: 'E63946', subtle: '90A4AE', orange: 'F4A261',
};
const W = 10, H = 5.625;
const IMG = 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/results/figures/manchester/manchester_roberta_loss_curves.png';

const slide = pres.addSlide();
slide.background = { color: C.lightBg };

// Header
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 1.0, fill: { color: C.darkBg } });
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 1.0, fill: { color: C.teal } });
slide.addText('Training & Validation Loss — RoBERTa (5-Fold CV)', { x: 0.28, y: 0.08, w: 9.5, h: 0.55, fontSize: 23, bold: true, color: C.white, fontFace: 'Calibri', margin: 0 });
slide.addText('Manchester — convergence behaviour and why the noisy val-loss is not a problem', { x: 0.28, y: 0.6, w: 9.5, h: 0.34, fontSize: 13, color: C.teal, italic: true, fontFace: 'Calibri', margin: 0 });
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.0, w: W, h: 0.035, fill: { color: C.teal } });

// Figure (1064x728 aspect ~1.463) — fit into top band
// place at x centered, width 9.0 -> height 9.0/ (1064/728?) the png is 1064x... actually wide ~ 1064? it's the saved 150dpi. Use known ratio from file: ~1064x... we sized 14x5 fig => ratio ~2.8? Actually displayed is wide. Use w=9.0,h=2.45
slide.addImage({ path: IMG, x: 0.7, y: 1.2, w: 8.6, h: 2.45 });

// Two captions under the two panels
slide.addText('◀  Train Loss — smooth descent to ~0', { x: 0.7, y: 3.66, w: 4.3, h: 0.28, fontSize: 11, bold: true, color: C.tealDark, fontFace: 'Calibri', align: 'center', margin: 0 });
slide.addText('Validation Loss — noisy across folds  ▶', { x: 5.0, y: 3.66, w: 4.3, h: 0.28, fontSize: 11, bold: true, color: C.orange, fontFace: 'Calibri', align: 'center', margin: 0 });

// ── Explanation: 3 cards ──────────────────────────────────────────────────────
const cards = [
  { icon: '📉', title: 'Train Loss converges', body: 'Drops smoothly from ~0.55 to near 0 — the model learns efficiently. The final model (dashed) also reaches a low validation loss (~0.065).', color: C.tealDark },
  { icon: '📊', title: 'Why val-loss is noisy', body: 'Only ~340 validation tweets per fold, so a few hard examples swing the curve. Small data → noisy curves. This is expected, not a defect.', color: C.navy },
  { icon: '✅', title: 'Not harmful overfitting', body: 'We select the model by F1, not loss (early stopping on F1). The slight rise is over-confidence raising cross-entropy without lowering accuracy.', color: C.green },
];
cards.forEach((c, i) => {
  const x = 0.35 + i * 3.15;
  slide.addShape(pres.shapes.RECTANGLE, { x, y: 4.05, w: 2.95, h: 1.25, fill: { color: C.white }, line: { color: 'E2E8F0', width: 0.75 },
    shadow: { type: 'outer', color: '000000', blur: 4, offset: 1, angle: 135, opacity: 0.08 } });
  slide.addShape(pres.shapes.RECTANGLE, { x, y: 4.05, w: 2.95, h: 0.04, fill: { color: c.color } });
  slide.addText(c.icon + '  ' + c.title, { x: x + 0.12, y: 4.12, w: 2.75, h: 0.3, fontSize: 12, bold: true, color: c.color, fontFace: 'Calibri', margin: 0 });
  slide.addText(c.body, { x: x + 0.12, y: 4.44, w: 2.75, h: 0.82, fontSize: 9.5, color: C.textBody, fontFace: 'Calibri', margin: 0 });
});

// Bottom proof strip
slide.addText([
  { text: 'Proof it is not overfitting:  ', options: { bold: true, color: C.navy } },
  { text: 'Test F1 (0.986) ≈ CV F1 (0.961). We even tried label smoothing + fewer epochs — it did not improve results, confirming the original config.', options: { color: C.textBody, italic: true } },
], { x: 0.35, y: 5.32, w: 9.3, h: 0.3, fontSize: 10, fontFace: 'Calibri', margin: 0 });

pres.writeFile({ fileName: 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/Loss_Curves_Manchester_slide.pptx' })
  .then(() => console.log('✅ Saved Loss_Curves_Manchester_slide.pptx'))
  .catch(e => { console.error('❌', e); process.exit(1); });
