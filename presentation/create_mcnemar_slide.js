const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.title = 'McNemar Test — Manchester';

const C = {
  darkBg: '0D1B2A', navy: '1A3A5C', teal: '00B4D8', tealDark: '0077A8',
  lightBg: 'F4F6FA', white: 'FFFFFF', textDark: '1A2B4A', textBody: '2D3748',
  green: '2DC653', red: 'E63946', subtle: '90A4AE', orange: 'F4A261',
};
const W = 10, H = 5.625;

const slide = pres.addSlide();
slide.background = { color: C.lightBg };

// Header bar
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 1.15, fill: { color: C.darkBg } });
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 1.15, fill: { color: C.teal } });
slide.addText("McNemar's Test — Statistical Significance", { x: 0.28, y: 0.1, w: 9.5, h: 0.6, fontSize: 24, bold: true, color: C.white, fontFace: 'Calibri', margin: 0 });
slide.addText('Manchester — RoBERTa Fine-Tuned vs Llama 3.1 Zero-Shot (n = 365)', { x: 0.28, y: 0.72, w: 9.5, h: 0.35, fontSize: 13, color: C.teal, italic: true, fontFace: 'Calibri', margin: 0 });
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.15, w: W, h: 0.04, fill: { color: C.teal } });

// ── 2x2 contingency table (left) ─────────────────────────────────────────────
const tx = 0.45, ty = 1.65;
const colLabelW = 1.55, cellW = 1.85;
const rowH = 0.62, headH = 0.5;

// corner empty
function box(x, y, w, h, fill, line) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: line || 'FFFFFF', width: 1 } });
}
function txt(t, x, y, w, h, opts = {}) {
  slide.addText(t, { x, y, w, h, fontSize: opts.size || 12, bold: opts.bold || false, color: opts.color || C.textDark, fontFace: 'Calibri', align: opts.align || 'center', valign: 'middle', margin: 0 });
}

// title above table
txt('Agreement Matrix', tx, ty - 0.02, colLabelW + cellW * 2, 0.3, { size: 13, bold: true, color: C.navy, align: 'left' });

const gy = ty + 0.35;
// Column headers
box(tx + colLabelW, gy, cellW, headH, C.green);
txt('Zero-Shot\nCORRECT', tx + colLabelW, gy, cellW, headH, { size: 11, bold: true, color: C.white });
box(tx + colLabelW + cellW, gy, cellW, headH, C.red);
txt('Zero-Shot\nWRONG', tx + colLabelW + cellW, gy, cellW, headH, { size: 11, bold: true, color: C.white });

// Row 1 — RoBERTa correct
let ry = gy + headH;
box(tx, ry, colLabelW, rowH, C.green);
txt('RoBERTa\nCORRECT', tx, ry, colLabelW, rowH, { size: 11, bold: true, color: C.white });
box(tx + colLabelW, ry, cellW, rowH, 'EAF6EE');
txt('273', tx + colLabelW, ry, cellW, rowH, { size: 20, bold: true, color: C.textDark });
box(tx + colLabelW + cellW, ry, cellW, rowH, 'D6EAF8');
txt('89', tx + colLabelW + cellW, ry, cellW, rowH, { size: 20, bold: true, color: C.tealDark });

// Row 2 — RoBERTa wrong
ry += rowH;
box(tx, ry, colLabelW, rowH, C.red);
txt('RoBERTa\nWRONG', tx, ry, colLabelW, rowH, { size: 11, bold: true, color: C.white });
box(tx + colLabelW, ry, cellW, rowH, 'FDEBD0');
txt('2', tx + colLabelW, ry, cellW, rowH, { size: 20, bold: true, color: C.orange });
box(tx + colLabelW + cellW, ry, cellW, rowH, 'F4F6FA');
txt('1', tx + colLabelW + cellW, ry, cellW, rowH, { size: 20, bold: true, color: C.subtle });

// discordant labels
txt('b (discordant)', tx + colLabelW + cellW, ry - rowH - 0.0, cellW, 0.2, { size: 8, color: C.tealDark, align: 'center' });
slide.addText('b = 89', { x: tx + colLabelW + cellW, y: gy + headH + rowH - 0.18, w: cellW, h: 0.16, fontSize: 8, italic: true, color: C.tealDark, align: 'center', margin: 0 });
slide.addText('c = 2', { x: tx + colLabelW, y: ry + rowH - 0.18, w: cellW, h: 0.16, fontSize: 8, italic: true, color: C.orange, align: 'center', margin: 0 });

// ── Right panel: interpretation + result ──────────────────────────────────────
const rx = 6.4;
slide.addShape(pres.shapes.RECTANGLE, { x: rx, y: 1.65, w: 3.2, h: 3.55, fill: { color: C.white }, line: { color: 'E2E8F0', width: 0.75 },
  shadow: { type: 'outer', color: '000000', blur: 4, offset: 1, angle: 135, opacity: 0.08 } });

slide.addText('Result', { x: rx + 0.2, y: 1.78, w: 2.8, h: 0.35, fontSize: 16, bold: true, color: C.navy, fontFace: 'Calibri', margin: 0 });

// big p-value
slide.addShape(pres.shapes.RECTANGLE, { x: rx + 0.2, y: 2.2, w: 2.8, h: 0.95, fill: { color: 'EBF5FB' }, line: { color: C.teal, width: 1 } });
slide.addText('p-value', { x: rx + 0.2, y: 2.27, w: 2.8, h: 0.25, fontSize: 11, color: C.tealDark, fontFace: 'Calibri', align: 'center', margin: 0 });
slide.addText('3.38 × 10⁻²⁴', { x: rx + 0.2, y: 2.5, w: 2.8, h: 0.45, fontSize: 22, bold: true, color: C.tealDark, fontFace: 'Calibri', align: 'center', margin: 0 });

// significant badge
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: rx + 0.2, y: 3.3, w: 2.8, h: 0.5, fill: { color: C.green }, rectRadius: 0.08, line: { color: C.green } });
slide.addText('✓  Statistically Significant', { x: rx + 0.2, y: 3.3, w: 2.8, h: 0.5, fontSize: 13, bold: true, color: C.white, fontFace: 'Calibri', align: 'center', valign: 'middle', margin: 0 });

slide.addText([
  { text: 'RoBERTa beat Zero-Shot on ', options: {} },
  { text: '89', options: { bold: true, color: C.tealDark } },
  { text: ' tweets; Zero-Shot beat RoBERTa on only ', options: {} },
  { text: '2', options: { bold: true, color: C.orange } },
  { text: '. The gap is far too large to be chance (α = 0.05).', options: {} },
], { x: rx + 0.2, y: 3.95, w: 2.8, h: 1.15, fontSize: 11, color: C.textBody, fontFace: 'Calibri', margin: 0, valign: 'top' });

// ── Bottom explanatory strip ──────────────────────────────────────────────────
slide.addShape(pres.shapes.RECTANGLE, { x: 0.45, y: 5.28, w: 5.55, h: 0.0 });
slide.addText([
  { text: 'How it works:  ', options: { bold: true, color: C.navy } },
  { text: 'McNemar looks only at the discordant cells (b, c) — where the models disagree — and tests whether b ≠ c.', options: { color: C.textBody, italic: true } },
], { x: 0.45, y: 5.18, w: 9.1, h: 0.35, fontSize: 10, fontFace: 'Calibri', margin: 0 });

pres.writeFile({ fileName: 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/McNemar_Manchester_slide.pptx' })
  .then(() => console.log('✅ Saved McNemar_Manchester_slide.pptx'))
  .catch(e => { console.error('❌', e); process.exit(1); });
