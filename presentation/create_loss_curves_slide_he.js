const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.rtlMode = true;
pres.title = 'Training Loss Curves — Manchester (Hebrew)';

const C = {
  darkBg: '0D1B2A', navy: '1A3A5C', teal: '00B4D8', tealDark: '0077A8',
  lightBg: 'F4F6FA', white: 'FFFFFF', textDark: '1A2B4A', textBody: '2D3748',
  green: '2DC653', red: 'E63946', subtle: '90A4AE', orange: 'F4A261',
};
const W = 10, H = 5.625;
const IMG = 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/results/figures/manchester/manchester_roberta_loss_curves.png';

// Hebrew text run helper (RTL)
function he(text, opts = {}) {
  return { text, options: Object.assign({ rtlMode: true, fontFace: 'Arial' }, opts) };
}

const slide = pres.addSlide();
slide.background = { color: C.lightBg };

// Header (text right-aligned for Hebrew)
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 1.0, fill: { color: C.darkBg } });
slide.addShape(pres.shapes.RECTANGLE, { x: W - 0.12, y: 0, w: 0.12, h: 1.0, fill: { color: C.teal } });
slide.addText('עקומות Loss של אימון ו-Validation — RoBERTa (5-Fold CV)', {
  x: 0.3, y: 0.08, w: 9.4, h: 0.55, fontSize: 23, bold: true, color: C.white, fontFace: 'Arial', align: 'right', rtlMode: true, margin: 0 });
slide.addText('Manchester — התכנסות האימון, ולמה ה-validation loss הרועש אינו בעיה', {
  x: 0.3, y: 0.6, w: 9.4, h: 0.34, fontSize: 13, color: C.teal, italic: true, fontFace: 'Arial', align: 'right', rtlMode: true, margin: 0 });
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.0, w: W, h: 0.035, fill: { color: C.teal } });

// Figure
slide.addImage({ path: IMG, x: 0.7, y: 1.2, w: 8.6, h: 2.45 });

// Captions under the two panels (English labels in the plot stay LTR; captions in Hebrew)
slide.addText('Train Loss — ירידה חלקה לכמעט 0  ◀', {
  x: 0.7, y: 3.66, w: 4.3, h: 0.28, fontSize: 11, bold: true, color: C.tealDark, fontFace: 'Arial', align: 'center', rtlMode: true, margin: 0 });
slide.addText('▶  Validation Loss — רועש בין ה-folds', {
  x: 5.0, y: 3.66, w: 4.3, h: 0.28, fontSize: 11, bold: true, color: C.orange, fontFace: 'Arial', align: 'center', rtlMode: true, margin: 0 });

// ── Three explanation cards (right-to-left order: first card on the right) ─────
const cards = [
  { icon: '📉', title: 'Train Loss מתכנס', body: 'יורד חלק מ-~0.55 לכמעט 0 — המודל לומד ביעילות. גם המודל הסופי (קו מקווקו) מגיע ל-validation loss נמוך (~0.065).', color: C.tealDark },
  { icon: '📊', title: 'למה ה-val loss רועש', body: 'רק ~340 ציוצי validation בכל fold, אז כמה דוגמאות קשות מזיזות את העקומה. דאטא קטן ← עקומות רועשות. צפוי, לא פגם.', color: C.navy },
  { icon: '✅', title: 'לא Overfitting מזיק', body: 'בוחרים את המודל לפי F1, לא לפי loss (early stopping על F1). העלייה הקלה היא ביטחון-יתר שמעלה cross-entropy בלי להוריד דיוק.', color: C.green },
];
// place right-to-left: card 0 rightmost
cards.forEach((c, i) => {
  const x = 6.7 - i * 3.15;  // rightmost first
  slide.addShape(pres.shapes.RECTANGLE, { x, y: 4.05, w: 2.95, h: 1.25, fill: { color: C.white }, line: { color: 'E2E8F0', width: 0.75 },
    shadow: { type: 'outer', color: '000000', blur: 4, offset: 1, angle: 135, opacity: 0.08 } });
  slide.addShape(pres.shapes.RECTANGLE, { x, y: 4.05, w: 2.95, h: 0.04, fill: { color: c.color } });
  slide.addText(c.title + '  ' + c.icon, { x: x + 0.12, y: 4.12, w: 2.71, h: 0.3, fontSize: 12, bold: true, color: c.color, fontFace: 'Arial', align: 'right', rtlMode: true, margin: 0 });
  slide.addText(c.body, { x: x + 0.12, y: 4.44, w: 2.71, h: 0.82, fontSize: 9.5, color: C.textBody, fontFace: 'Arial', align: 'right', rtlMode: true, margin: 0 });
});

// Bottom proof strip (RTL)
slide.addText([
  he('Test F1 (0.986) ≈ CV F1 (0.961). אפילו ניסינו label smoothing + פחות epochs — זה לא שיפר את התוצאה, מה שאישש את הקונפיגורציה המקורית.  ', { color: C.textBody, italic: true }),
  he(':ההוכחה שאין overfitting', { bold: true, color: C.navy }),
], { x: 0.35, y: 5.32, w: 9.3, h: 0.3, fontSize: 10, fontFace: 'Arial', align: 'right', rtlMode: true, margin: 0 });

pres.writeFile({ fileName: 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/Loss_Curves_Manchester_slide_HE.pptx' })
  .then(() => console.log('✅ Saved Loss_Curves_Manchester_slide_HE.pptx'))
  .catch(e => { console.error('❌', e); process.exit(1); });
