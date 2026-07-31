const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.title = 'How Transformers Work';

const C = {
  darkBg: '0D1B2A', navy: '1A3A5C', teal: '00B4D8', tealDark: '0077A8',
  tealLight: 'E0F7FF', lightBg: 'F4F6FA', white: 'FFFFFF',
  textDark: '1A2B4A', textBody: '2D3748', orange: 'F4A261',
  green: '2DC653', red: 'E63946', subtle: '90A4AE', purple: '6A0572',
};
const W = 10, H = 5.625;

function addHeaderBar(slide, title, subtitle) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 1.15, fill: { color: C.darkBg } });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 1.15, fill: { color: C.teal } });
  slide.addText(title, { x: 0.28, y: 0.1, w: 9.5, h: 0.6, fontSize: 24, bold: true, color: C.white, fontFace: 'Calibri', margin: 0 });
  if (subtitle) slide.addText(subtitle, { x: 0.28, y: 0.72, w: 9.5, h: 0.35, fontSize: 13, color: C.teal, fontFace: 'Calibri', margin: 0, italic: true });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.15, w: W, h: 0.04, fill: { color: C.teal } });
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill || C.white }, line: { color: 'E2E8F0', width: 0.75 },
    shadow: { type: 'outer', color: '000000', blur: 4, offset: 1, angle: 135, opacity: 0.08 } });
}

// ════════════════════════════════════════════════════════════════════════════════
// SLIDE 1 — How a Transformer Works (the engine inside RoBERTa)
// ════════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeaderBar(slide, 'How a Transformer Works', 'The engine inside RoBERTa — and every modern LLM');

  // LEFT: vertical flow of layers
  const flow = [
    { label: '1. Tokenization', desc: 'Text → sub-word tokens (BPE)', color: C.tealDark },
    { label: '2. Embeddings', desc: 'Each token → 768-dim vector', color: C.navy },
    { label: '3. + Positional Encoding', desc: 'Adds word-order information', color: C.tealDark },
    { label: '4. Self-Attention ×12', desc: 'Each token "looks at" all others', color: C.teal },
    { label: '5. Feed-Forward ×12', desc: 'Non-linear transformation', color: C.navy },
    { label: '6. Output Vectors', desc: 'Context-aware representation', color: C.green },
  ];
  flow.forEach((f, i) => {
    const y = 1.4 + i * 0.65;
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.35, y, w: 4.0, h: 0.55,
      fill: { color: f.color }, line: { color: f.color },
      shadow: { type: 'outer', color: '000000', blur: 3, offset: 1, angle: 135, opacity: 0.1 } });
    slide.addText(f.label, { x: 0.45, y: y + 0.03, w: 3.8, h: 0.27, fontSize: 12, bold: true, color: C.white, fontFace: 'Calibri', margin: 0 });
    slide.addText(f.desc, { x: 0.45, y: y + 0.29, w: 3.8, h: 0.24, fontSize: 9.5, color: 'E8F4FA', fontFace: 'Calibri', margin: 0 });
    if (i < flow.length - 1) slide.addText('▼', { x: 2.1, y: y + 0.54, w: 0.5, h: 0.12, fontSize: 9, color: C.subtle, align: 'center', margin: 0 });
  });

  // RIGHT: self-attention explained (the key idea)
  card(slide, 4.6, 1.35, 5.1, 2.35, '1A1A2E');
  slide.addText('🔑  The Key Idea: Self-Attention', { x: 4.75, y: 1.42, w: 4.8, h: 0.32, fontSize: 13, bold: true, color: C.teal, fontFace: 'Calibri', margin: 0 });
  slide.addText('Every word is interpreted in the context of all other words in the sentence — simultaneously.', {
    x: 4.75, y: 1.78, w: 4.8, h: 0.5, fontSize: 11, color: C.white, fontFace: 'Calibri', margin: 0 });
  // example sentence with attention highlight
  slide.addText([
    { text: '"The bomb claim was ', options: { color: 'CCCCCC' } },
    { text: 'fake', options: { color: C.red, bold: true } },
    { text: '"', options: { color: 'CCCCCC' } },
  ], { x: 4.75, y: 2.35, w: 4.8, h: 0.3, fontSize: 13, fontFace: 'Consolas', margin: 0 });
  slide.addText('→ to classify "fake", the model attends strongly to "claim" and "bomb" — it weighs which words matter for meaning.', {
    x: 4.75, y: 2.7, w: 4.8, h: 0.6, fontSize: 10, color: C.subtle, italic: true, fontFace: 'Calibri', margin: 0 });
  slide.addText('Q · K → attention weights → weighted sum of V', {
    x: 4.75, y: 3.32, w: 4.8, h: 0.3, fontSize: 10, color: C.green, fontFace: 'Consolas', margin: 0 });

  // RIGHT bottom: why it matters
  card(slide, 4.6, 3.85, 5.1, 1.5, 'EBF5FB');
  slide.addText('Why it beats older models (RNN/LSTM):', { x: 4.75, y: 3.92, w: 4.8, h: 0.3, fontSize: 11, bold: true, color: C.tealDark, fontFace: 'Calibri', margin: 0 });
  slide.addText([
    { text: '• Parallel ', options: { bold: true } }, { text: '— processes all words at once (fast on GPU)\n', options: {} },
    { text: '• Long-range ', options: { bold: true } }, { text: '— links distant words directly, no memory decay\n', options: {} },
    { text: '• Pre-trained ', options: { bold: true } }, { text: '— learns language from 160GB text first, then fine-tuned', options: {} },
  ], { x: 4.75, y: 4.24, w: 4.8, h: 1.05, fontSize: 10, color: C.textBody, fontFace: 'Calibri', margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════════
// SLIDE 2 — From Tweet to Prediction: the full pipeline (both models)
// ════════════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addHeaderBar(slide, 'From Tweet to Prediction — All Stages', 'The same input, two different paths');

  // Shared input box at top
  card(slide, 3.0, 1.32, 4.0, 0.62, 'FFFBF0');
  slide.addText('📥  INPUT TWEET', { x: 3.1, y: 1.36, w: 3.8, h: 0.22, fontSize: 9, bold: true, color: C.orange, fontFace: 'Calibri', margin: 0 });
  slide.addText('"Refugees caused the Manchester bombing"', { x: 3.1, y: 1.57, w: 3.8, h: 0.3, fontSize: 10, italic: true, color: C.textDark, fontFace: 'Calibri', margin: 0 });

  // arrows splitting
  slide.addText('Path A — Fine-Tuned', { x: 0.35, y: 2.15, w: 4.5, h: 0.3, fontSize: 13, bold: true, color: C.tealDark, fontFace: 'Calibri', margin: 0, align: 'center' });
  slide.addText('Path B — Zero-Shot', { x: 5.15, y: 2.15, w: 4.5, h: 0.3, fontSize: 13, bold: true, color: C.orange, fontFace: 'Calibri', margin: 0, align: 'center' });

  // PATH A — RoBERTa
  const pathA = [
    'Clean text + tokenize (BPE, max 128 tokens)',
    'RoBERTa-base encoder (12 layers, 125M params)',
    '[CLS] vector → Linear head (768 → 2)',
    'Softmax → probabilities per class',
    'argmax → "misinformation" (0.94)',
  ];
  pathA.forEach((t, i) => {
    const y = 2.5 + i * 0.56;
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.35, y, w: 4.45, h: 0.46, fill: { color: i === pathA.length - 1 ? C.green : C.tealDark }, line: { color: 'FFFFFF', width: 0.5 } });
    slide.addText(`${i + 1}. ${t}`, { x: 0.45, y, w: 4.3, h: 0.46, fontSize: 10, bold: i === pathA.length - 1, color: C.white, fontFace: 'Calibri', margin: 0, valign: 'middle' });
    if (i < pathA.length - 1) slide.addText('▼', { x: 2.4, y: y + 0.45, w: 0.4, h: 0.1, fontSize: 8, color: C.subtle, align: 'center', margin: 0 });
  });

  // PATH B — LLM
  const pathB = [
    'Wrap tweet in a Chain-of-Thought prompt',
    'LLM reads prompt (no training, frozen weights)',
    'Generates reasoning + JSON answer',
    'Parse JSON → extract "label" field',
    'Result → "misinformation" (conf 0.87)',
  ];
  pathB.forEach((t, i) => {
    const y = 2.5 + i * 0.56;
    slide.addShape(pres.shapes.RECTANGLE, { x: 5.15, y, w: 4.45, h: 0.46, fill: { color: i === pathB.length - 1 ? C.green : C.orange }, line: { color: 'FFFFFF', width: 0.5 } });
    slide.addText(`${i + 1}. ${t}`, { x: 5.25, y, w: 4.3, h: 0.46, fontSize: 10, bold: i === pathB.length - 1, color: C.white, fontFace: 'Calibri', margin: 0, valign: 'middle' });
    if (i < pathB.length - 1) slide.addText('▼', { x: 7.2, y: y + 0.45, w: 0.4, h: 0.1, fontSize: 8, color: C.subtle, align: 'center', margin: 0 });
  });

  // bottom contrast note
  card(slide, 0.35, 5.32, 9.3, 0.0); // spacer (no-op visual)
  slide.addText([
    { text: 'Key difference:  ', options: { bold: true, color: C.navy } },
    { text: 'Path A learned from labeled tweets and updates its weights; Path B only follows instructions with frozen, pre-trained knowledge.', options: { color: C.textBody } },
  ], { x: 0.35, y: 5.28, w: 9.3, h: 0.32, fontSize: 10, italic: true, fontFace: 'Calibri', margin: 0 });
}

pres.writeFile({ fileName: 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/Transformer_Explained_slides.pptx' })
  .then(() => console.log('✅ Saved Transformer_Explained_slides.pptx'))
  .catch(e => { console.error('❌', e); process.exit(1); });
