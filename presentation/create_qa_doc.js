const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, Header, Footer, PageBreak, TableOfContents,
} = require('docx');

// ── Colors ────────────────────────────────────────────────────────────────────
const NAVY = '0D1B2A', TEAL = '0077A8', TEALBG = 'E0F2F9';
const RED = 'C0392B', GREEN = '1E7E34', ORANGE = 'B8651B';
const GREY = '5A6675', LIGHTBG = 'F4F6FA', CARDBG = 'FBFCFE';
const WARNBG = 'FDECEA';

// ── RTL helpers ─────────────────────────────────────────────────────────────────
// run: {text, bold, italic, color, size, ltr}
function R(opts) {
  if (typeof opts === 'string') opts = { text: opts };
  return new TextRun({
    text: opts.text,
    bold: opts.bold || false,
    italics: opts.italic || false,
    color: opts.color || '1A2B3C',
    size: opts.size || 22,
    font: 'Arial',
    rightToLeft: opts.ltr ? false : true,
  });
}

// Hebrew paragraph (RTL, right-aligned). children: array of run-opts or strings
function P(children, opts = {}) {
  if (!Array.isArray(children)) children = [children];
  return new Paragraph({
    bidirectional: true,
    alignment: opts.align || AlignmentType.RIGHT,
    spacing: { after: opts.after != null ? opts.after : 120, before: opts.before || 0, line: 276 },
    shading: opts.shading ? { type: ShadingType.CLEAR, fill: opts.shading, color: 'auto' } : undefined,
    indent: opts.indent,
    border: opts.border,
    children: children.map(c => R(c)),
  });
}

// Bullet (RTL)
function Bullet(children, level = 0) {
  if (!Array.isArray(children)) children = [children];
  return new Paragraph({
    bidirectional: true,
    alignment: AlignmentType.RIGHT,
    numbering: { reference: 'hebullets', level },
    spacing: { after: 80, line: 264 },
    children: children.map(c => R(c)),
  });
}

// Section heading
function H1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    bidirectional: true,
    alignment: AlignmentType.RIGHT,
    spacing: { before: 280, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: TEAL, space: 4 } },
    children: [new TextRun({ text, bold: true, size: 32, color: NAVY, font: 'Arial', rightToLeft: true })],
  });
}
function H2(text, color = TEAL) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    bidirectional: true,
    alignment: AlignmentType.RIGHT,
    spacing: { before: 180, after: 100 },
    children: [new TextRun({ text, bold: true, size: 26, color, font: 'Arial', rightToLeft: true })],
  });
}

// ── Q&A block ───────────────────────────────────────────────────────────────────
// q: question string, a: array of run-opts/strings (the answer), warn: bool, num: int
function QA(num, q, answerRuns, warn = false) {
  const out = [];
  // Question line — shaded box
  out.push(new Paragraph({
    bidirectional: true,
    alignment: AlignmentType.RIGHT,
    spacing: { before: 200, after: 60, line: 276 },
    shading: { type: ShadingType.CLEAR, fill: warn ? WARNBG : TEALBG, color: 'auto' },
    border: {
      top: { style: BorderStyle.SINGLE, size: 4, color: warn ? RED : TEAL },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: warn ? RED : TEAL },
      left: { style: BorderStyle.SINGLE, size: 4, color: warn ? RED : TEAL },
      right: { style: BorderStyle.SINGLE, size: 24, color: warn ? RED : TEAL },
    },
    children: [
      new TextRun({ text: (warn ? '⚠ ' : '') + num + '. ' + q, bold: true, size: 23, color: warn ? RED : NAVY, font: 'Arial', rightToLeft: true }),
    ],
  }));
  // Answer
  out.push(new Paragraph({
    bidirectional: true,
    alignment: AlignmentType.RIGHT,
    spacing: { after: 100, line: 276 },
    indent: { right: 120 },
    children: answerRuns.map(c => R(c)),
  }));
  return out;
}

// ── Table builder (RTL: first logical column appears on the right) ─────────────
function tcell(text, { w, fill, bold, color, size, align } = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: 'CCD4DD' },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: 'CCD4DD' },
      left: { style: BorderStyle.SINGLE, size: 1, color: 'CCD4DD' },
      right: { style: BorderStyle.SINGLE, size: 1, color: 'CCD4DD' },
    },
    children: [new Paragraph({
      bidirectional: true,
      alignment: align || AlignmentType.CENTER,
      spacing: { after: 0, line: 252 },
      children: [new TextRun({ text, bold: bold || false, color: color || '1A2B3C', size: size || 19, font: 'Arial', rightToLeft: true })],
    })],
  });
}

// rows: array of arrays (logical order, will be reversed for RTL). colW logical order.
function rtlTable(colW, rows, headerFill = NAVY) {
  const revW = [...colW].reverse();
  const trows = rows.map((row, ri) => {
    const cells = [...row].reverse().map((cellText, ci) => {
      const wLogical = colW[row.length - 1 - ci];
      const isHeader = ri === 0;
      return tcell(cellText, {
        w: wLogical,
        fill: isHeader ? headerFill : (ri % 2 === 0 ? 'EEF3F8' : 'FFFFFF'),
        bold: isHeader,
        color: isHeader ? 'FFFFFF' : '1A2B3C',
        size: isHeader ? 19 : 19,
      });
    });
    return new TableRow({ children: cells, tableHeader: ri === 0 });
  });
  return new Table({
    width: { size: colW.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: revW,
    rows: trows,
  });
}

// ════════════════════════════════════════════════════════════════════════════════
// BUILD CONTENT
// ════════════════════════════════════════════════════════════════════════════════
const body = [];

// ── Title block ──────────────────────────────────────────────────────────────────
body.push(new Paragraph({
  bidirectional: true, alignment: AlignmentType.CENTER,
  spacing: { before: 200, after: 60 },
  children: [new TextRun({ text: 'דף הכנה להגנה', bold: true, size: 52, color: NAVY, font: 'Arial', rightToLeft: true })],
}));
body.push(new Paragraph({
  bidirectional: true, alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({ text: 'Zero-Shot מול Fine-Tuned LLMs לזיהוי מידע כוזב בציוצים', bold: true, size: 28, color: TEAL, font: 'Arial', rightToLeft: true })],
}));
body.push(new Paragraph({
  bidirectional: true, alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: TEAL, space: 6 } },
  children: [new TextRun({ text: 'שאלות ותשובות • צ׳קליסט Overfitting • משפטי מפתח', size: 22, color: GREY, font: 'Arial', rightToLeft: true })],
}));

// warning banner
body.push(new Paragraph({
  bidirectional: true, alignment: AlignmentType.RIGHT,
  spacing: { before: 60, after: 200, line: 276 },
  shading: { type: ShadingType.CLEAR, fill: WARNBG, color: 'auto' },
  border: {
    top: { style: BorderStyle.SINGLE, size: 4, color: RED }, bottom: { style: BorderStyle.SINGLE, size: 4, color: RED },
    left: { style: BorderStyle.SINGLE, size: 4, color: RED }, right: { style: BorderStyle.SINGLE, size: 24, color: RED },
  },
  children: [new TextRun({ text: 'השאלות המסוכנות ביותר: 1, 5 — לתרגל אותן קודם. (שאלה 13 על דליפה כבר נבדקה ונסגרה — נקודת חוזק.)', bold: true, size: 22, color: RED, font: 'Arial', rightToLeft: true })],
}));

// ── TOC ──────────────────────────────────────────────────────────────────────────
body.push(new Paragraph({
  bidirectional: true, alignment: AlignmentType.RIGHT, spacing: { after: 100 },
  children: [new TextRun({ text: 'תוכן העניינים', bold: true, size: 28, color: NAVY, font: 'Arial', rightToLeft: true })],
}));
body.push(new TableOfContents('Table of Contents', { hyperlink: true, headingStyleRange: '1-2' }));
body.push(new Paragraph({ children: [new PageBreak()] }));

// ════════════════════════════════════════════════════════════════════════════════
// SECTION A — Fundamentals
// ════════════════════════════════════════════════════════════════════════════════
body.push(H1('חלק א׳ — שאלות יסוד (חובה לדעת בעל-פה)'));

body.push(H2('א1. מה שאלת המחקר?'));
body.push(P('האם מודל שפה גדול, ללא אף דוגמת אימון (Zero-Shot), יכול לזהות מידע כוזב באותה רמה כמו RoBERTa שעבר Fine-Tuning על אלפי ציוצים מתויגים?'));

body.push(H2('א2. מהי התשובה בשורה אחת?'));
body.push(P([
  { text: 'לא — ' },
  { text: 'RoBERTa מנצח בכל שלושת הדאטאסטים', bold: true },
  { text: ' בפער של 20–30 נקודות F1 Macro (0.986 / 0.913 / 0.598 מול 0.684 / 0.613 / 0.398), וההבדל מובהק סטטיסטית בצורה קיצונית (McNemar exact: Manchester p≈3.4e-24, Monkeypox p≈5e-29, PHEME p<1e-80). אבל מודל 8B מקומי הוא הקצה התחתון — Phase 2 בודק מודלים גדולים בהרבה.' },
]));

body.push(H2('א3. מה זה Fine-Tuning?'));
body.push(P('לוקחים מודל שאומן מראש על טקסט כללי (roberta-base, 125M פרמטרים), מוסיפים שכבת סיווג (768→2), וממשיכים לאמן את כל המשקולות על הדוגמאות המתויגות שלנו עם Learning Rate קטן (2e-5) כדי לא למחוק את הידע הקיים (Catastrophic Forgetting).'));

body.push(H2('א4. מה זה Zero-Shot?'));
body.push(P('המודל לא רואה אף דוגמה מתויגת מהמשימה. נותנים לו פרומפט שמגדיר את המחלקות ומבקשים סיווג — הוא מסתמך רק על הידע מהאימון המקדים שלו. "Zero-shot" = אפס דוגמאות למשימה, לא אפס חשיפה לנושא (ראו שאלה 23).'));

body.push(H2('א5. למה F1 Macro ולא Accuracy?'));
body.push(P('הדאטא לא מאוזן. ב-Manchester מודל שאומר תמיד "reliable" מקבל כ-82% accuracy על ה-gold standard (ו-99.4% על הנתונים הגולמיים!) בלי לזהות אף misinformation. F1 Macro ממצע F1 לכל מחלקה בנפרד — מעניש התעלמות מהמחלקה הקטנה.'));

body.push(H2('א6. הנתונים המרכזיים (לזכור!)'));
body.push(rtlTable(
  [1700, 1400, 1400, 1400, 1400],
  [
    ['', 'Manchester', 'Monkeypox', 'PHEME', 'הערה'],
    ['גולמי', '89,147', '~6,287', '62,445', 'ציוצים גולמיים'],
    ['Gold Standard', '2,427', '3,043', '12,887', 'מדגם מתויג'],
    ['Train / Val / Test', '1,698/364/365', '2,130/456/457', '9,020/1,933/1,934', '70/15/15'],
    ['RoBERTa Test F1', '0.986', '0.913', '0.598', 'מנצח'],
    ['RoBERTa CV F1', '0.961±0.012', '0.893±0.014', '0.599±0.007', '5-fold'],
    ['Llama 3.1 ZS F1', '0.684', '0.613', '0.398', 'zero-shot'],
  ],
));
body.push(P([{ text: 'שימו לב: ב-PHEME הרוב הוא דווקא rumour (10,887) — הפוך משאר הדאטאסטים.', italic: true, color: GREY, size: 20 }], { before: 80 }));
body.push(new Paragraph({ children: [new PageBreak()] }));

// ════════════════════════════════════════════════════════════════════════════════
// SECTION B — 25 committee questions
// ════════════════════════════════════════════════════════════════════════════════
body.push(H1('חלק ב׳ — 25 שאלות הוועדה (מהקשה לקל)'));

const Qs = [
  { n: 1, warn: true, q: 'Manchester F1=0.986 — זה לא Overfitting או דליפת מידע?',
    a: [{ text: 'חשש לגיטימי שבדקנו. ' }, { text: 'ההסבר המרכזי: ', bold: true }, { text: 'הדאטא ספציפי לאירוע אחד — מידע כוזב על פיגוע בודד מתרכז בקומץ טענות שקריות חוזרות (נעדרים מזויפים, "מחבל שני"), כך שהאות הלשוני חזק במיוחד והמשימה באמת קלה יותר מזיהוי כללי. הסיכון האמיתי: כפילויות-כמעט (retweets עם עריכה קלה) שניקוי exact-match לא תופס, ויכלו לנחות גם ב-train וגם ב-test. אימות מוצע: בדיקת דמיון embedding/MinHash בין test ל-train וסינון זוגות מעל 0.9. ה-CV היציב (0.961±0.012) מראה שהתוצאה לפחות יציבה, לא split מוצלח במקרה.' }] },
  { n: 2, q: 'ה-Test (0.986) גבוה מה-CV (0.961) — לא חשוד?',
    a: [{ text: 'לא: מודלי ה-CV אומנו על 80% מהנתונים בכל fold, ואילו המודל הסופי אומן על כל Train+Val (יותר דאטא = +1–3 נקודות באופן טיפוסי). בנוסף, test של כ-365 ציוצים נותן שגיאת דגימה של כמה נקודות. הדפוס לא חוזר פתולוגית בדאטאסטים האחרים (Monkeypox +2, PHEME שטוח) — אופטימיות קטנה ושפירה, לא דליפה. דליפה אמיתית הייתה מציגה פער 5–10 נקודות בכל הדאטאסטים.' }] },
  { n: 3, q: 'מי תייג את הנתונים? מה ה-Inter-Annotator Agreement?',
    a: [{ text: 'בשקיפות: השתמשנו בתוויות של יוצרי הדאטאסטים המקוריים ולא תייגנו מחדש. PHEME תויג ע"י עיתונאים עם מתודולוגיה מתועדת; Manchester ו-Monkeypox מגיעים מהמחקרים המקוריים. אין לנו מדדי הסכמה משלנו — מגבלה אמיתית: רעש תיוג מציב תקרה לא ידועה ל-F1 ויכול להסביר חלק מציון PHEME הנמוך. עבודה עתידית: תיוג מחדש של כ-200 ציוצים לכל דאטאסט עם שני מתייגים ודיווח Cohen’s kappa.' }] },
  { n: 4, q: 'דגמתם מחדש את מחלקת הרוב — זה לא מעוות את המדדים לפריסה אמיתית?',
    a: [{ text: 'כן, וזו מגבלת scope מוצהרת. הגבלת reliable ל-2,000 מעלה את שכיחות ה-misinformation הרבה מעל המציאות, ולכן ה-Precision שלנו אופטימי ביחס לפריסה. ' }, { text: 'ההשוואה בין המודלים נשארת תקפה', bold: true }, { text: ' כי שניהם מתמודדים עם אותה התפלגות בדיוק — וזו שאלת המחקר. לפריסה אמיתית צריך הערכה ב-prevalence טבעי או מדדים שאינם תלויי-שכיחות (ROC-AUC, Sensitivity/Specificity).' }] },
  { n: 5, warn: true, q: 'מילוי תחזיות null של ה-LLM ב-majority class מהתפלגות האמת — זו לא דליפת test?',
    a: [{ text: 'תשובה כנה: טכנית כן, יש כאן ביט אחד של מידע. שלושה גורמים מגבילים את הנזק: (א) שיעור ה-nulls נמוך (ספרות בודדות באחוזים, חוץ מ-PHEME עם 118 מתוך 1,934 = 6%); (ב) המילוי מנפח accuracy אבל כמעט לא עוזר ל-F1 Macro על המחלקה הקטנה; (ג) זהות מחלקת הרוב ידועה מתיעוד הדאטאסט בלי לגעת בתוויות. הפרוטוקול הנקי יותר — מילוי במחלקה קבועה a-priori או ספירת nulls כשגיאות — היה מאומץ ברוויזיה; שיעור ה-null מדווח כך שניתן לחסום את ההשפעה.' }] },
  { n: 6, q: 'המילוי הזה לא מעדיף את ה-LLM ומטה את ההשוואה?',
    a: [{ text: 'שולית כן — המילוי יכול רק לעזור ל-LLM. אבל ' }, { text: 'כיוון ההטיה מחזק את המסקנה', bold: true }, { text: ': גם עם היתרון הקטן הזה ה-LLM מפסיד ב-20–30 נקודות בכל דאטאסט. הסרת המילוי הייתה רק מרחיבה את הפער. המילוי קיים כדי ש-McNemar יהיה תקף (שני המודלים על אותן דוגמאות בדיוק).' }] },
  { n: 7, q: 'ב-PHEME הרוב הוא rumour — הפוך מהאחרים. השלכות?',
    a: [{ text: 'הפייפליין מטפל בזה: weighted cross-entropy לפי שכיחויות לכל דאטאסט, ו-F1 Macro סימטרי למחלקות. ההשלכות המהותיות: ה-LLM עם prior של "רוב הטקסט אינו rumour" נלחם נגד ה-prior של הדאטאסט — חלק מההסבר ל-0.398 הנמוך. וגם: "rumour" שונה מושגית מ-"misinformation" — שמועה יכולה להתברר כנכונה — ולכן שלושת הדאטאסטים לא מודדים בדיוק את אותו מבנה.' }] },
  { n: 8, q: 'פער של 39 נקודות בין Manchester (0.986) ל-PHEME (0.598) באותו מודל — איך?',
    a: [{ text: 'שלושה גורמים: (א) הבחנת rumour/not_rumour פרגמטית ולא לקסיקלית — מוגדרת לפי סטטוס אימות, לא ניסוח; (ב) PHEME מכסה 9 אירועים שונים — שונות נושאית גבוהה; (ג) חוסר האיזון 10,887/2,000 מושך F1 Macro מטה דרך המחלקה הקטנה. CV≈Test (0.599 מול 0.598) מראה שהמודל בתקרה האמיתית שלו. הפער הזה הוא ממצא בעצמו: ביצועי גלאי תלויים מאוד במשימה ובאירוע.' }] },
  { n: 9, q: 'למה אין baseline של Few-Shot?',
    a: [{ text: 'החלטת scope מודעת: שאלת המחקר כיוונה לקיצון המעניין מעשית — האם מודל מדף ללא דאטא מתויג מחליף fine-tuning. Few-shot כבר דורש דוגמאות מתויגות והנדסת בחירת-דוגמאות, ומטשטש את הטענה. העלות: לא נדע אם 8–16 דוגמאות היו סוגרות את הפער (הספרות מציעה תוספת 5–15 נקודות). ראשון ברשימת ההמשך — התשתית של Phase 2 תומכת בזה.' }] },
  { n: 10, q: 'פרומפט אחד לכל דאטאסט בלי Ablation — אולי התוצאה משקפת את הפרומפט ולא את המודל?',
    a: [{ text: 'לא ניתן לשלול לגמרי רגישות לפרומפט. מיטיגציות: הפרומפט פותח על דוגמאות פיתוח (לא test), עוקב אחרי תבניות CoT+JSON מהספרות, והוחזק קבוע על כל ה-test. מחקרי רגישות מדווחים ±3–5 נקודות — לא הופך פער של 20–30. Ablation מסודר (3–5 פרפרזות, עם/בלי CoT) מתוכנן ל-Phase 2.' }] },
  { n: 11, q: 'למה דווקא Llama 3.1 8B — לא בניתם קרב לא הוגן שתכננתם לנצח?',
    a: [{ text: 'האילוץ היה מעשי: פייפליין מקומי, שחזיר וחינמי על GPU צרכני — Llama 3.1 8B היה המודל הפתוח החזק ביותר שעומד בזה דרך Ollama. ההשוואה עונה על שאלה צרה: "האם מודל 8B מקומי מחליף fine-tuning?" והתשובה (לא) אולי לא תקפה ב-70B+. בדיוק לכן קיים Phase 2 (GPT-OSS 120B, Llama 3.3 70B, Qwen3 32B דרך Groq). הניסוח הוגן-לפרמטרים: 125M מכוון מנצח 8B zero-shot — תוצאה שימושית לפריסות מוגבלות-משאבים.' }] },
  { n: 12, q: 'למה roberta-base ולא large? למה Full Fine-Tuning ולא LoRA?',
    a: [{ text: 'roberta-base נכנס בנוחות ל-GPU מקומי עם fp16 ומריץ 5-fold CV בזמן סביר; large היה מוסיף 1–3 נקודות אבל בפי-3 compute — ומאחר ש-base כבר מנצח בפער, large היה רק מרחיב אותו. Full FT של 125M זול מספיק כך ש-PEFT מיותר; LoRA רלוונטי אם נעשה fine-tune ל-8B — התא החסר והמעניין ביותר במטריצה (fine-tuned LLM מול fine-tuned RoBERTa), רשום כעבודה עתידית.' }] },
  { n: 13, warn: false, q: '✓ Dedup רק על exact-match — איך אתם יודעים שאין near-duplicates בין train ל-test?',
    a: [{ text: 'בדקנו את זה — ואין דליפה. ', bold: true, color: GREEN }, { text: 'הרצנו Leakage Audit כפול (לקסיקלי TF-IDF char n-grams + סמנטי עם sentence-embeddings) בין כל ציוץ test לכל ה-train+val. הממצאים: (א) 0 כפילויות מדויקות; (ב) הסרת 40 הציוצים הדומים ביותר סמנטית (≥0.90, 11% מה-test) העלתה את ה-F1 ל-0.994 במקום להוריד; (ג) הסרת הדומים לקסיקלית (≥0.90) → 0.989. ' }, { text: 'אם הציון נבע מדליפה, הסרת הזוגות הדומים הייתה מורידה את ה-F1 בחדות — קרה ההפך.', bold: true }, { text: ' הדמיון הסמנטי הגבוה (חציון 0.737) הוא סמיכות נושאית (כל הציוצים על אותו אירוע), לא memorization. המסקנה: ה-0.986 אמיתי ולא מנופח.' }] },
  { n: 14, q: 'McNemar — האם התנאים מתקיימים? תיקנתם להשוואות מרובות?',
    a: [{ text: 'תנאי הזיווג מתקיים: שני המודלים חוזים על אותם ציוצים בדיוק, והשתמשנו בגרסת exact binomial על הזוגות הדיסקורדנטיים. ערכים שחושבו מקבצי התחזיות: Manchester b=89/c=2 → p≈3.4e-24; Monkeypox b=159/c=18 → p≈5e-29; PHEME b−c=750 → p<1e-80. השוואות מרובות: 3 מבחנים ללא Bonferroni בגרסה הראשונית — ביקורת הוגנת — אבל ה-p-values כה קיצוניים שהם שורדים כל תיקון. ב-Phase 2 עם 5 מודלים נחיל Holm-Bonferroni מפורשות.' }] },
  { n: 15, q: 'הכל אנגלית, טוויטר, 2014–2022 — מה אפשר לטעון על תוקף חיצוני?',
    a: [{ text: 'מעט מאוד, ואנחנו אומרים זאת: סחיפה טמפורלית (מידע כוזב מ-LLMs לא קיים בדאטא), שינויי פלטפורמה (X, community notes), אנגלית בלבד. בנוסף — זיהום בכיוון ה-LLM: אימון Llama 3.1 כנראה כולל דיווחים על האירועים, מה שאמור לעזור לו — והוא עדיין מפסיד, מה שהופך את התוצאה לשמרנית. הטענה ממוקדת: "fine-tuning עם כ-2,000 דוגמאות מנצח zero-shot מקומי על מידע כוזב היסטורי באנגלית בטוויטר".' }] },
  { n: 16, q: 'כמה ה-LLM באמת ניתן לשחזור?',
    a: [{ text: 'אמצעים סטנדרטיים: temperature=0, גרסת מודל מקובעת ב-Ollama, פרומפטים קבועים, וכל הפלטים הגולמיים נשמרו ב-CSV. temperature=0 לא מבטיח דטרמיניזם ביט-מדויק (אי-אסוציאטיביות של GPU), אז ריצה חוזרת עשויה להזיז F1 בשבריר נקודה. טענת השחזור היא סטטיסטית ולא ביטית: הארטיפקטים מאפשרים לאמת כל מספר מהפלטים השמורים. במודלי ענן (Phase 2) השחזור חלש יותר — לכן מתועדים מזהי מודל ותאריכים.' }] },
  { n: 17, q: 'ה-keyword fallback יכול לפרש לא נכון תשובות?',
    a: [{ text: 'ה-fallback מופעל רק כשפרסור JSON נכשל (מיעוט קטן), והסיכון העיקרי: תגובת CoT שמזכירה את שתי התוויות ("this is not misinformation") נתפסת על המילה הלא נכונה. כל התגובות הגולמיות שמורות וניתנות לאודיט; בהינתן גודל הפער, שום שיעור שגיאות פרסור סביר לא הופך מסקנה. לתיקון ב-Phase 2: אכיפת structured output (format=json).' }] },
  { n: 18, q: 'למה Weighted Cross-Entropy ולא Oversampling או Focal Loss?',
    a: [{ text: 'השיטה הפשוטה ביותר שלא משכפלת דוגמאות (שכפול היה מסתבך עם dedup ושלמות ה-folds) ולא מוסיפה היפר-פרמטר (gamma של focal דורש כוונון). לא בוצע ablation — מודים. Monkeypox (0.913 עם יחס 2:1) מראה שהשקלול עבד; ב-PHEME אי אפשר להפריד "התרופה לא הספיקה" מ-"המשימה קשה מהותית" — ניסוי oversampling/focal על PHEME בלבד יפתור, מהיר לעתיד.' }] },
  { n: 19, q: 'מה רווחי הסמך על F1 עם test כל כך קטן?',
    a: [{ text: 'לא דיווחנו bootstrap CI בגרסה הראשונה — וצריך. ל-test של כ-365 ה-CI ב-95% הוא בערך ±2–3 נקודות; PHEME (כ-1,934) הדוק יותר. קריטית: הפערים (20–30 נקודות) גדולים בסדר גודל מהאינטרוולים, ו-McNemar — שכן דווח — הוא בדיוק הכלי שמטפל בגודל ה-test המשותף. הוספת CI לכל גרף — רוויזיה פשוטה ומתוכננת.' }] },
  { n: 20, q: 'סינון ציוצים מתחת ל-5 מילים מסיר בדיוק את הקשים — לא מנפח ציונים?',
    a: [{ text: 'כן, הפילטר מחליף ריאליזם באמינות תיוג: "OMG is this true??" לא ניתן לתיוג משמעותי גם ע"י אדם. המשמעות: הציונים תקפים לציוצים בעלי תוכן, ומערכת פרוסה צריכה מדיניות נפרדת לקצרים (abstain או הקשר שרשור). שני המודלים נהנים שווה — ההשוואה תקפה; המספרים האבסולוטיים הם חסם עליון ביחס לזרם מלא, וספירות ההסרה מתועדות.' }] },
  { n: 21, q: 'אולי ה-CoT עצמו confound? בדקתם בלי?',
    a: [{ text: 'לא — ablation חסר. CoT אומץ כי הספרות מראה עקבית שיפור בסיווג תלוי-נימוק, ועקבות הנימוק השמורות מראות עיסוק בתוכן הטענה. אבל CoT יכול "לשכנע מודל לצאת" משיפוט מהיר נכון, ומאריך פלטים (יותר כשלי פרסור). כל התגובות שמורות — השוואת no-CoT דורשת רק ריצה חוזרת, מתוזמנת ל-Phase 2.' }] },
  { n: 22, q: 'אתיקה: מה ההשלכות של פריסת מסווג עם שיעורי שגיאה כאלה?',
    a: [{ text: 'ברמת Manchester אפשר להגן על שימוש כ-triage; ברמת PHEME (0.598) המערכת תסמן בטעות כמויות גדולות של פוסטים לגיטימיים — False Positive במסווג מידע כוזב הוא שגיאת השתקת-ביטוי עם נזק אמיתי, שפוגעת באופן לא פרופורציונלי בדיאלקטים לא-סטנדרטיים. עמדתנו: מערכות כאלה צריכות רק לדרג להמלצת בדיקה אנושית, לעולם לא auto-moderation. קיים גם חשש dual-use: מסווג מפורסם מלמד יריבים מה חומק.' }] },
  { n: 23, q: 'Llama אומן על דאטא שכולל את האירועים — מה בכלל "zero-shot" כאן?',
    a: [{ text: 'כמעט בוודאות המודל ראה דיווחים על מנצ’סטר 2017, Monkeypox 2022 ואירועי PHEME. "Zero-shot" = אפס דוגמאות מתויגות למשימה, לא אפס חשיפה נושאית. הזיהום פועל לטובת ה-LLM: הוא אולי יודע אילו טענות הופרכו. שהוא עדיין מפסיד ב-20–30 נקודות הופך את התוצאה לשמרנית. השאלה הפתוחה האמיתית: אירוע חדש אחרי ה-cutoff, שבו גם ל-RoBERTa אין דאטא רלוונטי — שם zero-shot אולי ינצח. עבודה עתידית.' }] },
  { n: 24, q: 'misinformation מול rumour — אתם בכלל מודדים משימה אחת?',
    a: [{ text: 'מבנים קרובים אך שונים: misinformation מבוסס-שקריות, rumour מבוסס-סטטוס-אימות (שמועה יכולה להתברר נכונה). לכן לא ממצעים בין דאטאסטים ולא טוענים לציון "זיהוי מידע כוזב" אחיד. העיצוב: שלושה case studies מקבילים עם pipeline משותף. ההטרוגניות אינפורמטיבית — יתרון ה-fine-tuning מחזיק בשני סוגי המבנים בעוד הקושי האבסולוטי משתנה דרמטית.' }] },
  { n: 25, q: 'עוד שלושה חודשים — מה השינוי האחד שהכי יחזק את הפרויקט?',
    a: [{ text: 'אודיט ה-near-duplicates על Manchester (שאלה 13) — ה-0.986 הוא המספר שהכי יותקף והכי קשה להגן עליו בלעדיו; שבוע עבודה שמאשר או מתקן. שני: השלמת Phase 2 — הופך את המסקנה מ-"8B לא מחליף fine-tuning" לאמירה מודעת-scale. שלישי: few-shot baseline + prompt ablation שיחסמו כמה מהגירעון הוא ארטיפקט פרומפט.' }] },
];

Qs.forEach(item => {
  QA(item.n, item.q, item.a, item.warn).forEach(p => body.push(p));
});
body.push(new Paragraph({ children: [new PageBreak()] }));

// ════════════════════════════════════════════════════════════════════════════════
// SECTION C — Overfitting checklist
// ════════════════════════════════════════════════════════════════════════════════
body.push(H1('חלק ג׳ — צ׳קליסט Overfitting (לשליפה מהירה)'));
body.push(P([{ text: 'איך אנחנו יודעים שאין Overfitting:', bold: true, size: 24, color: NAVY }], { after: 100 }));

const ofItems = [
  [{ text: 'Test set נעול — ', bold: true }, { text: '15% שלא נגעו בו באימון, בבחירת היפר-פרמטרים או בעיצוב פרומפט. הוערך פעם אחת בלבד.' }],
  [{ text: 'CV ≈ Test — ', bold: true }, { text: 'Manchester 0.961→0.986, Monkeypox 0.893→0.913, PHEME 0.599→0.598. Overfitting קלאסי היה מראה CV גבוה ו-test צונח — אצלנו ההפך הקטן והמוסבר (המודל הסופי ראה יותר דאטא).' }],
  [{ text: 'Early Stopping — ', bold: true }, { text: 'האימון נעצר כש-validation F1 מפסיק להשתפר (patience=2), לפני שינון.' }],
  [{ text: 'סטיות תקן קטנות בין folds ', bold: true }, { text: '(±0.007–0.014) — המודל לא תלוי ב-split מסוים.' }],
  [{ text: 'אין Oversampling — ', bold: true }, { text: 'אין שכפול דוגמאות שיכול לדלוף בין folds; הטיפול בחוסר איזון בפונקציית ה-loss בלבד, ומשקולות מחושבות מ-fold האימון בלבד.' }],
  [{ text: 'Zero-Shot לא יכול לעשות Overfit ', bold: true }, { text: 'בהגדרה — לא ראה דאטא אימון; temperature=0 הופך אותו לדטרמיניסטי.' }],
];
ofItems.forEach(it => body.push(Bullet(it)));

body.push(P([
  { text: 'בדקנו את זה אמפירית: ', bold: true, color: GREEN },
  { text: 'ניסינו label_smoothing=0.1 + הפחתת epochs מ-4 ל-3 כדי "ליישר" את עקומת ה-Validation Loss. התוצאה: Test F1 ירד מעט (0.986→0.981), CV ירד (0.961→0.951), והגרף לא השתפר. זה אישש שעליית ה-loss היא overconfidence שאינו פוגע בדיוק — בחרנו את המודל לפי F1 ולא לפי loss, וה-Test (0.986) ≈ CV (0.961) מוכיח שאין overfitting מזיק. לכן נשארנו עם הקונפיגורציה המקורית.' },
], { before: 120, shading: 'EAF6EE', after: 120 }));

body.push(P([
  { text: 'הנקודה שהיתה פגיעה — נסגרה: ', bold: true, color: GREEN },
  { text: 'dedup הוא exact-match, אז חששנו ש-near-duplicates יחצו בין train ל-test. הרצנו Leakage Audit (לקסיקלי + סמנטי): 0 כפילויות מדויקות, והסרת כל הזוגות הדומים (≥0.90) לא הורידה את ה-F1 אלא העלתה אותו (0.986→0.994). אין דליפה — ה-0.986 אמיתי.' },
], { before: 60, shading: 'EAF6EE', after: 160 }));
body.push(new Paragraph({ children: [new PageBreak()] }));

// ════════════════════════════════════════════════════════════════════════════════
// SECTION D — Key phrases
// ════════════════════════════════════════════════════════════════════════════════
body.push(H1('חלק ד׳ — משפטי מפתח לפתיחת תשובות'));
const phrases = [
  'זו ביקורת הוגנת, וכך טיפלנו בה...',
  'המגבלה הזו מוצהרת ב-limitations; היא משפיעה על המספרים האבסולוטיים אך לא על ההשוואה, כי שני המודלים מתמודדים עם אותה התפלגות.',
  'כיוון ההטיה כאן דווקא מחזק את המסקנה, לא מחליש אותה.',
  'כל הפלטים הגולמיים שמורים — הניתוח ניתן לאודיט מלא.',
  'זה בדיוק מה ש-Phase 2 נועד לבדוק.',
];
phrases.forEach(ph => body.push(Bullet([{ text: '"' + ph + '"', italic: true }])));

// ════════════════════════════════════════════════════════════════════════════════
// DOCUMENT
// ════════════════════════════════════════════════════════════════════════════════
const doc = new Document({
  creator: 'Yoni',
  title: 'דף הכנה להגנה — Zero-Shot Project',
  styles: {
    default: { document: { run: { font: 'Arial', size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: 'Arial', color: NAVY },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Arial', color: TEAL },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: 'hebullets',
        levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.RIGHT,
          style: { paragraph: { indent: { right: 480, hanging: 280 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1180, right: 1180, bottom: 1180, left: 1180 },
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        bidirectional: true, alignment: AlignmentType.RIGHT,
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: 'CCD4DD', space: 2 } },
        children: [new TextRun({ text: 'דף הכנה להגנה — Zero-Shot vs Fine-Tuned LLMs', size: 16, color: GREY, font: 'Arial', rightToLeft: true })],
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: 'עמוד ', size: 16, color: GREY, font: 'Arial', rightToLeft: true }),
                   new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY, font: 'Arial' })],
      })] }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const dir = 'C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/';
  const candidates = ['QA_Defense_Prep.docx', 'QA_Defense_Prep_v2.docx', 'QA_Defense_Prep_v3.docx', 'QA_Defense_Prep_FINAL.docx'];
  let saved = null;
  for (const name of candidates) {
    try { fs.writeFileSync(dir + name, buf); saved = name; break; }
    catch (e) { if (e.code !== 'EBUSY') throw e; }
  }
  if (saved) console.log('✅ Saved ' + saved + (saved !== candidates[0] ? '  (earlier names were locked/open in Word)' : ''));
  else console.log('❌ All candidate filenames are locked. Close Word and retry.');
}).catch(e => { console.error('❌', e); process.exit(1); });
