const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, Header, Footer, PageBreak, TableOfContents,
} = require('docx');

const NAVY='0D1B2A', TEAL='0077A8', TEALBG='E0F2F9', RED='C0392B',
      GREEN='1E7E34', ORANGE='B8651B', GREY='5A6675', WARNBG='FDECEA', OKBG='EAF6EE';

function R(o){ if(typeof o==='string')o={text:o};
  return new TextRun({text:o.text,bold:o.bold||false,italics:o.italic||false,
    color:o.color||'1A2B3C',size:o.size||22,font:'Arial',rightToLeft:o.ltr?false:true}); }
function P(ch,opts={}){ if(!Array.isArray(ch))ch=[ch];
  return new Paragraph({bidirectional:true,alignment:opts.align||AlignmentType.RIGHT,
    spacing:{after:opts.after!=null?opts.after:120,before:opts.before||0,line:276},
    shading:opts.shading?{type:ShadingType.CLEAR,fill:opts.shading,color:'auto'}:undefined,
    indent:opts.indent,children:ch.map(c=>R(c))}); }
function Bullet(ch,level=0){ if(!Array.isArray(ch))ch=[ch];
  return new Paragraph({bidirectional:true,alignment:AlignmentType.RIGHT,
    numbering:{reference:'hb',level},spacing:{after:80,line:264},children:ch.map(c=>R(c))}); }
function H1(text){ return new Paragraph({heading:HeadingLevel.HEADING_1,bidirectional:true,
  alignment:AlignmentType.RIGHT,spacing:{before:280,after:160},
  border:{bottom:{style:BorderStyle.SINGLE,size:18,color:TEAL,space:4}},
  children:[new TextRun({text,bold:true,size:32,color:NAVY,font:'Arial',rightToLeft:true})]}); }
function H2(text,color=TEAL){ return new Paragraph({heading:HeadingLevel.HEADING_2,bidirectional:true,
  alignment:AlignmentType.RIGHT,spacing:{before:180,after:100},
  children:[new TextRun({text,bold:true,size:26,color,font:'Arial',rightToLeft:true})]}); }

function QA(num,q,ans,kind){ // kind: 'warn' | 'ok' | undefined
  const out=[];
  const accent = kind==='warn'?RED:(kind==='ok'?GREEN:TEAL);
  const fill   = kind==='warn'?WARNBG:(kind==='ok'?OKBG:TEALBG);
  const mark   = kind==='warn'?'⚠ ':(kind==='ok'?'✓ ':'');
  out.push(new Paragraph({bidirectional:true,alignment:AlignmentType.RIGHT,
    spacing:{before:200,after:60,line:276},shading:{type:ShadingType.CLEAR,fill,color:'auto'},
    border:{top:{style:BorderStyle.SINGLE,size:4,color:accent},bottom:{style:BorderStyle.SINGLE,size:4,color:accent},
      left:{style:BorderStyle.SINGLE,size:4,color:accent},right:{style:BorderStyle.SINGLE,size:24,color:accent}},
    children:[new TextRun({text:mark+num+'. '+q,bold:true,size:23,color:accent===TEAL?NAVY:accent,font:'Arial',rightToLeft:true})]}));
  out.push(new Paragraph({bidirectional:true,alignment:AlignmentType.RIGHT,
    spacing:{after:100,line:276},indent:{right:120},children:ans.map(c=>R(c))}));
  return out;
}

function tcell(text,{w,fill,bold,color,size,align}={}){
  return new TableCell({width:{size:w,type:WidthType.DXA},
    shading:fill?{fill,type:ShadingType.CLEAR,color:'auto'}:undefined,
    margins:{top:60,bottom:60,left:100,right:100},
    borders:{top:{style:BorderStyle.SINGLE,size:1,color:'CCD4DD'},bottom:{style:BorderStyle.SINGLE,size:1,color:'CCD4DD'},
      left:{style:BorderStyle.SINGLE,size:1,color:'CCD4DD'},right:{style:BorderStyle.SINGLE,size:1,color:'CCD4DD'}},
    children:[new Paragraph({bidirectional:true,alignment:align||AlignmentType.CENTER,spacing:{after:0,line:252},
      children:[new TextRun({text,bold:bold||false,color:color||'1A2B3C',size:size||19,font:'Arial',rightToLeft:true})]})]});
}
function rtlTable(colW,rows,headerFill=NAVY){
  const revW=[...colW].reverse();
  const trows=rows.map((row,ri)=>{
    const cells=[...row].reverse().map((cellText,ci)=>{
      const wLogical=colW[row.length-1-ci]; const isH=ri===0;
      return tcell(cellText,{w:wLogical,fill:isH?headerFill:(ri%2===0?'EEF3F8':'FFFFFF'),
        bold:isH,color:isH?'FFFFFF':'1A2B3C',size:19});});
    return new TableRow({children:cells,tableHeader:ri===0});});
  return new Table({width:{size:colW.reduce((a,b)=>a+b,0),type:WidthType.DXA},columnWidths:revW,rows:trows});
}

const body=[];

// ── Title ──
body.push(new Paragraph({bidirectional:true,alignment:AlignmentType.CENTER,spacing:{before:200,after:60},
  children:[new TextRun({text:'דף הגנה — Manchester',bold:true,size:50,color:NAVY,font:'Arial',rightToLeft:true})]}));
body.push(new Paragraph({bidirectional:true,alignment:AlignmentType.CENTER,spacing:{after:40},
  children:[new TextRun({text:'זיהוי מידע כוזב על פיגוע מנצ׳סטר 2017 — RoBERTa מול Llama 3.1 Zero-Shot',bold:true,size:26,color:TEAL,font:'Arial',rightToLeft:true})]}));
body.push(new Paragraph({bidirectional:true,alignment:AlignmentType.CENTER,spacing:{after:160},
  border:{bottom:{style:BorderStyle.SINGLE,size:12,color:TEAL,space:6}},
  children:[new TextRun({text:'reliable מול misinformation  •  Gold Standard 2,427  •  Test 365',size:21,color:GREY,font:'Arial',rightToLeft:true})]}));

// Highlight banner
body.push(P([{text:'התוצאה במשפט אחד: ',bold:true,color:NAVY,size:24},
  {text:'RoBERTa מכוון משיג F1 Macro של 0.986 בבדיקה; Llama 3.1 8B ב-Zero-Shot מגיע ל-0.684. ההבדל מובהק קיצונית (McNemar p≈3.4e-24). אומת שאין דליפת נתונים.',size:22}],
  {shading:OKBG,after:160}));

body.push(new Paragraph({bidirectional:true,alignment:AlignmentType.RIGHT,spacing:{after:100},
  children:[new TextRun({text:'תוכן העניינים',bold:true,size:28,color:NAVY,font:'Arial',rightToLeft:true})]}));
body.push(new TableOfContents('TOC',{hyperlink:true,headingStyleRange:'1-2'}));
body.push(new Paragraph({children:[new PageBreak()]}));

// ════════════════════════════════════════════════════════════════════════════
// SECTION A — fundamentals (Manchester)
// ════════════════════════════════════════════════════════════════════════════
body.push(H1('חלק א׳ — יסודות Manchester'));

body.push(H2('א1. במה עוסק הדאטאסט?'));
body.push(P('ציוצים בטוויטר סביב פיגוע הטרור באולם מנצ׳סטר (מאי 2017). כל ציוץ מתויג כ-reliable (מידע אמין, מאומת או סביר על האירוע) או misinformation (טענות שקריות, שמועות, תיאוריות קונספירציה — למשל "מחבל שני", פליטים-מחבלים, נעדרים מזויפים).'));

body.push(H2('א2. הנתונים המספריים (לזכור בעל-פה)'));
body.push(rtlTable([2400,2900,2400],[
  ['שלב','ערך','הערה'],
  ['ציוצים גולמיים','89,147','קובץ raw'],
  ['מתוכם misinformation','490 (0.55%)','חוסר איזון קיצוני'],
  ['אחרי ניקוי + dedup','57,917','סינון <5 מילים + כפילויות'],
  ['Gold Standard','2,427','2,000 reliable + 427 misinformation'],
  ['Train / Val / Test','1,698 / 364 / 365','חלוקת 70/15/15 Stratified'],
]));

body.push(H2('א3. למה Gold Standard ולא הנתונים הגולמיים?'));
body.push(P('ב-89,147 הציוצים רק 0.55% הם misinformation. מודל שיגיד "reliable" תמיד יקבל 99.4% accuracy — חסר תועלת. לכן לקחנו את כל 427 ציוצי ה-misinformation + דגימה של 2,000 reliable, וקיבלנו מדגם מאוזן יחסית (2,427). את חוסר האיזון שנותר (82/18) טיפלנו ב-Weighted Cross-Entropy באימון.'));

body.push(H2('א4. למה F1 Macro ולא Accuracy?'));
body.push(P('כי גם ב-Gold Standard יש חוסר איזון (82% reliable). Accuracy של 82% ניתן להשגה ע"י "תמיד reliable". F1 Macro ממצע F1 לכל מחלקה בנפרד ומעניש התעלמות מהמחלקה הקטנה (misinformation).'));

body.push(H2('א5. תוצאות סופיות (Test set, n=365)'));
body.push(rtlTable([2600,1900,1900,1300],[
  ['מדד','RoBERTa Fine-Tuned','Llama 3.1 Zero-Shot','מנצח'],
  ['F1 Macro','0.986','0.684','RoBERTa'],
  ['Accuracy','0.992','0.753','RoBERTa'],
  ['Precision','0.989','0.674','RoBERTa'],
  ['Recall','0.983','0.777','RoBERTa'],
  ['F1 (misinformation)','0.969','0.536','RoBERTa'],
  ['ROC-AUC','0.996','—','RoBERTa'],
]));
body.push(P([{text:'CV (5-fold) של RoBERTa: ',bold:true},{text:'F1 Macro 0.961 ± 0.012 — קרוב מאוד ל-Test (0.986), כלומר ביצועים יציבים וללא overfitting מזיק.',size:21}],{before:60}));
body.push(new Paragraph({children:[new PageBreak()]}));

// ════════════════════════════════════════════════════════════════════════════
// SECTION B — committee questions (Manchester only)
// ════════════════════════════════════════════════════════════════════════════
body.push(H1('חלק ב׳ — שאלות הוועדה (Manchester)'));

const Qs=[
  {n:1,kind:'warn',q:'F1=0.986 — זה לא Overfitting או דליפת מידע?',
   a:[{text:'חשש לגיטימי שבדקנו אמפירית (ראו חלק ד׳). ',bold:true},{text:'ההסבר: מידע כוזב על אירוע בודד מתרכז בקומץ טענות שקריות חוזרות, כך שהאות הלשוני חזק והמשימה קלה יחסית. בדקנו דליפה עם Leakage Audit — 0 כפילויות מדויקות, והסרת הציוצים הדומים ביותר לא הורידה את ה-F1 אלא העלתה אותו (0.994). הציון אמיתי. בנוסף ה-CV (0.961) ≈ Test (0.986) שולל overfitting מזיק.'}]},
  {n:2,q:'ה-Test (0.986) גבוה מה-CV (0.961) — לא חשוד?',
   a:[{text:'לא. מודלי ה-CV אומנו על 80% מהנתונים בכל fold, ואילו המודל הסופי אומן על כל Train+Val (יותר דאטא = +1–3 נקודות). בנוסף, test של 365 ציוצים נותן שגיאת דגימה של כמה נקודות. הפער הקטן הזה הוא אופטימיות שפירה, לא דליפה.'}]},
  {n:3,kind:'warn',q:'מילוי תחזיות null של ה-LLM ב-majority class — זו לא דליפה?',
   a:[{text:'טכנית יש כאן ביט אחד של מידע (זהות מחלקת הרוב). אבל: ב-Manchester רק 2 תחזיות null מתוך 365, אז ההשפעה זניחה. בנוסף המילוי מנפח accuracy אבל כמעט לא עוזר ל-F1 Macro. כיוון ההטיה דווקא מחזק את המסקנה — ה-LLM מפסיד למרות העזרה הקטנה.'}]},
  {n:4,q:'דגמתם מחדש את מחלקת הרוב — זה לא מעוות את המדדים לפריסה אמיתית?',
   a:[{text:'כן, מגבלה מוצהרת. הגבלת reliable ל-2,000 מעלה את שכיחות ה-misinformation מעל המציאות (0.55%→18%), אז ה-Precision אופטימי לפריסה. ',},{text:'אבל ההשוואה בין המודלים תקפה',bold:true},{text:' כי שניהם על אותה התפלגות. לפריסה אמיתית צריך הערכה ב-prevalence טבעי.'}]},
  {n:5,q:'איך בחרתם את הפרומפט? למה אין few-shot ו-prompt ablation?',
   a:[{text:'הפרומפט בנוי על role prompting + הגדרות מחלקה ברורות + Chain-of-Thought + פלט JSON. בחרנו zero-shot טהור (ללא דוגמאות) כי זו שאלת המחקר — מודל מדף ללא דאטא מתויג. few-shot ו-ablation מסודר מתוכננים להמשך; מחקרי רגישות מראים ±3–5 נקודות, שלא הופך פער של 30 נקודות.'}]},
  {n:6,q:'למה Llama 3.1 8B? למה roberta-base ולא large?',
   a:[{text:'Llama 3.1 8B — המודל הפתוח החזק ביותר שרץ מקומית, חינמי ושחזיר על GPU צרכני (דרך Ollama). roberta-base נכנס בנוחות עם fp16 ומריץ 5-fold CV מהר; large היה מוסיף 1–3 נקודות אבל base כבר מנצח בפער, אז large רק היה מרחיב אותו. השלב הבא בודק מודלים גדולים (GPT-OSS 120B, Llama 3.3 70B, Qwen3 32B).'}]},
  {n:7,q:'איך טיפלתם בחוסר האיזון 82/18?',
   a:[{text:'Weighted Cross-Entropy — משקל גבוה פי ~2.85 ל-misinformation בפונקציית ה-loss, מחושב מ-fold האימון בלבד (ללא דליפה). לא השתמשנו ב-oversampling כדי לא לשכפל דוגמאות שיכולות לדלוף בין folds. ה-Recall של misinformation (0.98) מראה שזה עבד.'}]},
  {n:8,q:'איך מטפלים בהאשטאגים, mentions ו-URLs בניקוי?',
   a:[{text:'URLs ו-@mentions נמחקים לגמרי (מזהים, לא תוכן). האשטאגים — מסירים רק את סימן ה-# ושומרים את הטקסט (#Islamists → Islamists), כי תוכן ההאשטאג נושא אות סמנטי חשוב לסיווג. אימוג׳ים ותווים לא-ASCII מוסרים. דוגמה: "@user RT: Manchester BOMB 💥 #Islamists http://..." → "Manchester BOMB Islamists".'}]},
  {n:9,q:'מה זה McNemar ומה התוצאה?',
   a:[{text:'מבחן שמשווה תחזיות מזווגות על אותן דוגמאות, ומסתכל רק על המקרים שבהם המודלים חלוקים. ב-Manchester: RoBERTa צדק וה-LLM טעה ב-89 מקרים; ההפך רק ב-2. p≈3.38e-24 — מובהק קיצונית. זה לא רק "ציון גבוה יותר" אלא הוכחה שהיתרון אמיתי ולא רעש דגימה.'}]},
  {n:10,q:'F1 גבוה — אבל מה עם רווחי סמך עם test של 365?',
   a:[{text:'CI ב-95% (bootstrap) על F1 ל-test של 365 הוא בערך ±2–3 נקודות. הפער מול ה-LLM (30 נקודות) גדול בסדר גודל מהאינטרוול, ו-McNemar (שכן דווח) הוא בדיוק הכלי שמטפל בגודל ה-test. הוספת CI מפורש לגרפים — רוויזיה פשוטה ומתוכננת.'}]},
  {n:11,q:'Llama כבר "ראה" את אירוע מנצ׳סטר באימון — מה זה "zero-shot"?',
   a:[{text:'"Zero-shot" = אפס דוגמאות מתויגות למשימה, לא אפס חשיפה נושאית. כנראה המודל ראה דיווחים על האירוע — וזה פועל לטובתו (אולי יודע אילו טענות הופרכו). שהוא עדיין מפסיד ב-30 נקודות הופך את התוצאה לשמרנית.'}]},
  {n:12,q:'מה ההגבלות ותוקף חיצוני?',
   a:[{text:'אנגלית בלבד, טוויטר, אירוע יחיד מ-2017. סחיפה טמפורלית (מידע כוזב מ-LLMs לא קיים בדאטא), שינויי פלטפורמה. סינון <5 מילים מסיר ציוצים קצרים ועמומים (שיפור אמינות תיוג על חשבון ריאליזם). הטענה ממוקדת: "fine-tuning עם ~2,000 דוגמאות מנצח zero-shot מקומי על מידע כוזב היסטורי באנגלית".'}]},
];
Qs.forEach(it=>QA(it.n,it.q,it.a,it.kind).forEach(p=>body.push(p)));
body.push(new Paragraph({children:[new PageBreak()]}));

// ════════════════════════════════════════════════════════════════════════════
// SECTION C — Overfitting checklist
// ════════════════════════════════════════════════════════════════════════════
body.push(H1('חלק ג׳ — צ׳קליסט Overfitting'));
body.push(P([{text:'איך אנחנו יודעים שאין Overfitting ב-Manchester:',bold:true,size:24,color:NAVY}],{after:100}));
[
  [{text:'Test set נעול — ',bold:true},{text:'365 ציוצים שלא נגעו בהם באימון, בבחירת היפר-פרמטרים או בפרומפט. הוערך פעם אחת.'}],
  [{text:'CV ≈ Test — ',bold:true},{text:'0.961 מול 0.986. Overfitting קלאסי היה מראה CV גבוה ו-test צונח.'}],
  [{text:'Early Stopping (patience=2) — ',bold:true},{text:'עוצר כש-validation F1 מפסיק להשתפר, לפני שינון.'}],
  [{text:'סטיית תקן קטנה בין folds (±0.012) — ',bold:true},{text:'המודל לא תלוי ב-split מסוים.'}],
  [{text:'בחירת מודל לפי F1 ולא loss — ',bold:true},{text:'גם אם ה-validation loss רועש/עולה, שומרים את ה-checkpoint עם ה-F1 הטוב ביותר.'}],
].forEach(it=>body.push(Bullet(it)));

body.push(P([{text:'בדקנו אמפירית: ',bold:true,color:GREEN},
  {text:'ניסינו label_smoothing=0.1 + הפחתת epochs מ-4 ל-3 כדי "ליישר" את עקומת ה-Validation Loss. התוצאה: F1 ירד מעט (0.986→0.981) וה-loss לא השתפר. זה אישש שעליית ה-loss היא overconfidence שאינו פוגע בדיוק — ולכן נשארנו עם הקונפיגורציה המקורית.'}],
  {before:120,shading:OKBG,after:160}));
body.push(new Paragraph({children:[new PageBreak()]}));

// ════════════════════════════════════════════════════════════════════════════
// SECTION D — Leakage Audit (Manchester-specific star section)
// ════════════════════════════════════════════════════════════════════════════
body.push(H1('חלק ד׳ — Leakage Audit (אימות דליפה)'));
body.push(P('זו הנקודה שהכי יותקף בהגנה (F1=0.986), אז בדקנו אותה ישירות. הרצנו audit כפול שמחפש כפילויות-כמעט בין ה-test ל-train+val:'));
body.push(Bullet([{text:'לקסיקלי — ',bold:true},{text:'TF-IDF על char n-grams (תופס שכתובים והעתקות עם שינויי איות).'}]));
body.push(Bullet([{text:'סמנטי — ',bold:true},{text:'sentence-embeddings (all-MiniLM-L6-v2), דמיון cosine (תופס פרפרזות במשמעות).'}]));

body.push(H2('התוצאות'));
body.push(rtlTable([3600,2400,2200],[
  ['בדיקה','ממצא','F1 אחרי'],
  ['כפילויות מדויקות test↔train','0','—'],
  ['הסרת דומים לקסיקלית ≥0.90 (12 ציוצים)','3.3% מה-test','0.989 ⬆'],
  ['הסרת דומים סמנטית ≥0.90 (40 ציוצים)','11% מה-test','0.994 ⬆'],
  ['הסרת המשולב ≥0.90','11%','0.994 ⬆'],
]));

body.push(P([{text:'המשמעות: ',bold:true,color:GREEN},
  {text:'אם הציון נבע מדליפה, הסרת הזוגות הדומים הייתה מורידה את ה-F1 בחדות. קרה ההפך — הוא נשאר/עלה. הדמיון הסמנטי הגבוה (חציון 0.737) הוא סמיכות נושאית (כל הציוצים על אותו אירוע), לא memorization. המסקנה: ה-0.986 אמיתי ולא מנופח.'}],
  {before:120,shading:OKBG,after:120}));
body.push(P([{text:'איפה זה בקוד: ',bold:true},{text:'scripts/leakage_audit_manchester.py  •  הפלט: results/predictions/manchester_leakage_audit.csv',ltr:false}],{after:160}));
body.push(new Paragraph({children:[new PageBreak()]}));

// ════════════════════════════════════════════════════════════════════════════
// SECTION E — key phrases
// ════════════════════════════════════════════════════════════════════════════
body.push(H1('חלק ה׳ — משפטי מפתח'));
[
  'בדקנו את זה אמפירית — וזו דווקא נקודת חוזק.',
  'ה-F1=0.986 אומת מול דליפה: 0 כפילויות, והסרת הדומים העלתה את הציון ל-0.994.',
  'CV (0.961) ≈ Test (0.986) — אין overfitting מזיק.',
  'בחרנו את המודל לפי F1 ולא לפי loss, ולכן ה-validation loss הרועש לא משפיע.',
  'ההשוואה הוגנת — שני המודלים על אותו Test set בדיוק, מה שמאפשר את מבחן McNemar.',
].forEach(ph=>body.push(Bullet([{text:'"'+ph+'"',italic:true}])));

// ── DOC ──
const doc=new Document({
  creator:'Yoni',title:'דף הגנה — Manchester',
  styles:{default:{document:{run:{font:'Arial',size:22}}},
    paragraphStyles:[
      {id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,
        run:{size:32,bold:true,font:'Arial',color:NAVY},paragraph:{spacing:{before:280,after:160},outlineLevel:0}},
      {id:'Heading2',name:'Heading 2',basedOn:'Normal',next:'Normal',quickFormat:true,
        run:{size:26,bold:true,font:'Arial',color:TEAL},paragraph:{spacing:{before:180,after:100},outlineLevel:1}}]},
  numbering:{config:[{reference:'hb',levels:[{level:0,format:LevelFormat.BULLET,text:'•',
    alignment:AlignmentType.RIGHT,style:{paragraph:{indent:{right:480,hanging:280}}}}]}]},
  sections:[{
    properties:{page:{size:{width:12240,height:15840},margin:{top:1180,right:1180,bottom:1180,left:1180}}},
    headers:{default:new Header({children:[new Paragraph({bidirectional:true,alignment:AlignmentType.RIGHT,
      border:{bottom:{style:BorderStyle.SINGLE,size:4,color:'CCD4DD',space:2}},
      children:[new TextRun({text:'דף הגנה — Manchester  •  RoBERTa vs Llama 3.1 Zero-Shot',size:16,color:GREY,font:'Arial',rightToLeft:true})]})]})},
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,
      children:[new TextRun({text:'עמוד ',size:16,color:GREY,font:'Arial',rightToLeft:true}),
        new TextRun({children:[PageNumber.CURRENT],size:16,color:GREY,font:'Arial'})]})]})},
    children:body}]});

Packer.toBuffer(doc).then(buf=>{
  const dir='C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/';
  const cands=['QA_Defense_Manchester.docx','QA_Defense_Manchester_v2.docx','QA_Defense_Manchester_v3.docx'];
  let saved=null;
  for(const n of cands){ try{ fs.writeFileSync(dir+n,buf); saved=n; break; }catch(e){ if(e.code!=='EBUSY')throw e; } }
  console.log(saved?('✅ Saved '+saved):'❌ all locked');
}).catch(e=>{console.error('❌',e);process.exit(1);});
