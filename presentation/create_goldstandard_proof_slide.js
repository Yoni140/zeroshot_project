const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.rtlMode = true;
pres.title = 'Gold Standard vs Full Data — Proof';

const C = {
  darkBg:'0D1B2A', navy:'1A3A5C', teal:'00B4D8', tealDark:'0077A8',
  lightBg:'F4F6FA', white:'FFFFFF', textDark:'1A2B4A', textBody:'2D3748',
  green:'2DC653', red:'E63946', subtle:'90A4AE', orange:'F4A261',
};
const W=10, H=5.625;
const slide = pres.addSlide();
slide.background = { color: C.lightBg };

// Header (RTL)
slide.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:W,h:1.0,fill:{color:C.darkBg}});
slide.addShape(pres.shapes.RECTANGLE,{x:W-0.12,y:0,w:0.12,h:1.0,fill:{color:C.teal}});
slide.addText('למה Gold Standard? — ניסוי השוואתי',{x:0.3,y:0.08,w:9.4,h:0.55,fontSize:23,bold:true,color:C.white,fontFace:'Arial',align:'right',rtlMode:true,margin:0});
slide.addText('אימון על כל הדאטא (פי-24) דווקא הרע את זיהוי ה-misinformation',{x:0.3,y:0.6,w:9.4,h:0.34,fontSize:13,color:C.teal,italic:true,fontFace:'Arial',align:'right',rtlMode:true,margin:0});
slide.addShape(pres.shapes.RECTANGLE,{x:0,y:1.0,w:W,h:0.035,fill:{color:C.teal}});

// ── Comparison table (RTL: header rightmost logical col on the right) ──────────
// columns (logical R->L): Setup | F1 Macro | Accuracy | Recall misinfo | misses
const rows = [
  ['גישה','F1 Macro','Accuracy','Recall (misinfo)','פספוסי misinfo'],
  ['Gold Standard (שלנו)','0.986','0.992','0.983','~1 / 64'],
  ['Full-data ← אותו GOLD test','0.965','0.981','0.891','7 / 64'],
  ['Full-data ← NATURAL test','0.871','0.997 ⚠','0.722','5 / 18'],
];
// logical widths R->L
const colW=[3.05,1.45,1.45,1.7,1.55];
const totalW=colW.reduce((a,b)=>a+b,0); // 9.2
const startX=0.4;
let y=1.3;
const rowH=0.62;
rows.forEach((row,ri)=>{
  // RTL: draw from right. rightmost logical col0 at x = startX+totalW-colW0
  let xRight = startX + totalW;
  const isH = ri===0;
  const isGold = ri===1;
  row.forEach((cell,ci)=>{
    const w=colW[ci];
    xRight -= w;
    const fill = isH?C.darkBg:(isGold?'D5F0DD':(ri%2===0?'EEF3F8':'FFFFFF'));
    slide.addShape(pres.shapes.RECTANGLE,{x:xRight,y,w,h:rowH,fill:{color:fill},line:{color:'CCD4DD',width:0.75}});
    const col = isH?C.white:(isGold&&ci>0?C.green:C.textDark);
    slide.addText(cell,{x:xRight+0.05,y:y+0.04,w:w-0.1,h:rowH-0.08,fontSize:ci===0?12:13,
      bold:isH||isGold||ci===0,color:col,fontFace:'Arial',align:'center',valign:'middle',rtlMode:true,margin:0});
  });
  y+=rowH;
});

// ── Three takeaway cards ──────────────────────────────────────────────────────
const cards=[
  {icon:'📉',title:'יותר דאטא = גרוע יותר',body:'על אותו test: F1 ירד 0.986→0.965, ו-Recall של misinformation צנח 0.983→0.891 (7 פספוסים במקום 1).',color:C.red},
  {icon:'🪤',title:'מלכודת ה-Accuracy',body:'על test טבעי Accuracy=0.997 נראה מושלם — אבל F1 רק 0.871 ו-Recall רק 0.722. בדיוק מה ש-Gold מונע.',color:C.orange},
  {icon:'⚖️',title:'אות חלש מדי',body:'class weight קפץ ל-79.4 (מול 2.85): 363 דוגמאות misinfo טבועות ב-49K reliable. ופי-24 זמן אימון.',color:C.navy},
];
cards.forEach((c,i)=>{
  const x=6.7-i*3.15; // rightmost first
  slide.addShape(pres.shapes.RECTANGLE,{x,y:3.85,w:2.95,h:1.32,fill:{color:C.white},line:{color:'E2E8F0',width:0.75},
    shadow:{type:'outer',color:'000000',blur:4,offset:1,angle:135,opacity:0.08}});
  slide.addShape(pres.shapes.RECTANGLE,{x,y:3.85,w:2.95,h:0.04,fill:{color:c.color}});
  slide.addText(c.title+'  '+c.icon,{x:x+0.12,y:3.92,w:2.71,h:0.3,fontSize:12,bold:true,color:c.color,fontFace:'Arial',align:'right',rtlMode:true,margin:0});
  slide.addText(c.body,{x:x+0.12,y:4.24,w:2.71,h:0.88,fontSize:9.5,color:C.textBody,fontFace:'Arial',align:'right',rtlMode:true,margin:0});
});

// Bottom line
slide.addText([
  {text:'  Random Undersampling היא טכניקה סטנדרטית לחוסר איזון קיצוני — לא קיצור דרך.',options:{color:C.textBody,italic:true,rtlMode:true,fontFace:'Arial'}},
  {text:':המסקנה',options:{bold:true,color:C.green,rtlMode:true,fontFace:'Arial'}},
],{x:0.4,y:5.27,w:9.2,h:0.3,fontSize:11,align:'right',rtlMode:true,margin:0});

pres.writeFile({fileName:'C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/GoldStandard_Proof_slide_HE.pptx'})
  .then(()=>console.log('✅ Saved GoldStandard_Proof_slide_HE.pptx'))
  .catch(e=>{console.error('❌',e);process.exit(1);});
