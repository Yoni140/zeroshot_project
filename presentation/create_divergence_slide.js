const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.rtlMode = true;
pres.title = 'Loss vs F1 Divergence';

const C = {
  darkBg:'0D1B2A', navy:'1A3A5C', teal:'00B4D8', tealDark:'0077A8',
  lightBg:'F4F6FA', white:'FFFFFF', textDark:'1A2B4A', textBody:'2D3748',
  green:'1E7E34', red:'C0392B', subtle:'90A4AE', orange:'B8651B',
};
const W=10, H=5.625;
const IMG='C:/Users/yoni1/Desktop/ZEROSHO_CODE/results/_fulldata_experiment/loss_vs_f1_divergence.png';

const slide=pres.addSlide();
slide.background={color:C.lightBg};

// Header
slide.addShape(pres.shapes.RECTANGLE,{x:0,y:0,w:W,h:1.0,fill:{color:C.darkBg}});
slide.addShape(pres.shapes.RECTANGLE,{x:W-0.12,y:0,w:0.12,h:1.0,fill:{color:C.teal}});
slide.addText('ה-Validation Loss וה-F1 מתפצלים',{x:0.3,y:0.08,w:9.4,h:0.55,fontSize:23,bold:true,color:C.white,fontFace:'Arial',align:'right',rtlMode:true,margin:0});
slide.addText('הוכחה: עליית ה-loss אינה overfitting — היא overconfidence שלא פוגע בדיוק',{x:0.3,y:0.6,w:9.4,h:0.34,fontSize:13,color:C.teal,italic:true,fontFace:'Arial',align:'right',rtlMode:true,margin:0});
slide.addShape(pres.shapes.RECTANGLE,{x:0,y:1.0,w:W,h:0.035,fill:{color:C.teal}});

// Graph (left, LTR content) — 1350x707 ~ ratio 1.91 ; width 5.7 -> height 2.98
slide.addImage({path:IMG,x:0.35,y:1.35,w:5.7,h:2.98});
slide.addText('מודל הדאטא המלא — Manchester (validation לכל epoch)',{x:0.35,y:4.35,w:5.7,h:0.3,fontSize:10,italic:true,color:C.subtle,fontFace:'Arial',align:'center',rtlMode:true,margin:0});

// Right panel: explanation
slide.addShape(pres.shapes.RECTANGLE,{x:6.25,y:1.35,w:3.4,h:3.55,fill:{color:C.white},line:{color:'E2E8F0',width:0.75},
  shadow:{type:'outer',color:'000000',blur:4,offset:1,angle:135,opacity:0.08}});
slide.addText('מה רואים בין epoch 2 ל-3',{x:6.4,y:1.46,w:3.1,h:0.35,fontSize:14,bold:true,color:C.navy,fontFace:'Arial',align:'right',rtlMode:true,margin:0});

// loss up box
slide.addShape(pres.shapes.RECTANGLE,{x:6.4,y:1.9,w:3.1,h:0.62,fill:{color:'FDECEA'},line:{color:C.red,width:1}});
slide.addText([
  {text:'  0.174 → 0.312  ▲',options:{bold:true,color:C.red,rtlMode:false,fontFace:'Arial'}},
  {text:'Loss',options:{bold:true,color:C.red,rtlMode:false,fontFace:'Arial'}},
],{x:6.5,y:1.96,w:2.9,h:0.5,fontSize:13,align:'right',valign:'middle',rtlMode:true,margin:0});
slide.addText('ה-loss עלה — "אמור" להיות רע',{x:6.5,y:2.28,w:2.9,h:0.22,fontSize:9.5,color:C.textBody,fontFace:'Arial',align:'right',rtlMode:true,margin:0});

// f1 up box
slide.addShape(pres.shapes.RECTANGLE,{x:6.4,y:2.62,w:3.1,h:0.62,fill:{color:'EAF6EE'},line:{color:C.green,width:1}});
slide.addText([
  {text:'  0.846 → 0.896  ▲',options:{bold:true,color:C.green,rtlMode:false,fontFace:'Arial'}},
  {text:'F1 Macro',options:{bold:true,color:C.green,rtlMode:false,fontFace:'Arial'}},
],{x:6.5,y:2.68,w:2.9,h:0.5,fontSize:13,align:'right',valign:'middle',rtlMode:true,margin:0});
slide.addText('ה-F1 השתפר — בפועל טוב יותר',{x:6.5,y:3.0,w:2.9,h:0.22,fontSize:9.5,color:C.textBody,fontFace:'Arial',align:'right',rtlMode:true,margin:0});

slide.addText('שניהם עולים יחד — בלתי אפשרי אם ה-loss היה מדד אמין לאיכות. המודל פשוט נעשה בטוח יותר (cross-entropy עולה) בלי לטעות יותר.',
  {x:6.4,y:3.4,w:3.1,h:1.0,fontSize:10.5,color:C.textBody,fontFace:'Arial',align:'right',rtlMode:true,margin:0});

// Bottom conclusion strip
slide.addShape(pres.shapes.RECTANGLE,{x:0.35,y:4.78,w:9.3,h:0.62,fill:{color:'EAF6EE'},line:{color:C.green,width:1}});
slide.addText([
  {text:'  בחרנו את המודל לפי F1 ולא לפי loss (early stopping על F1). לכן ה-validation loss הרועש לא משפיע על התוצאות — וזה מאושש בשני המודלים, Gold Standard וגם דאטא מלא.',options:{color:C.textDark,rtlMode:true,fontFace:'Arial'}},
  {text:':המסקנה',options:{bold:true,color:C.green,rtlMode:true,fontFace:'Arial'}},
],{x:0.5,y:4.84,w:9.0,h:0.5,fontSize:11,align:'right',valign:'middle',rtlMode:true,margin:0});

pres.writeFile({fileName:'C:/Users/yoni1/Desktop/ZEROSHO_CODE/presentation/Loss_F1_Divergence_slide_HE.pptx'})
  .then(()=>console.log('✅ Saved Loss_F1_Divergence_slide_HE.pptx'))
  .catch(e=>{console.error('❌',e);process.exit(1);});
