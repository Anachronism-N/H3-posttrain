// Native editable PowerPoint, using the bundled Artifact Tool (no bitmap slides).
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import './pdf_standard_fonts.cjs';

const here=path.dirname(fileURLToPath(import.meta.url));
const root=path.dirname(here);
const runtime=process.env.ARTIFACT_NODE_MODULES || 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-611-0_26.1112.611.0_x64__3k8sg7r9htsxt/dependencies/node/node_modules';
const {Presentation,PresentationFile,registerPdfFontFace}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const fontDir=process.env.FIGURE_FONT_DIR || 'C:/Windows/Fonts';
for (const [file,style,weight] of [['times.ttf','normal',400],['timesbd.ttf','normal',700],['timesi.ttf','italic',400],['timesbi.ttf','italic',700]]) {
  registerPdfFontFace({family:'Times New Roman',style,weight,source:new Uint8Array(await fs.readFile(path.join(fontDir,file)))});
}
const scene=JSON.parse(await fs.readFile(path.join(here,'native_scenes.json'),'utf8'));
const ppt=Presentation.create({slideSize:{width:scene.width,height:scene.height}});
const privateDir=path.join(root,'.pptx-build');
await fs.mkdir(privateDir,{recursive:true});
await fs.mkdir(path.join(root,'pptx_preview'),{recursive:true});
const stats=[];
for(const [index,spec] of scene.slides.entries()){
  const slide=ppt.slides.add();slide.background.fill='#ffffff';
  let texts=0,shapes=0,images=0;
  for(const [n,item] of spec.items.entries()){
    const pos={left:item.x,top:item.y,width:item.w,height:item.h};
    const name=`${spec.title}/${String(n+1).padStart(4,'0')}/${item.name}`;
    if(item.kind==='text'){
      const sh=slide.shapes.add({geometry:'textbox',name,position:{...pos,rotation:item.rotation||0},fill:'none',line:{fill:'none',width:0}});
      sh.text=item.text;
      sh.text.style={fontSize:item.size,typeface:item.font,bold:item.bold,italic:item.italic,
        color:item.fill,wrap:'none',autoFit:'none',insets:0,alignment:'left',verticalAlignment:'top'};
      texts++;
    }else if(item.kind==='image'){
      const mime=item.data.split(';')[0].replace('data:','');
      const blob=new Uint8Array(Buffer.from(item.data.split(',').slice(1).join(',').replace(/\s/g,''),'base64'));
      const image=slide.images.add({blob,contentType:mime,alt:'Unchanged source-video frame',position:pos,fit:'contain'});
      image.flipVertical=!!item.flipV;image.flipHorizontal=!!item.flipH;
      images++;
    }else{
      const line={style:item.dash?'dash':'solid',fill:item.stroke||'none',width:item.stroke==='none'?0:item.sw};
      const conf={name,position:pos,fill:item.fill||'none',line};
      if(item.kind==='shape'){
        conf.geometry=item.geometry;
        if(item.radius)conf.borderRadius=item.radius;
      }else{
        conf.geometry='custom';
        conf.customPaths=[{width:item.w,height:item.h,commands:item.commands}];
      }
      slide.shapes.add(conf);shapes++;
    }
  }
  slide.speakerNotes.text=`Source: ${spec.file}. Native editable reconstruction of HereWeChoreography figure. Text boxes, geometry and paths are editable; only unchanged video photographs are image objects. No new benchmark results or claims. Source file and generator scripts accompany this deck.`;
  stats.push({slide:index+1,source:spec.file,texts,shapes,images});
  console.log(JSON.stringify(stats.at(-1)));
}
await (await PresentationFile.exportPptx(ppt)).save(path.join(privateDir,'candidate.pptx'));
await fs.writeFile(path.join(root,'native_pptx_objects.json'),JSON.stringify(stats,null,2));
for(let i=0;i<ppt.slides.items.length;i++){
  const slide=ppt.slides.items[i];
  const blob=await ppt.export({slide,format:'png',scale:2});
  await fs.writeFile(path.join(root,'pptx_preview',`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
  const pdfBlob=await ppt.export({slide,format:'pdf'});
  await fs.writeFile(path.join(root,'pptx_preview',`slide-${i+1}.pdf`),new Uint8Array(await pdfBlob.arrayBuffer()));
  console.log('Rendered',i+1);
}
await fs.writeFile(path.join(privateDir,'native_scene_exported.json'),JSON.stringify({slides:scene.slides.length,source:'Artifact Tool',canvas:[scene.width,scene.height]}));
console.log('Candidate exported and rendered; run finalization before delivery.');
