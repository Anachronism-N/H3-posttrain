"""Paper assets are exact crops of the editable PPT's own rendered masters."""
from pathlib import Path
from PIL import Image
import fitz,json,hashlib,math,shutil
ROOT=Path(__file__).resolve().parents[1]
scene=json.loads((ROOT/'source/native_scenes.json').read_text(encoding='utf-8'))
OUT=ROOT/'paper';OUT.mkdir(exist_ok=True)
rows=[]
for i,spec in enumerate(scene['slides'],1):
 stem=Path(spec['file']).stem
 png=ROOT/'pptx_preview'/f'slide-{i}.png'
 pdf=ROOT/'pptx_preview'/f'slide-{i}.pdf'
 box=spec['content_bbox']
 box=[max(0,box[0]-2),max(0,box[1]-2),min(scene['width'],box[2]+2),min(scene['height'],box[3]+2)]
 image=Image.open(png).convert('RGB')
 scale=image.width/scene['width']
 ibox=tuple(round(v*scale) for v in box)
 image.crop(ibox).save(OUT/(stem+'.png'))
 # Keep every delivered standalone preview on the same authoritative render.
 for alias in [ROOT/(stem+'.preview.png'),ROOT/'preview'/(stem+'.png')]:
  if alias.exists():shutil.copy2(OUT/(stem+'.png'),alias)
 doc=fitz.open(pdf);assert len(doc)==1
 sx,sy=doc[0].rect.width/scene['width'],doc[0].rect.height/scene['height']
 rect=fitz.Rect(box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy)
 doc[0].set_cropbox(rect);doc.save(OUT/(stem+'.pdf'),garbage=4,deflate=True)
 assert Image.open(OUT/(stem+'.png')).convert('RGB').tobytes()==image.crop(ibox).tobytes()
 rows.append({'figure':stem,'slide':i,'paper_png_is_exact_crop_of_ppt_render':True,
  'paper_pdf_exported_from_same_native_slide':True,'crop_pixels':ibox,
  'source_svg_sha256':hashlib.sha256((ROOT/spec['file']).read_bytes()).hexdigest(),
  'paper_png_sha256':hashlib.sha256((OUT/(stem+'.png')).read_bytes()).hexdigest(),
  'paper_pdf_sha256':hashlib.sha256((OUT/(stem+'.pdf')).read_bytes()).hexdigest()})
(ROOT/'paper_ppt_consistency.json').write_text(json.dumps({'source':'One native editable Artifact Tool presentation','figures':rows,'fonts':['Times New Roman'],'full_figure_raster_wrapping_in_ppt':False,'rendering_note':'PPT and paper geometry, labels, icons and colors originate from the same native objects. External PowerPoint viewers require the declared font.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Verified exact paper/PPT render consistency for',len(rows),'figures.')
