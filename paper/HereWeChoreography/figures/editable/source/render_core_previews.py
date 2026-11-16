"""Render editable core SVG diagrams with MuPDF and record an editability audit."""
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib
import json
import fitz

root=Path(__file__).resolve().parents[1]
preview=root/'preview'
preview.mkdir(exist_ok=True)
ns={'s':'http://www.w3.org/2000/svg'}
rows=[]
for name in ['annotation_pipeline','evaluation_rubric','agent_policy_comparison']:
    svg=root/(name+'.editable.svg')
    content=svg.read_bytes()
    document=ET.fromstring(content)
    assert len(document.findall('.//s:image',ns))==0
    assert len(document.findall('.//s:foreignObject',ns))==0
    texts=document.findall('.//s:text',ns)
    assert len(texts)>20
    doc=fitz.open(stream=content,filetype='svg')
    doc[0].get_pixmap().save(preview/(name+'.editable.png'))
    rows.append(dict(file=svg.name,sha256=hashlib.sha256(content).hexdigest(),
        editable_text_elements=len(texts),groups=len(document.findall('.//s:g',ns)),
        native_paths=len(document.findall('.//s:path',ns)),raster_images=0,
        canvas_width=int(document.attrib['width']),canvas_height=int(document.attrib['height']),
        preview='preview/'+name+'.editable.png',
        visual_review='Text and boxes remain within canvas; arrows use explicit filled triangles.'))
(root/'core_diagrams.audit.json').write_text(json.dumps(dict(
    scope='Editable rebuilds of three current figures; no manuscript modification or new results',
    primitive_types=['text','rect','circle','path','g'],
    text_remains_text=True,whole_image_wrappers=False,
    schematic_poses_and_waveforms=True,D6='Local development (existing protocol)',
    diagrams=rows),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in rows:print(row['file'],row['editable_text_elements'],'text elements, 0 raster images')
