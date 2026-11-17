"""Editable protocol schematic: unit-level evidence and whole-song structure."""
from pathlib import Path
from html import escape
from PIL import ImageFont
import hashlib,json,math,xml.etree.ElementTree as ET

OUT=Path(__file__).resolve().parents[1]
W,H=1800,1040
INK,TEAL,BLUE,PURPLE,ORANGE,CORAL='#202C3C','#008F95','#2474B5','#7852A9','#E69A28','#D45D64'
LINE,MUTED='#C4CFD9','#52677D'
PALE_BLUE,PALE_PURPLE,PALE_WARM='#FAFCFD','#FBFAFE','#FFFCF7'
parts,labels,fonts=[],[],{}
def raw(s):parts.append(s)
def rect(x,y,w,h,fill='white',stroke=LINE,sw=1.6,r=5):
 raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def circle(x,y,r,fill='white',stroke='none',sw=1.6):
 raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def path(d,color=INK,sw=2.2,fill='none',dash=None):
 raw(f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
def line(points,color=LINE,sw=2,arrow=False,dash=None):
 path('M'+' L'.join(f'{x},{y}' for x,y in points),color,sw,dash=dash)
 if arrow:
  a,b=points[-2:];ang=math.atan2(b[1]-a[1],b[0]-a[0]);le,ha=10,4.7
  p=[(b[0]-le*math.cos(ang)+ha*math.sin(ang),b[1]-le*math.sin(ang)-ha*math.cos(ang)),(b[0]-le*math.cos(ang)-ha*math.sin(ang),b[1]-le*math.sin(ang)+ha*math.cos(ang))]
  path(f'M{b[0]},{b[1]} L{p[0][0]},{p[0][1]} L{p[1][0]},{p[1][1]} Z',color,0,color)
def measure(s,size,bold=False,italic=False):
 name='timesbi.ttf' if bold and italic else 'timesbd.ttf' if bold else 'timesi.ttf' if italic else 'times.ttf'
 key=name,size
 if key not in fonts:fonts[key]=ImageFont.truetype(str(Path('C:/Windows/Fonts')/name),size)
 return fonts[key].getlength(s)
def text(x,y,s,size=28,bold=False,color=INK,anchor='start',italic=False):
 for i,row in enumerate(str(s).split('\n')):
  yy=y+i*size*1.18
  raw(f'<text x="{x}" y="{yy}" font-family="Times New Roman" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}"'+(' font-style="italic"' if italic else '')+'>'+escape(row)+'</text>')
  wi=measure(row,size,bold,italic)
  left=x-wi if anchor=='end' else x-wi/2 if anchor=='middle' else x
  labels.append(dict(text=row,size=size,bbox=[left,yy-size,left+wi,yy+size*.24]))
def rich(x,y,chunks,size=29):
 for s,c,b in chunks:text(x,y,s,size,b,c);x+=measure(s,size,b)
def badge(x,y,s,color,r=22):
 circle(x,y,r,color);text(x,y+9,s,27,True,'white','middle')
def wave(x,y,wi,he,color,n=100):
 for i in range(n):
  t=i/(n-1);a=he*(.14+.86*math.sin(i*1.19)**2)*(.25+.75*math.sin(8*t+.3)**2)
  line([(x+t*wi,y-a),(x+t*wi,y+a)],color,2)
def torso(x,y,color=BLUE,sc=1,opened=True):
 circle(x,y-34*sc,8*sc,color)
 line([(x,y-23*sc),(x,y+20*sc)],color,3.2)
 for sign in [-1,1]:
  line([(x,y-10*sc),(x+sign*26*sc,y-18*sc),(x+sign*43*sc,y-35*sc if opened else y+5*sc)],color,3.2)
  line([(x,y+20*sc),(x+sign*24*sc,y+55*sc)],color,3.2)
 for px,py in [(x-26*sc,y-18*sc),(x+26*sc,y-18*sc),(x,y+20*sc)]:circle(px,py,3.5*sc,'white',color,1.8)
def note(x,y,color=TEAL,sc=1):
 path(f'M{x},{y} L{x},{y-36*sc} L{x+25*sc},{y-43*sc} L{x+25*sc},{y-8*sc}',color,3)
 raw(f'<ellipse cx="{x-5*sc}" cy="{y+2*sc}" rx="{9*sc}" ry="{6*sc}" fill="{color}"/>')
 raw(f'<ellipse cx="{x+20*sc}" cy="{y-6*sc}" rx="{9*sc}" ry="{6*sc}" fill="{color}"/>')
def sections(x,y,widths,labs,colors,h=44):
 for wi,lab,c in zip(widths,labs,colors):
  rect(x,y,wi,h,'white',c,2,3);rect(x,y,6,h,c,'none',0,0)
  text(x+wi/2+3,y+31,lab,29,True,c,'middle');x+=wi+8

raw(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
raw('<title>Evidence-grounded evaluation across local units and the whole song</title><desc>Editable protocol schematic. Q1-Q3 local text quality; M1-M3 local music compatibility; G1-G3 global composition. M3 is affective compatibility without an event gate. All waveforms, poses and quotation are illustrative, with no assigned candidate scores.</desc>')
rect(0,0,W,H,'white','none',0,0)

raw('<g id="aligned-inputs">')
text(30,36,'Music audio',31,True);wave(30,82,603,29,TEAL,103)
rect(30,44,118,75,'none',ORANGE,2.5,2);line([(40,122),(138,122)],ORANGE,3)
text(750,36,'Hierarchical choreography',31,True)
sections(750,54,[155]*5,['A','B','A′','C','…'],[BLUE,PURPLE,BLUE,ORANGE,MUTED],48)
rect(745,49,165,58,'none',ORANGE,2.5,3)
text(1750,36,'One song clock',26,False,MUTED,'end',True)
line([(89,124),(89,149)],ORANGE,2.3,True)
line([(827,112),(827,149)],ORANGE,2.3,True)
line([(1485,112),(1485,149)],PURPLE,2.3,True)
raw('</g>')

raw('<g id="local-unit-inspection">')
rect(24,158,1152,440,'white',LINE,1.8,6)
circle(61,188,15,'white',ORANGE,3);line([(72,200),(84,212)],ORANGE,3.5)
text(100,198,'Zoom into one aligned unit',32,True)
text(1153,198,'Local assessment',26,False,MUTED,'end',True)
line([(45,219),(1152,219)],LINE,1.4)
text(45,255,'Local text quality',31,True,BLUE)
rect(45,272,535,92,PALE_BLUE,'none',0,3);rect(45,272,5,92,BLUE,'none',0,0)
rich(62,307,[('“Both arms ',BLUE,True),('sweep outward;',TEAL,True)],29)
rich(62,342,[('then ',ORANGE,True),('hold ',TEAL,True),('and return to the waist.”',INK,False)],29)
for x in [45,226,407]:rect(x,380,173,212,PALE_BLUE,'none',0,4)

# Q1: action detail; Q2: bodily frame; Q3: serial and concurrent intervals.
rect(78,393,107,66,'white',BLUE,2,3)
line([(91,407),(171,407)],BLUE,2.3);line([(91,422),(157,422)],TEAL,3.2);line([(91,438),(165,438)],ORANGE,2.3)
badge(131,482,'Q1',BLUE);text(131,525,'Specificity',26,True,BLUE,'middle')
text(131,555,'Action +',26,False,INK,'middle');text(131,584,'attributes',26,False,INK,'middle')
torso(313,427,BLUE,.66,False)
line([(273,428),(254,421)],ORANGE,2.2,True);line([(353,428),(372,421)],ORANGE,2.2,True)
badge(312,482,'Q2',BLUE);text(312,525,'Body / space',26,True,BLUE,'middle')
text(312,555,'Reference +',26,False,INK,'middle');text(312,584,'coordination',26,False,INK,'middle')
line([(427,411),(558,411)],BLUE,2.2,True)
for x,wi,c in [(427,52,BLUE),(484,27,ORANGE),(516,42,TEAL)]:rect(x,422,wi,13,c,'none',0,1)
rect(427,445,75,12,PURPLE,'none',0,1)
badge(493,482,'Q3',BLUE);text(493,525,'Local timing',26,True,BLUE,'middle')
text(493,555,'Order +',26,False,INK,'middle');text(493,584,'concurrency',26,False,INK,'middle')

text(625,255,'Local music compatibility',31,True,TEAL)
badge(645,302,'M1',TEAL);wave(686,303,131,17,TEAL,30)
for x in [691,733,775,817]:line([(x,327),(x,340)],TEAL,2)
line([(692,347),(734,347)],ORANGE,3.4);line([(775,347),(815,347)],ORANGE,3.4)
text(844,300,'Rhythm',28,True,TEAL);text(844,334,'Pulse + repetition',26);text(844,365,'Sustained motion',26)
line([(625,380),(1149,380)],LINE,1.2)
badge(645,417,'M2',TEAL)
path('M689,450 L713,398 L729,438 L749,405 L764,450',ORANGE,3.5)
path('M779,446 C793,397 807,399 823,446',TEAL,3.5)
text(844,414,'Movement dynamics',28,True,TEAL);text(844,448,'Force + range',26);text(844,479,'Articulation',26)
line([(625,493),(1149,493)],LINE,1.2)
badge(645,533,'M3',TEAL);note(699,550,TEAL,.72)
line([(728,531),(752,531)],PURPLE,2,True);torso(790,540,PURPLE,.46,True)
text(844,531,'Affective compatibility',27,True,TEAL);text(844,562,'Character + expression',26)
text(625,590,'Steady passages remain assessable.',26,False,TEAL,italic=True)
raw('</g>')

raw('<g id="whole-song-assessment">')
rect(1200,158,576,440,PALE_PURPLE,LINE,1.8,6)
text(1223,198,'Global composition',32,True,PURPLE)
text(1753,198,'Whole song',26,False,MUTED,'end',True)
line([(1222,219),(1753,219)],LINE,1.4)
sections(1223,238,[117]*4,['A','B','A′','C'],[PURPLE,BLUE,PURPLE,ORANGE],44)
for x,t in [(1281,'introduce'),(1406,'develop'),(1531,'return'),(1656,'resolve')]:text(x,315,t,26,False,MUTED,'middle')
path('M1282,233 C1282,216 1532,216 1532,233',PURPLE,2.3);line([(1525,227),(1532,233)],PURPLE,2.3,True)
badge(1247,360,'G1',PURPLE);circle(1310,359,9,'white',PURPLE,2)
line([(1320,359),(1357,359)],PURPLE,2.4);circle(1368,359,9,'white',PURPLE,2)
text(1400,369,'Boundary continuity',28,True,PURPLE)
badge(1247,428,'G2',PURPLE)
for x,he,c in [(1301,20,BLUE),(1322,35,PURPLE),(1343,25,TEAL),(1364,43,ORANGE)]:rect(x,447-he,13,he,c,'none',0,1)
text(1400,437,'Musical structure',28,True,PURPLE)
badge(1247,499,'G3',PURPLE)
for x,lab in [(1310,'A'),(1361,'A′')]:circle(x,498,21,'white',PURPLE,2);text(x,507,lab,27,True,PURPLE,'middle')
line([(1332,498),(1337,498)],PURPLE,2,True);text(1400,508,'Motifs + variation',28,True,PURPLE)
text(1223,569,'Check across units and musical sections.',26,False,MUTED,italic=True)
raw('</g>')

raw('<g id="strict-score-anchors">')
text(30,643,'Strict 1–5 anchors',32,True)
text(1770,642,'Independent criteria, not a weighted total',26,False,MUTED,'end',True)
anchors=[(1,'Core failure','Core unsupported','or contradicted',CORAL),(2,'Critical gap','Useful content,','critical defect',ORANGE),(3,'Usable','Core identifiable,','key detail missing',BLUE),(4,'Minor gaps','Core supported,','small omission',PURPLE),(5,'Supported','Core choices linked','to clear evidence',TEAL)]
for i,(score,title,first,second,color) in enumerate(anchors):
 x=30+i*354;top=697-i*7
 rect(x,top,332,820-top,'white',LINE,1.4,5);rect(x,top,332,6,color,'none',0,0)
 circle(x+32,731,21,color);text(x+32,740,score,28,True,'white','middle');text(x+65,740,title,29,True,color)
 text(x+18,774,first,27);text(x+18,804,second,27)
 if i<4:line([(x+335,top+3),(x+349,top-4)],MUTED,1.6,True)
raw('</g>')

raw('<g id="evidence-procedure">')
text(30,857,'Evidence first, then a justified rating',31,True)
steps=[(30,BLUE,'1','Observe audio first','Audible cues + uncertainty'),(625,TEAL,'2','Quote the movement','Exact words + event IDs'),(1220,PURPLE,'3','Justify each criterion','Support + audible time span')]
for x,c,n,title,desc in steps:
 circle(x+30,906,24,'white',c,2.5);text(x+30,916,n,30,True,c,'middle')
 text(x+71,903,title,30,True,c);text(x+71,940,desc,26)
line([(554,913),(604,913)],MUTED,2,True);line([(1149,913),(1199,913)],MUTED,2,True)
line([(30,970),(1770,970)],LINE,1.4)
text(30,1007,'Unknown: insufficient audio evidence.',26,False,MUTED)
text(1770,1007,'Vague text receives a content score.',26,False,MUTED,'end')
text(900,1031,'Allow half-time, holds, grounded contrast and choreography reused across songs.',26,False,TEAL,'middle')
raw('</g></svg>')

OUT.mkdir(parents=True,exist_ok=True)
svg=OUT/'evaluation_framework.svg';svg.write_text('\n'.join(parts)+'\n',encoding='utf-8')
tree=ET.fromstring(svg.read_text(encoding='utf-8'));counts={}
for el in tree.iter():
 k=el.tag.split('}')[-1];counts[k]=counts.get(k,0)+1
outside=[a for a in labels if a['bbox'][0]<0 or a['bbox'][1]<0 or a['bbox'][2]>W or a['bbox'][3]>H+3]
audit={'schema':'herewechoreography.evaluation_layout.v3','source':'source/build_evaluation.py','output':'evaluation_framework.svg',
 'svg_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'viewbox':[0,0,W,H],'font':'Times New Roman',
 'minimum_text_size':min(a['size'] for a in labels),'palette':dict(ink=INK,teal=TEAL,blue=BLUE,purple=PURPLE,orange=ORANGE,coral=CORAL),
 'element_counts':counts,'whole_figure_raster':False,'embedded_images':counts.get('image',0),'text_bounds_violations':outside,
 'scope':'Illustrative protocol schematic; no candidate, heard recording or model is assigned a score.',
 'definitions':{'Q':'Local text quality','M':'Local music compatibility','G':'Global composition','M3':'Affective compatibility between musical character and bodily expression'},
 'M3_has_local_event_gate':False,'unknown_is_not_poor_text':True,'multi_song_compatible_choreography_allowed':True,
 'prior_D6_scores_reused':False,'actual_synchronization_measured':False,'new_training_or_benchmark_scores':False,'visual_review':'pending'}
(OUT/'source/evaluation_framework.v3.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if outside:raise RuntimeError('Out-of-canvas text: '+str(outside))
print(f'Wrote {svg}; {counts.get("text",0)} native labels, zero raster images.')

