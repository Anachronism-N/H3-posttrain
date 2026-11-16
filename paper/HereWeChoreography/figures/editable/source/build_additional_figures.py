"""Native editable SVG construction diagram and unchanged vector source copies."""
from pathlib import Path
from xml.sax.saxutils import escape
import math
import shutil

OUT = Path(__file__).resolve().parents[1]
FIG = OUT.parent
NAVY, TEAL, PURPLE, GOLD = '#20384b', '#167e83', '#745493', '#b08130'
elements=[]
def text(x,y,s,size=21,fill=NAVY,weight='normal',anchor='start'):
    elements.append(f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')
def rect(x,y,w,h,fill='white',stroke=NAVY,rx=10,dash=False):
    elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"'+(' stroke-dasharray="8 6"' if dash else '')+'/>')
def line(points,color=NAVY,width=2,arrow=False,dash=False):
    coords=' '.join(f'{x},{y}' for x,y in points)
    elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="7 5"' if dash else '')+'/>')
def circle(x,y,r,fill=TEAL):
    elements.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')
def body(cx,cy,scale=1,pose=0):
    pts=[(0,-49),(0,-20),(-19,-19),(19,-19),(-29,7),(29,-5),(-42,13),(43,-27),(-12,18),(12,18),(-14,51),(18,49),(-20,80),(20,80)]
    if pose: pts[4]=(-29,-37);pts[6]=(-40,-55);pts[5]=(32,-30);pts[7]=(43,-51)
    q=[(cx+x*scale,cy+y*scale) for x,y in pts]
    for a,b in [(1,2),(1,3),(2,4),(4,6),(3,5),(5,7),(1,8),(1,9),(8,9),(8,10),(9,11),(10,12),(11,13)]:line([q[a],q[b]],TEAL,3*scale)
    circle(*q[0],8*scale,TEAL)
    for x,y in q[1:]:circle(x,y,3*scale,NAVY)
def wave(x,y,w=235,h=25):
    p=[]
    for i in range(160):
        t=i/159;env=.15+.48*math.sin(9*t)**2+.3*math.exp(-((t-.65)/.12)**2)
        p.append((x+t*w,y+h*env*math.sin(i*2.1)))
    line(p,TEAL,1.7)
def card(x,w,n,title,color):
    rect(x,64,w,286,'#ffffff',color)
    rect(x,64,w,46,'#edf5f6',color)
    text(x+16,95,f'{n}  {title}',23,color,'bold')

OUT.mkdir(parents=True,exist_ok=True)
elements.append('<rect width="1800" height="600" fill="white"/>')
text(24,35,'Current reference-label route',26,TEAL,'bold')
cards=[(24,270,1,'Paired media',TEAL),(326,338,2,'Rule-event evidence',GOLD),(696,260,3,'Text refinement',PURPLE),(988,328,4,'Attribution + checks',GOLD),(1348,428,5,'Music–text sample',TEAL)]
for x,w,n,title,c in cards:card(x,w,n,title,c)
for left,right in [(294,326),(664,696),(956,988),(1316,1348)]:line([(left,183),(right-5,183)],arrow=True)
rect(45,135,228,68,'#f8fafc','#94a5b1')
text(159,176,'Dance video',23,anchor='middle')
rect(45,227,228,86,'#eff8f7','#94a5b1')
text(159,252,'Aligned music',21,anchor='middle');wave(61,281,194,18)
body(372,208,.69);body(454,208,.69,1)
rect(510,133,135,61,'#f8fafc','#b5c0c8');text(577,160,'Wrist',18,anchor='middle');text(577,182,'directions',18,anchor='middle')
rect(510,212,135,80,'#f8fafc','#b5c0c8');text(577,238,'Elbow / knee',17,anchor='middle');text(577,261,'angle change',17,anchor='middle')
text(495,329,'Windowed event IDs',20,anchor='middle')
rect(718,141,216,65,'#f1ecf7',PURPLE);text(826,182,'GPT-6.1-sol',24,PURPLE,'bold','middle')
text(826,250,'Rewrite supported',21,anchor='middle');text(826,282,'movements',21,anchor='middle');text(826,322,'Text-only input',17,PURPLE,anchor='middle')
rect(1007,142,290,63,'#fffaf1','#c5b085');text(1152,181,'Sentence ↔ event links',21,anchor='middle')
rect(1007,229,290,63,'#fffaf1','#c5b085');text(1152,267,'Coverage / repetition flags',19,anchor='middle')
rect(1368,134,388,73,'#f8fafc','#9badb9');text(1562,165,'Choreography caption',24,anchor='middle');text(1562,192,'with source-event attribution',17,anchor='middle')
rect(1368,225,388,49,'#e7f4f1',TEAL);text(1562,256,'Paired music',23,TEAL,'bold','middle')
text(1562,319,'Source + quality metadata',21,anchor='middle')
line([(159,313),(159,376),(1558,376),(1558,351)],TEAL,2.4,True)
text(805,399,'Retain aligned audio independently of text refinement',18,TEAL,anchor='middle')
rect(20,438,1760,144,'#fbf9fd',PURPLE,12,True)
text(38,469,'Implemented geometry extension',24,PURPLE,'bold')
for x,w,title,sub in [(70,294,'NLF','3D body geometry'),(423,333,'Temporal event extraction','Geometry + support masks'),(815,360,'Evidence consolidation','Traceable source occurrences')]:
    rect(x,488,w,73,'white',PURPLE,8,True);text(x+w/2,517,title,21,PURPLE,'bold','middle');text(x+w/2,545,sub,18,anchor='middle')
line([(364,526),(417,526)],PURPLE,2,True,True);line([(756,526),(809,526)],PURPLE,2,True,True)
text(1452,517,'Separate from current-label provenance',20,PURPLE,'bold','middle')
text(1452,545,'Documented integration interface',18,PURPLE,anchor='middle')
line([(115,135),(12,135),(12,524),(63,524)],PURPLE,1.8,True,True)
line([(970,488),(970,421),(826,421),(826,354)],PURPLE,1.8,True,True)
svg='<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="600" viewBox="0 0 1800 600"><title>Dataset construction with explicit provenance</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="#20384b"/></marker></defs>'+''.join(elements)+'</svg>'
(OUT/'dataset_construction.editable.svg').write_text(svg,encoding='utf-8')
for src,dst in [('annotation_distributions_v3.svg','annotation_distributions.editable.svg'),('choreography_example_v4.svg','choreography_example.editable.svg')]:
    shutil.copy2(FIG/src,OUT/dst)
print('Wrote dataset_construction, annotation_distributions and choreography_example editable SVGs')
