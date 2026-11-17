"""One editable scientific master for grouped local/global evaluation."""
from pathlib import Path
from html import escape
import math

OUT=Path(__file__).resolve().parents[1]
INK,LINE='#242A33','#B7BFC7'
SLATE,SAGE,PLUM,OCHRE='#65778A','#667A70','#85778C','#A18C69'
PALE=['#FAFBFD','#FBFCFA','#FCFAFD','#FDFBFA']
parts=[]
def raw(s):parts.append(s)
def box(x,y,w,h,fill='white',stroke=LINE,r=4):
 raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
def text(x,y,s,size=23,bold=False,color=INK,anchor='start',italic=False):
 for i,t in enumerate(str(s).split('\n')):
  raw(f'<text x="{x}" y="{y+i*size*1.2}" font-family="Times New Roman" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}"'+(' font-style="italic"' if italic else '')+'>'+escape(t)+'</text>')
def line(points,color=LINE,sw=1.6,arrow=False):
 raw('<path d="M'+' L'.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{sw}"/>')
 if arrow:
  a,b=points[-2:];ang=math.atan2(b[1]-a[1],b[0]-a[0]);l=8;r=3
  q=[b,(b[0]-l*math.cos(ang)+r*math.sin(ang),b[1]-l*math.sin(ang)-r*math.cos(ang)),(b[0]-l*math.cos(ang)-r*math.sin(ang),b[1]-l*math.sin(ang)+r*math.cos(ang))]
  raw('<path d="M'+' L'.join(f'{x},{y}' for x,y in q)+f' Z" fill="{color}"/>')
def circ(x,y,r,c):raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>')
def wave(x,y,w,h,c):
 for i in range(90):
  t=i/89;a=(.15+.65*math.sin(i*.91)**2)*(.4+.6*math.sin(t*9)**2)*h
  line([(x+t*w,y-a),(x+t*w,y+a)],c,1.4)
def row(x,y,code,title,desc,c):
 circ(x+22,y-7,20,'white');text(x+22,y+1,code,22,True,c,'middle')
 text(x+59,y-5,title,26,True);text(x+59,y+22,desc,24,color=c)

raw('<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="850" viewBox="0 0 1800 850">')
raw('<title>Local and global evidence-grounded choreography evaluation</title><desc>Editable master; Q1-Q3 text quality, M1-M3 music compatibility, G1-G3 global composition. Rubric is a proposed protocol, not a performance result.</desc>')
raw('<rect width="1800" height="850" fill="white"/>')
text(30,37,'Music + hierarchical choreography',27,True)
wave(500,26,510,17,SLATE)
for i,(s,w) in enumerate([('A',160),('B',130),('A′',200)]):
 x=[1060,1240,1390][i];box(x,9,w,35,PALE[i],LINE,2);text(x+w/2,34,s,23,True,[SAGE,SLATE,PLUM][i],'middle')
text(1735,34,'song clock',19,color=SLATE,anchor='end',italic=True)
for x in [303,902,1498]:line([(x,54),(x,83)],SLATE,1.5,True)

panels=[(30,SLATE,PALE[0],'Local text quality','Assess a short movement passage'),(625,SAGE,PALE[1],'Local music compatibility','Listen to the aligned music window'),(1220,PLUM,PALE[2],'Global composition','Read and listen across the whole song')]
for x,c,fill,title,sub in panels:
 raw(f'<g id="{title.lower().replace(" ","-")}">')
 box(x,91,550,400,fill,LINE,6)
 line([(x+18,153),(x+532,153)],c,2)
 text(x+23,126,title,29,True);text(x+23,179,sub,20,color=c)
 raw('</g>')

# Explicit limb references linked to readable text.
for a,b in [((88,211),(88,252)),((88,226),(60,240)),((88,226),(116,213)),((88,252),(69,277)),((88,252),(111,280))]:line([a,b],SLATE,3)
circ(88,202,8,SLATE)
text(149,224,'“Both arms sweep outward,',23,True)
text(149,253,'then return toward the waist.”',23,True)
line([(149,265),(508,265)],LINE,1)
row(48,312,'Q1','Movement specificity','Explicit actions and key attributes',SLATE)
row(48,371,'Q2','Body and space','References, direction and coordination',SLATE)
row(48,430,'Q3','Local temporal coherence','Order, concurrency and duration',SLATE)

wave(651,216,485,22,SAGE)
for i in range(9):
 x=655+i*59;line([(x,247),(x,266)],LINE,1)
for i,word in enumerate(['step','step','hold','return']):
 x=660+i*120;text(x,282,word,19,color=SAGE,italic=True)
row(643,322,'M1','Rhythmic compatibility','Pulse, repetition and sustained motion',SAGE)
row(643,381,'M2','Movement dynamics','Force, range and articulation',SAGE)
row(643,440,'M3','Affective compatibility','Musical character and bodily expression',SAGE)

for i,(lab,c) in enumerate([('A',PLUM),('B',SLATE),('A′',PLUM),('C',OCHRE)]):
 x=1245+i*126;box(x,205,111,43,'white',LINE,2);text(x+55,235,lab,26,True,c,'middle')
 if i<3:line([(x+112,226),(x+123,226)],LINE,1.4,True)
text(1246,277,'Opening → development → return / resolution',20,color=PLUM)
row(1238,322,'G1','Cross-unit continuity','Explicit states connect at boundaries',PLUM)
row(1238,381,'G2','Musical structure','Section roles and overall progression',PLUM)
row(1238,440,'G3','Motifs and variation','Purposeful recurrence and variation',PLUM)

text(30,536,'Evidence-grounded scoring',29,True)
text(1770,534,'Each dimension is rated independently',21,color=SLATE,anchor='end',italic=True)
for i,(n,title,desc) in enumerate([(1,'Core failure','Unsupported core / conflict'),(2,'Critical gap','Useful, with a critical defect'),(3,'Usable','Core present, key omission'),(4,'Minor gaps','Core holds; minor omissions'),(5,'Supported','Evidence supports core choices')]):
 x=30+i*354;box(x,557,332,102,PALE[i%4],LINE,4)
 circ(x+31,591,19,[SLATE,SLATE,SAGE,SAGE,PLUM][i]);text(x+31,599,str(n),24,True,'white','middle')
 text(x+61,596,title,24,True)
 text(x+16,634,desc,23,color=SLATE)

steps=[(30,425,'1  Observe music','Audible cues + uncertainty'),(525,590,'2  Quote the movement','Action quote + event IDs + reason'),(1185,585,'3  Record a justified rating','Times for music-linked claims')]
for x,w,title,sub in steps:
 line([(x,707),(x+w,707)],LINE,1.2)
 text(x+12,743,title,26,True);text(x+12,777,sub,24,color=SLATE)
line([(467,746),(504,746)],SLATE,1.8,True);line([(1129,746),(1165,746)],SLATE,1.8,True)
text(30,827,'Allow reusable choreography and grounded contrast; labels alone do not establish quality.',20,color=SLATE,italic=True)
text(1770,827,'Unknown ≠ low quality',20,color=SLATE,anchor='end',italic=True)
raw('</svg>')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'evaluation_framework.svg').write_text('\n'.join(parts)+'\n',encoding='utf-8')
print('Wrote evaluation_framework.svg')
