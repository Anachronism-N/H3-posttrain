"""Native SVG overview masters, designed directly in editable vector primitives.

Run with Python + Pillow. PyMuPDF is optional and writes inspection previews.
All text uses Times New Roman. No bitmap or font outline is embedded.
"""
from __future__ import annotations
from contextlib import contextmanager
from html import escape
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
from PIL import ImageFont

ROOT=Path(__file__).resolve().parent.parent
INK='#242A33'; BORDER='#B7BFC7'
SLATE='#65778A'; SAGE='#667A70'; PLUM='#85778C'; OCHRE='#A18C69'
FILL={'slate':'#FAFBFD','sage':'#FBFCFA','plum':'#FCFAFD','ochre':'#FDFBFA'}
FONT='Times New Roman'

def font(size,bold=False,italic=False):
    name='timesbi.ttf' if bold and italic else 'timesbd.ttf' if bold else 'timesi.ttf' if italic else 'times.ttf'
    return ImageFont.truetype('C:/Windows/Fonts/'+name,round(size))

class Fig:
    def __init__(self,title,w=2400,h=1300):
        self.w,self.h=w,h;self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',f'<title>{escape(title)}</title>',
            '<desc>Native vector master. All labels are editable Times New Roman text. Waveforms, poses, and score snippets are schematic illustrations, not measurements or experimental results.</desc>',
            f'<rect id="background" x="0" y="0" width="{w}" height="{h}" fill="#FFFFFF"/>']
        self.scene=[];self.texts=[]
    @contextmanager
    def group(self,name):
        self.parts.append(f'<g id="{escape(name)}">');yield;self.parts.append('</g>')
    def rect(self,x,y,w,h,fill='white',stroke=BORDER,sw=1.6,r=0):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        self.scene.append(dict(type='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,sw=sw,r=r))
    def line(self,x1,y1,x2,y2,col=BORDER,sw=1.5):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"/>')
        self.scene.append(dict(type='line',x1=x1,y1=y1,x2=x2,y2=y2,col=col,sw=sw))
    def path(self,d,col=INK,sw=1.5,fill='none'):
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"/>')
        self.scene.append(dict(type='path',d=d,col=col,sw=sw,fill=fill))
    def circle(self,x,y,r,fill=INK,stroke='none',sw=1):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        self.scene.append(dict(type='circle',x=x,y=y,r=r,fill=fill,stroke=stroke,sw=sw))
    def arrow(self,points,col=SLATE,sw=1.8,both=False):
        self.path('M'+' L'.join(f'{x},{y}' for x,y in points),col,sw)
        def head(tip,prev):
            dx,dy=tip[0]-prev[0],tip[1]-prev[1];n=math.hypot(dx,dy);ux,uy=dx/n,dy/n
            bx,by=tip[0]-10*ux,tip[1]-10*uy
            self.path(f'M{tip[0]},{tip[1]} L{bx-4*uy},{by+4*ux} L{bx+4*uy},{by-4*ux} Z',col,0,col)
        head(points[-1],points[-2])
        if both:head(points[0],points[1])
    def text(self,x,y,text,size=26,bold=False,col=INK,anchor='start',italic=False):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Times New Roman" font-size="{size}" font-weight="{700 if bold else 400}" font-style="{"italic" if italic else "normal"}" text-anchor="{anchor}" fill="{col}">{escape(text)}</text>')
        width=font(size,bold,italic).getlength(text)
        self.texts.append(dict(text=text,x=x,y=y,size=size,width=width,anchor=anchor))
        self.scene.append(dict(type='text',x=x,y=y,text=text,size=size,bold=bold,col=col,anchor=anchor,italic=italic))
    def lines(self,x,y,lines,size=26,leading=32,bold=False,col=INK,anchor='start',italic=False):
        for k,t in enumerate(lines):self.text(x,y+k*leading,t,size,bold,col,anchor,italic)
    def wrap(self,x,y,text,width,size=26,leading=32,bold=False,col=INK):
        lines=[]
        for para in text.split('\n'):
            line=''
            for word in para.split():
                p=(line+' '+word).strip()
                if line and font(size,bold).getlength(p)>width:lines.append(line);line=word
                else:line=p
            lines.append(line)
        self.lines(x,y,lines,size,leading,bold,col)
        return len(lines)*leading
    def frame(self,x,y,w,h,title,index=None,tint='slate',fs=32):
        self.rect(x,y,w,h,FILL[tint])
        self.line(x,y,x+w,y,{'slate':SLATE,'sage':SAGE,'plum':PLUM,'ochre':OCHRE}[tint],2.7)
        label=f'{index}  {title}' if index else title
        self.text(x+20,y+42,label,fs,True)
        self.line(x+20,y+62,x+w-20,y+62,BORDER,1)
    def boxlabel(self,x,y,w,h,lines,fs=25,col=INK,fill='white',bold=True):
        self.rect(x,y,w,h,fill)
        if isinstance(lines,str):lines=[lines]
        start=y+h/2-(len(lines)-1)*fs*.61+fs*.34
        self.lines(x+w/2,start,lines,fs,fs*1.22,bold,col,'middle')
    def wave(self,x,y,w,h,col=SLATE,n=100,phase=0):
        self.line(x,y,x+w,y,col,.9)
        for k in range(n):
            u=k/(n-1)
            envelope=.06+.88*(.5+.5*math.sin(u*6.4*math.pi+phase))**2*(.16+.84*math.sin(u*1.8*math.pi+.6)**2)
            amp=h*envelope*(.35+.65*abs(math.sin(2.37*k)))/2
            self.line(x+w*u,y-amp,x+w*u,y+amp,col,1.6)
    def skeleton(self,x,y,scale=1,pose=0,col=SLATE,joints=True,legs=False):
        poses=[[(0,-55),(0,-35),(-22,-22),(22,-22),(-35,4),(35,2),(-42,29),(45,26),(0,9),(-15,29),(15,29),(-24,58),(29,55),(-34,84),(38,82)],
               [(0,-55),(0,-35),(-22,-22),(22,-22),(-44,-30),(43,-39),(-58,-48),(56,-56),(0,9),(-15,29),(15,29),(-35,55),(39,24),(-50,81),(56,42)],
               [(0,-55),(0,-35),(-22,-22),(22,-22),(-25,-50),(25,-50),(-10,-73),(10,-73),(0,9),(-15,29),(15,29),(-30,59),(28,58),(-40,84),(39,84)]]
        pts=[(x+a*scale,y+b*scale) for a,b in poses[pose%3]]
        for a,b in [(0,1),(1,2),(1,3),(2,4),(4,6),(3,5),(5,7),(1,8),(8,9),(8,10),(9,11),(11,13),(10,12),(12,14)]:
            self.line(*pts[a],*pts[b],PLUM if legs and a>=8 else col,3.7*scale)
        self.circle(*pts[0],7*scale,col)
        if joints:
            for i,p in enumerate(pts[1:],1):self.circle(*p,2.8*scale,PLUM if legs and i>=9 else 'white',col,1.2)
    def dashed(self,x,y,w,h,col=BORDER,sw=1.5):
        for a,b in [((x,y),(x+w,y)),((x+w,y),(x+w,y+h)),((x+w,y+h),(x,y+h)),((x,y+h),(x,y))]:
            dx,dy=b[0]-a[0],b[1]-a[1];n=math.hypot(dx,dy);ux,uy=dx/n,dy/n
            for start in range(0,round(n),14):
                end=min(start+8,n);self.line(a[0]+start*ux,a[1]+start*uy,a[0]+end*ux,a[1]+end*uy,col,sw)
    def rule_label(self,x,y,label,col=SLATE,fs=26,w=0):
        self.text(x,y,label,fs,True,col)
        if w:self.line(x,y+12,x+w,y+12,col,1.4)
    def save(self,name):
        self.parts.append('</svg>');p=ROOT/name;p.write_text('\n'.join(self.parts)+'\n',encoding='utf-8')
        (ROOT/'source'/(p.stem+'.scene.json')).write_text(json.dumps(dict(width=self.w,height=self.h,font_family=FONT,objects=self.scene),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for t in self.texts:
            l=t['x']-t['width']*(.5 if t['anchor']=='middle' else 1 if t['anchor']=='end' else 0)
            assert l>=0 and l+t['width']<=self.w+1 and 0<t['y']<self.h,(name,t)
        r=ET.parse(p).getroot();ns='{http://www.w3.org/2000/svg}'
        assert not r.findall('.//'+ns+'image')
        assert all(e.attrib.get('font-family')==FONT for e in r.findall('.//'+ns+'text'))
        return p

def tools_strip(f):
    with f.group('source-music-and-shared-analysis'):
        f.text(40,61,'Music',40,True)
        f.wave(160,59,2200,72,n=260)
        f.line(40,114,2360,114,BORDER,1)
        for x,w,name,desc,col in [(40,480,'Beat This!','2–4-bar observation units',SLATE),(560,480,'JEV','Global dance-style condition',SAGE),(1080,930,'Acoustic analysis','Relative energy · onset activity · recurrence cues',OCHRE)]:
            f.rect(x,144,w,84,'white')
            f.text(x+17,175,name,34,True,col)
            f.text(x+17,214,desc,32)
            f.arrow([(x+w/2,115),(x+w/2,139)],col)

def method_master():
    f=Fig('Method overview: joint-aware local movement expert and stateful composition')
    tools_strip(f)
    with f.group('joint-aware-local-expert'):
        f.frame(40,268,480,970,'Trained local expert','A','plum',38)
        f.text(280,379,'Qwen3-Omni + LoRA',38,True,anchor='middle')
        f.boxlabel(65,418,185,80,['Audio','front end'],32)
        f.arrow([(254,458),(279,458)],SLATE)
        f.rect(286,404,209,192,'white')
        f.text(390,432,'Backbone',32,True,anchor='middle')
        for i in range(3):
            y=451+i*44
            f.rect(300,y,125,30,FILL['slate'])
            f.line(314,y+15,409,y+15,SLATE,1)
            f.rect(441,y,39,30,FILL['plum'],PLUM)
            f.text(460.5,y+23,'ΔW',24,False,PLUM,'middle')
            f.arrow([(437,y+15),(430,y+15)],PLUM,1.3)
        f.arrow([(390,600),(390,639)],SLATE)
        f.boxlabel(65,647,430,72,['Local movement proposals'],34)
        f.text(65,772,'Joint-aware clause weights',34,True,PLUM)
        f.text(65,825,'“Raise the left arm;',32)
        f.rect(60,842,322,42,FILL['plum'],'none')
        f.text(65,872,'bend the right knee.”',32,True,PLUM)
        f.line(65,886,376,886,PLUM,1.6)
        f.skeleton(443,850,.55,joints=True,legs=True)
        f.lines(65,934,['Emphasize the joint’s action','and its spatial relation.'],32,39)
        f.text(65,1033,'Lower-limb instance: w = 1.5',32,True,PLUM)
        f.text(65,1079,'Other caption tokens: w = 1',32)
        f.line(65,1124,495,1124,BORDER,1)
        f.text(280,1175,'Clause-weighted supervised learning',26,False,SLATE,'middle',True)
    with f.group('stateful-compositional-harness'):
        f.frame(560,268,950,970,'Stateful compositional harness','B','sage',38)
        f.rect(585,355,900,83,'white')
        f.text(606,386,'Choreographic brief',34,True)
        f.text(606,423,'Task goals · user conditions · editable scope',32)
        f.text(1470,386,'Task state',26,False,SAGE,'end',True)
        # Agentic decision loop: models and harness infrastructure are separate.
        f.rect(585,490,265,185,'white',PLUM)
        f.text(717.5,524,'Composer agent',34,True,PLUM,'middle')
        f.text(717.5,567,'Qwen3-VL',34,True,anchor='middle')
        f.lines(717.5,610,['Outline · motifs','Local proposals'],32,39,anchor='middle')
        f.rect(960,490,245,185,'white',SAGE)
        f.text(1082.5,524,'Controller',34,True,SAGE,'middle')
        f.lines(1082.5,568,['Stages','Permissions','Budget / stop'],32,37,anchor='middle')
        f.rect(1300,490,185,185,'white',OCHRE)
        f.text(1392.5,524,'Tools',34,True,OCHRE,'middle')
        f.lines(1392.5,565,['Read cache','Validate','Commit / edit'],32,38,anchor='middle')
        f.arrow([(955,548),(855,548)],SLATE)
        f.lines(905,500,['Context','feedback'],28,33,anchor='middle')
        f.arrow([(855,621),(955,621)],SLATE)
        f.text(905,663,'Proposal',28,anchor='middle')
        f.arrow([(1210,548),(1295,548)],SLATE)
        f.text(1252,528,'Call',28,anchor='middle')
        f.arrow([(1295,621),(1210,621)],SLATE)
        f.lines(1252,651,['Checks /','errors'],26,30,anchor='middle')
        f.arrow([(717,442),(717,485)],SLATE)
        # Short, straight evidence access; no serpentine read-only connector.
        f.rect(585,730,300,170,'white',SLATE)
        f.text(605,767,'Read-only cache',32,True,SLATE)
        f.lines(605,808,['Local proposals','Bars · style','Acoustics'],32,39)
        f.arrow([(717,725),(717,680)],SLATE)
        f.rect(915,730,570,170,'white',SAGE)
        f.text(935,766,'Persistent task state',34,True,SAGE)
        f.line(935,784,1465,784,BORDER)
        f.line(1200,785,1200,884,BORDER)
        f.lines(940,824,['Outline / motifs','Current draft'],32,44)
        f.lines(1217,824,['Committed history','Call / token ledger'],32,44)
        f.arrow([(1082,680),(1082,725)],SAGE,both=True)
        f.arrow([(1392,680),(1392,725)],SAGE,both=True)
        f.text(585,972,'Targeted revision',36,True)
        f.text(1485,972,'Preserve time · track · motif',28,False,SLATE,'end')
        f.line(585,991,1485,991,BORDER)
        for i,(label,motif) in enumerate([('Bars 1–2','A'),('Bars 3–4','B'),('Bars 5–6','A′'),('Bars 7–8','B′'),('Bars 9–10','A″')]):
            x=594+i*179
            f.rect(x,1027,164,120,'white',OCHRE if i==2 else BORDER,2.2 if i==2 else 1)
            if i==2:f.line(x,1027,x+164,1027,OCHRE,4)
            f.text(x+82,1058,label,30,True,anchor='middle')
            f.text(x+82,1097,'Revise text' if i==2 else 'Retain',28,False,OCHRE if i==2 else SLATE,'middle')
            f.text(x+82,1134,motif,32,True,PLUM,'middle')
        f.lines(1035,1190,['Revise one region; retain accepted context.'],30,anchor='middle')
    with f.group('hierarchical-choreography-score'):
        f.frame(1550,268,460,970,'Hierarchical score','C','slate',38)
        f.text(1574,382,'Whole-song outline',34,True)
        f.text(1574,421,'Dance style is …',32,False,SAGE)
        xs=[1620,1780,1940]
        for i,(x,motif,role) in enumerate(zip(xs,['A','B','A′'],['Introduce','Contrast','Resolve'])):
            f.circle(x,486,24,'white',PLUM,1.7)
            f.text(x,496,motif,32,True,PLUM,'middle')
            f.text(x,538,role,32,anchor='middle')
            if i<2:f.arrow([(x+30,486),(xs[i+1]-30,486)],SLATE)
        f.line(1574,570,1986,570,BORDER)
        f.text(1574,610,'Bar-unit movement detail',34,True)
        rows=[('Bars 1–2 · A','Step right, sweep the left arm across, then lunge.'),
              ('Bars 3–4 · B','Turn left, lift both arms, then lower with control.'),
              ('Bars 5–6 · A′','Repeat A with a wider arm sweep and a lower stance.')]
        for i,(bar,txt) in enumerate(rows):
            y=661+i*173
            f.text(1577,y,bar,32,True,PLUM)
            f.skeleton(1619,y+75,.48,pose=i,joints=False,col=SLATE)
            f.wrap(1680,y+42,txt,300,32,36)
            if i<2:f.line(1574,y+147,1986,y+147,BORDER,1)
        f.text(1780,1211,'Inspect · edit · compose',28,False,SLATE,'middle',True)
    f.arrow([(1489,583),(1545,583)],SLATE)
    with f.group('independent-global-local-evaluation'):
        f.frame(2050,268,310,970,'Evaluation','D','ochre',38)
        f.text(2070,381,'Audio + score',34,True)
        f.lines(2070,420,['Independent assessment'],26,col=SLATE)
        f.arrow([(2014,486),(2045,486)],SLATE)
        f.arrow([(2205,115),(2205,260)],SLATE)
        groups=[(480,'Q','Local text quality',[('Q1','Specificity'),('Q2','Body / space'),('Q3','Local timing')],SLATE),
                (697,'M','Local music compatibility',[('M1','Rhythm'),('M2','Dynamics'),('M3','Affective')],SAGE),
                (936,'G','Global composition',[('G1','Continuity'),('G2','Structure'),('G3','Motif variation')],PLUM)]
        for y,index,title,rows,col in groups:
            f.line(2070,y,2340,y,col,1.8)
            f.wrap(2070,y+35,title,270,32,37,True,col)
            start=y+(111 if index=='M' else 91)
            for k,(code,lab) in enumerate(rows):
                f.text(2070,start+k*39,code,32,True,col)
                f.text(2138,start+k*39,lab,32)
            if index=='M':f.text(2138,start+2*39+36,'compatibility',32)
        f.line(2070,1160,2340,1160,BORDER)
        f.text(2205,1203,'Quoted evidence',28,False,SLATE,'middle',True)
    with f.group('explicit-audio-and-evidence-dataflow'):
        # Music enters the expert directly; the composer only sees cached evidence.
        f.arrow([(40,114),(22,114),(22,458),(65,458)],SLATE)
        f.circle(40,114,3,SLATE)
        # A single monotone collector joins all shared analyses and local proposals.
        for x in [280,800,1545]:f.arrow([(x,228),(x,248)],SLATE)
        f.line(280,248,1545,248,SLATE,1.8)
        f.line(540,248,540,812,SLATE,1.8)
        f.circle(540,248,3,SLATE)
        f.arrow([(495,683),(540,683)],SLATE)
        f.circle(540,683,3,SLATE)
        f.arrow([(540,812),(585,812)],SLATE)
        f.text(1620,257,'Shared evidence bus',26,False,SLATE,italic=True)
    return f.save('overview_method.svg')

def benchmark_master():
    f=Fig('Data resource, hierarchical benchmark, and compact method baseline')
    with f.group('evidence-linked-annotation'):
        f.frame(40,45,930,1200,'Evidence-linked annotation','A','slate',38)
        # Media icons, intentionally schematic and editable.
        f.rect(65,151,557,208,'white')
        for i in range(3):
            x=77+i*182
            f.rect(x,167,168,174,FILL['slate'])
            f.skeleton(x+84,244,.88,pose=i,col=SLATE,joints=True)
            f.text(x+84,327,['Pose','Movement','Continuation'][i],26,anchor='middle')
        f.text(344,394,'Dance video observations',34,True,anchor='middle')
        f.wave(662,247,278,114,SAGE,n=64)
        f.text(800,328,'Aligned music',32,True,SAGE,'middle')
        f.arrow([(943,247),(1005,247)],SAGE)
        f.text(977,219,'retain',26,False,SAGE,'middle',True)
        f.rule_label(65,455,'Current label route',SLATE,34,880)
        for x,w,lab in [(65,250,'2D rule events'),(403,250,'GPT-6.1-sol'),(740,205,'Captions')]:f.boxlabel(x,497,w,65,lab,34)
        f.arrow([(320,529),(398,529)],SLATE)
        f.arrow([(658,529),(735,529)],SLATE)
        f.arrow([(949,529),(1005,529)],PLUM)
        f.text(528,600,'Evidence-grounded refinement',34,True,PLUM,'middle')
        f.rect(65,629,270,275,'white')
        f.skeleton(118,726,.64,joints=True)
        for i,(label,col) in enumerate([('Wrist rises',SLATE),('Elbow bends',OCHRE),('Knee bends',SAGE)]):
            y=665+i*65;f.text(177,y,label,28,True,col);f.line(177,y+24,311-i%2*30,y+24,col,3)
        f.text(200,866,'Timed source events',26,False,SLATE,'middle',True)
        f.text(378,654,'Refinement standards',34,True)
        checks=[('Source coverage','Retain supported events.'),('Body and direction','Keep references and relations.'),('Timing and repetition','Preserve order and recurrence.'),('Concise grounded prose','Add no unsupported actions.')]
        for i,(a,b) in enumerate(checks):
            y=697+i*64;f.circle(387,y-7,3.2,PLUM);f.text(404,y,a,32,True);f.text(404,y+32,b,26,False,SLATE)
        f.rect(65,945,880,80,FILL['plum'],'none')
        f.text(86,981,'“The left arm rises as the knees bend.”',34,True,PLUM)
        f.text(86,1015,'Sentence links preserve the source-event evidence.',28)
        # Production and geometry extension must remain visibly distinct.
        f.dashed(65,1054,880,164,BORDER)
        f.text(84,1088,'3D geometry extension',34,True,SLATE)
        for x,w,txt in [(84,185,'NLF pose'),(356,244,'Temporal events'),(674,251,'Consolidation')]:f.boxlabel(x,1122,w,46,txt,32)
        f.arrow([(274,1145),(351,1145)],SLATE);f.arrow([(605,1145),(669,1145)],SLATE)
        f.text(84,1201,'Support-aware evidence; consolidate overlaps and retain provenance.',26,False,SLATE)
    with f.group('paired-resource'):
        f.frame(1010,45,480,730,'Paired resource','B','sage',38)
        records=[['The left arm rises','as the knees bend.'],['Step right with','an arm sweep.'],['Turn left with','a wide arm circle.']]
        for i,lines in enumerate(records):
            y=157+i*137
            f.line(1035,y-16,1465,y-16,BORDER)
            f.wave(1037,y+41,126,68,[SLATE,SAGE,PLUM][i],n=32)
            f.text(1183,y+50,'↔',34)
            f.lines(1228,y+24,lines,32,40,True,[SLATE,SAGE,PLUM][i])
        f.line(1035,571,1465,571,BORDER)
        f.lines(1250,612,['Audio · movement text','Time · source provenance'],32,40,anchor='middle')
        f.line(1035,670,1465,670,BORDER)
        f.text(1140,722,'Training',34,True,SAGE,'middle')
        f.text(1360,722,'Evaluation',34,True,PLUM,'middle')
        f.arrow([(1494,722),(1525,722)],PLUM)
        f.arrow([(1140,779),(1140,836)],SAGE)
    with f.group('strict-hierarchical-benchmark'):
        f.frame(1530,45,830,730,'Evidence-based benchmark','C','ochre',38)
        f.text(1555,160,'Music audio',34,True)
        f.wave(1558,207,220,59,SAGE,n=42)
        f.text(1804,217,'+',38)
        f.boxlabel(1840,160,310,94,['Candidate score','Local / whole song'],32)
        f.boxlabel(2180,160,155,94,['Reference','agreement'],28)
        f.text(2257,285,'Diagnostic',26,False,SLATE,'middle',True)
        stages=[(1555,215,['Audio-only','observations']),(1843,215,['Movement','evidence']),(2130,205,['Independent','ratings'])]
        for x,w,lines in stages:f.boxlabel(x,320,w,86,lines,32)
        f.arrow([(1775,362),(1838,362)],SLATE);f.arrow([(2063,362),(2125,362)],SLATE)
        f.text(1555,444,'Local and global criteria',34,True)
        cols=[(1555,245,'Q','Local text quality',[('Q1','Specificity'),('Q2','Body / space'),('Q3','Local timing')],SLATE),
              (1830,245,'M','Local music compatibility',[('M1','Rhythm'),('M2','Dynamics'),('M3','Affective')],SAGE),
              (2105,230,'G','Global composition',[('G1','Continuity'),('G2','Structure'),('G3','Motif')],PLUM)]
        for x,w,index,title,rows,col in cols:
            f.line(x,470,x+w,470,col,2.3)
            f.text(x,513,index,40,True,col)
            f.wrap(x+48,500,title,w-48,30,34,True,col)
            for i,(code,label) in enumerate(rows):
                f.text(x,575+i*39,code,32,True,col);f.text(x+61,575+i*39,label,32)
            if index=='M':f.text(x+61,689,'compatibility',32)
            if index=='G':f.text(x+61,689,'variation',32)
        f.line(1555,701,2335,701,BORDER)
        f.text(1555,744,'Strict rubric',32,True)
        for x,n,desc in [(1750,'1','Unsupported'),(1990,'3','Partial'),(2160,'5','Supported')]:
            f.text(x,744,n,34,True,OCHRE);f.text(x+35,744,desc,32)
    with f.group('compact-baseline'):
        f.frame(1010,842,1350,403,'Our baseline: learned local expert + compositional harness','D','plum',36)
        f.boxlabel(1035,970,290,154,['Our Qwen3-Omni','+ LoRA','Weighted clauses'],32)
        f.arrow([(1330,1048),(1365,1048)],SLATE)
        f.rect(1370,937,615,242,'white')
        f.text(1390,974,'Stateful harness',36,True)
        f.text(1965,974,'JEV · bar units',28,False,SAGE,'end')
        for x,w,txt in [(1390,150,'Planner'),(1597,150,'Controller'),(1805,160,'Tools')]:f.boxlabel(x,1014,w,57,txt,32)
        f.arrow([(1545,1043),(1592,1043)],SLATE,both=True);f.arrow([(1752,1043),(1800,1043)],SLATE,both=True)
        f.line(1390,1103,1965,1103,BORDER)
        f.text(1677,1144,'Motifs · committed history · scoped revision',32,anchor='middle')
        f.arrow([(1990,1048),(2025,1048)],SLATE)
        f.rect(2030,970,305,154,'white')
        f.lines(2182,1010,['Hierarchical score','Outline + local detail'],32,39,True,anchor='middle')
        f.text(2182,1094,'Dance style is …',32,False,SAGE,'middle',True)
        # Return the baseline output to the actual Candidate input, not a panel edge.
        # The right gutter and inter-card gap keep the connector clear of all text.
        f.arrow([(2335,1048),(2380,1048),(2380,300),(2164,300),(2164,207),(2150,207)],SLATE)
        f.text(2348,820,'Candidate input',26,False,SLATE,'end',True)
        f.text(1685,1218,'Shared evidence · controlled comparisons · explicit inference budgets',28,False,SLATE,'middle',True)
    return f.save('overview_benchmark.svg')

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    paths=[method_master(),benchmark_master()]
    import fitz
    audit=[]
    for p in paths:
        svg=fitz.open(p);doc=fitz.open('pdf',svg.convert_to_pdf())
        doc[0].get_pixmap(matrix=fitz.Matrix(.8,.8),alpha=False).save(p.with_suffix('.preview.png'))
        root=ET.parse(p).getroot();types={}
        for e in root.iter():
            tag=e.tag.rsplit('}',1)[-1];types[tag]=types.get(tag,0)+1
        audit.append(dict(file=p.name,font='Times New Roman',elements=types,embedded_images=types.get('image',0),native_text=True))
        print(p)
    (ROOT/'source/overview_master_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
