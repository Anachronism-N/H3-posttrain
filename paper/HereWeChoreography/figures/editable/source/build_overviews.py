"""Rebuild the two overview diagrams as fully editable, native SVG objects.

No embedded bitmap, raster tracing, or outlined text. Each semantic component is
an SVG group; captions remain real <text>. Waveforms and pose icons are schematic.
Requires Pillow only for deterministic text measurements; PyMuPDF renders previews.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from html import escape
import json
import math
import re
from PIL import ImageFont

HERE = Path(__file__).resolve().parent.parent
INK = '#17324d'
TEAL = '#137f87'
BLUE = '#376d9d'
PLUM = '#795090'
AMBER = '#bb791c'
LINE = '#91a9bd'
BG = '#ffffff'
PALE = {'teal':'#e4f5f3', 'blue':'#edf4fb', 'plum':'#f4ecf8', 'amber':'#fff3dd', 'gray':'#f3f6f8'}
FONT = 'Arial, Helvetica, sans-serif'

def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf', round(size))

class SVG:
    def __init__(self, width, height, title):
        self.w,self.h=width,height
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    f'<title>{escape(title)}</title>',
                    '<desc>Editable research figure. Labels are native text, figures are grouped vector primitives. Media and example instructions are schematic.</desc>',
                    '<defs>']
        for key,col in [('ink',INK),('teal',TEAL),('plum',PLUM),('blue',BLUE),('amber',AMBER)]:
            self.parts.append(f'<marker id="arrow-{key}" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4 L0,8 Z" fill="{col}"/></marker>')
        self.parts+=['</defs>',f'<rect id="canvas-background" x="0" y="0" width="{width}" height="{height}" fill="white"/>']
        self.count=0;self.scene=[];self.texts=[]
    @contextmanager
    def group(self, name):
        self.parts.append(f'<g id="{escape(name)}" data-label="{escape(name)}">')
        yield
        self.parts.append('</g>')
    def rect(self,x,y,w,h,fill='white',stroke=LINE,r=10,sw=1.6,dash=None):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke if not dash else "none"}" stroke-width="{sw}"/>')
        self.scene.append(dict(type='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke if not dash else 'none',r=r))
        if dash:
            # Editable dash strokes, also supported by minimalist SVG importers.
            for a,b in [((x,y),(x+w,y)),((x+w,y),(x+w,y+h)),((x+w,y+h),(x,y+h)),((x,y+h),(x,y))]:
                dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy);ux,uy=dx/length,dy/length
                for start in range(0,round(length),12):
                    end=min(start+7,length)
                    self.line(a[0]+start*ux,a[1]+start*uy,a[0]+end*ux,a[1]+end*uy,stroke,sw)
    def line(self,x1,y1,x2,y2,col=INK,sw=2,dash=None):
        d=f' stroke-dasharray="{dash}"' if dash else ''
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{sw}" stroke-linecap="round"{d}/>')
        self.scene.append(dict(type='line',x1=x1,y1=y1,x2=x2,y2=y2,col=col,sw=sw))
    def path(self,d,col=INK,sw=2,fill='none'):
        self.parts.append(f'<path d="{d}" fill="{fill}" stroke="{col}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')
        self.scene.append(dict(type='path',d=d,col=col,sw=sw,fill=fill))
    def circle(self,x,y,r,fill=INK,stroke='none',sw=1):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        self.scene.append(dict(type='circle',x=x,y=y,r=r,fill=fill,stroke=stroke))
    def arrow(self,points,col='ink',sw=2.3,both=False,dash=None):
        d='M'+' L'.join(f'{x},{y}' for x,y in points)
        s=f' marker-start="url(#arrow-{col})"' if both else ''
        dd=f' stroke-dasharray="{dash}"' if dash else ''
        c={'ink':INK,'plum':PLUM,'teal':TEAL,'blue':BLUE,'amber':AMBER}[col]
        self.parts.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{sw}" stroke-linejoin="round"{dd}/>')
        self.scene.append(dict(type='arrow',points=points,col=c,sw=sw,both=both))
        # Explicit vector arrowheads survive SVG importers that ignore markers.
        def head(tip, other):
            dx,dy=tip[0]-other[0],tip[1]-other[1]
            length=math.hypot(dx,dy);ux,uy=dx/length,dy/length
            bx,by=tip[0]-9*ux,tip[1]-9*uy
            self.path(f'M{tip[0]},{tip[1]} L{bx-4*uy},{by+4*ux} L{bx+4*uy},{by-4*ux} Z',c,0,c)
        head(points[-1],points[-2])
        if both:head(points[0],points[1])
    def text(self,x,y,text,size=22,bold=False,col=INK,anchor='start',italic=False):
        sty=' font-style="italic"' if italic else ''
        weight='700' if bold else '400'
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{col}" text-anchor="{anchor}"{sty}>{escape(text)}</text>')
        width=font(size,bold).getlength(text)
        self.texts.append(dict(x=x,y=y,text=text,size=size,width=width,anchor=anchor))
        self.scene.append(dict(type='text',x=x,y=y,text=text,size=size,bold=bold,col=col,anchor=anchor,italic=italic))
    def lines(self,x,y,lines,size=22,leading=27,bold=False,col=INK,anchor='start'):
        for i,line in enumerate(lines):self.text(x,y+i*leading,line,size,bold,col,anchor)
    def wrap(self,x,y,text,width,size=22,leading=27,bold=False,col=INK):
        lines=[]
        for para in text.split('\n'):
            line=''
            for word in para.split():
                probe=(line+' '+word).strip()
                if line and font(size,bold).getlength(probe)>width:
                    lines.append(line);line=word
                else:line=probe
            lines.append(line)
        self.lines(x,y,lines,size,leading,bold,col)
        return len(lines)*leading
    def pill(self,x,y,w,h,label,fill=PALE['blue'],col=INK,size=22,bold=True):
        self.rect(x,y,w,h,fill,LINE,r=8)
        self.text(x+w/2,y+h/2+size*.34,label,size,bold,col,anchor='middle')
    def panel(self,x,y,w,h,title,tint='teal',dark=False,fs=29,dash=None):
        self.rect(x,y,w,h,'white',INK if dark else LINE,r=13,dash=dash)
        self.rect(x+1,y+1,w-2,48,INK if dark else PALE[tint],stroke='none',r=12)
        self.rect(x+1,y+35,w-2,15,INK if dark else PALE[tint],stroke='none',r=0)
        self.text(x+18,y+33,title,fs,True,'white' if dark else INK)
    def wave(self,x,y,w,h,col=BLUE,n=120,phase=0):
        # Every stroke is a separate editable line; intentionally schematic.
        self.line(x,y,x+w,y,col,1)
        for k in range(n):
            u=k/(n-1)
            env=.14+.6*(math.sin(3.8*math.pi*u+phase)**2)*(.3+.7*math.sin(1.7*math.pi*u+.4)**2)
            amp=h*(env*(.45+.55*abs(math.sin(2.13*k))))/2
            self.line(x+w*u,y-amp,x+w*u,y+amp,col,1.9)
    def skeleton(self,x,y,s=1,pose=0,col=TEAL,joints=False,lower=False):
        poses=[
          [(0,-52),(0,-32),(-21,-20),(20,-21),(-36,4),(37,0),(-43,29),(47,22),(0,11),(-16,30),(15,31),(-25,60),(30,58),(-33,85),(37,83)],
          [(0,-52),(0,-32),(-20,-20),(20,-20),(-41,-40),(40,-37),(-52,-58),(53,-55),(0,11),(-16,30),(15,31),(-34,58),(38,24),(-47,84),(56,41)],
          [(0,-52),(0,-32),(-21,-20),(20,-20),(-27,-48),(27,-48),(-11,-70),(10,-70),(0,11),(-16,30),(15,31),(-30,60),(26,60),(-38,85),(36,85)],
        ]
        pts=[(x+a*s,y+b*s) for a,b in poses[pose%3]]
        edges=[(0,1),(1,2),(1,3),(2,4),(4,6),(3,5),(5,7),(1,8),(8,9),(8,10),(9,11),(11,13),(10,12),(12,14)]
        for a,b in edges:self.line(*pts[a],*pts[b],PLUM if lower and a>=9 else col,4.8*s)
        self.circle(*pts[0],8*s,col)
        if joints:
            for a,(px,py) in enumerate(pts[1:],1):self.circle(px,py,3.4*s,PLUM if lower and a>=9 else 'white',col,1.5)
    def document(self,x,y,w=38,h=48,col=INK):
        self.rect(x,y,w,h,'white',col,r=3,sw=2)
        for k in range(3):self.line(x+7,y+12+k*9,x+w-7,y+12+k*9,col,2)
    def lock(self,x,y,size=17,col=INK):
        self.path(f'M{x+3},{y+7} v-4 a{size/4},{size/4} 0 0 1 {size/2},0 v4',col,1.8)
        self.rect(x,y+6,size,size*.65,'white',col,r=3,sw=1.5)
        self.circle(x+size/2,y+size*.68,1.3,col)
    def save(self,name):
        self.parts.append('</svg>')
        path=HERE/name
        path.write_text('\n'.join(self.parts)+'\n',encoding='utf-8')
        (HERE/'source'/(name.replace('.editable.svg','.scene.json'))).write_text(json.dumps(dict(width=self.w,height=self.h,objects=self.scene),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        bad=[]
        for t in self.texts:
            left=t['x']-t['width']*(.5 if t['anchor']=='middle' else 1 if t['anchor']=='end' else 0)
            if left<0 or left+t['width']>self.w+1 or t['y']>self.h or t['y']-t['size']<0:bad.append(t['text'])
        assert not bad, bad
        assert '<image' not in path.read_text(encoding='utf-8')
        return path

def method():
    s=SVG(1800,960,'Method overview: trained movement expert and stateful compositional harness')
    with s.group('source-music'):
        s.rect(20,20,1355,105,PALE['gray'],INK)
        s.text(45,65,'Music',31,True)
        s.wave(180,65,1165,58,n=190)
        for x,label in [(210,'Bars 1–2'),(590,'Bars 3–4'),(970,'Bars 5–6')]:
            s.line(x,108,x+330,108,BLUE,1.5);s.text(x+165,102,label,20,True,anchor='middle')
        s.rect(1400,20,380,105,PALE['gray'],INK)
        s.document(1420,40,35,51)
        s.text(1475,62,'Choreographic brief',25,True)
        s.lines(1475,90,['Optional user conditions'],20,col=BLUE)
    with s.group('shared-evidence-and-trained-model'):
        s.panel(20,150,480,730,'Specialists and shared evidence',fs=27)
        s.rect(36,212,448,336,PALE['plum'],PLUM)
        s.text(52,243,'Our trained choreography expert',25,True)
        for y,h,lines in [(268,60,['Audio front end']),(351,70,['Qwen3-Omni','+ LoRA']),(444,58,['Local movement','proposals'])]:
            s.rect(50,y,208,h,'white',PLUM)
            s.lines(154,y+(29 if len(lines)==1 else 27),lines,23,26,True,anchor='middle')
        s.arrow([(154,329),(154,347)],'plum');s.arrow([(154,423),(154,440)],'plum')
        s.rect(272,268,196,234,'white',PLUM)
        s.lines(286,295,['Joint-aware','text supervision'],22,26,True,PLUM)
        s.text(285,365,'… raise the left arm;',18)
        s.rect(282,377,178,54,PALE['plum'],'none',r=3)
        s.lines(287,398,['bend the right knee','and lower both legs …'],16,23,True,PLUM)
        s.skeleton(430,466,.25,joints=True,lower=True)
        s.lines(285,465,['Clause-level','text weights'],18,22)
        s.pill(50,516,418,23,'SFT-trained local expert',PALE['plum'],PLUM,18)
        s.text(260,582,'Parallel source-audio analysis',22,True,anchor='middle')
        for x,title,lines,col in [(36,'Beat This!',['2–4-bar','windows'],BLUE),(190,'JEV',['Global dance-','style prior'],TEAL),(344,'librosa',['Acoustic','summaries'],BLUE)]:
            s.rect(x,600,140,144,'white',col)
            s.text(x+70,630,title,23,True,anchor='middle')
            s.lines(x+70,660,lines,20,24,anchor='middle')
            if title=='Beat This!':
                for i in range(9):s.line(x+17+i*13,703,x+17+i*13,729-(i%3)*8,col,2)
            elif title=='JEV':
                s.pill(x+12,709,116,24,'Style condition',PALE['teal'],TEAL,16)
            else:s.wave(x+13,716,114,31,col,n=35)
        s.rect(36,765,448,96,PALE['blue'],BLUE)
        s.text(51,794,'Frozen evidence + proposal cache',23,True)
        for i,(bar,txt) in enumerate([('Bars 1–2','“step right …”'),('Bars 3–4','“turn, open …”'),('Bars 5–6','“lower, step …”')]):
            x=48+i*141;s.rect(x,807,132,43,'white',BLUE,r=5)
            s.text(x+66,824,bar,17,True,anchor='middle');s.text(x+66,844,txt,16,anchor='middle')
    s.arrow([(95,125),(95,146)])
    with s.group('stateful-harness'):
        s.panel(550,150,825,730,'Stateful compositional harness',fs=29)
        # Source proposal cache is read-only, distinct from mutable execution state.
        s.arrow([(484,820),(522,820),(522,447),(557,447)])
        s.rect(501,650,43,97,'white','none',r=0)
        s.lines(522,669,['Read-','only','evidence'],16,20,anchor='middle')
        s.rect(565,244,210,215,PALE['plum'],PLUM)
        s.text(670,274,'Composer agent',24,True,PLUM,anchor='middle')
        s.text(670,315,'Qwen3-VL',25,True,anchor='middle')
        s.lines(670,355,['Outline · motifs','Unit drafts'],23,28,anchor='middle')
        s.lock(590,417,19,PLUM);s.lines(624,417,['Fixed weights','at inference'],18,22,col=PLUM)
        s.rect(870,244,210,215,PALE['teal'],TEAL)
        s.text(975,274,'Controller',25,True,anchor='middle')
        for y,txt in [(307,'Schedule stages'),(356,'Budget + permissions'),(405,'Continue / stop')]:s.pill(882,y,186,36,txt,'white',INK,18,False)
        s.rect(1160,244,195,215,PALE['amber'],AMBER)
        s.text(1257,274,'Tool execution',23,True,anchor='middle')
        for y,txt in [(307,'Read evidence / history'),(345,'Validate schedule'),(383,'Commit unit'),(421,'Patch selected text')]:s.pill(1171,y,173,28,txt,'white',INK,16,False)
        s.arrow([(866,310),(779,310)]);s.lines(820,270,['Context +','feedback'],18,22,anchor='middle')
        s.arrow([(779,389),(866,389)]);s.lines(820,418,['Structured','proposal'],18,22,anchor='middle')
        s.arrow([(1084,312),(1155,312)]);s.text(1119,293,'Execute',18,anchor='middle')
        s.arrow([(1155,389),(1084,389)]);s.lines(1120,417,['Structural','feedback /','error'],16,19,anchor='middle')
        s.panel(565,498,790,137,'Persistent state','blue',fs=28)
        for i,lines in enumerate([['Outline +','motifs'],['Current','draft'],['Committed','history'],['Call / token','ledger']]):
            x=565+i*197.5
            if i:s.line(x,549,x,620,LINE,1)
            s.lines(x+98.75,582,lines,22,28,anchor='middle')
        s.arrow([(975,464),(975,493)],both=True);s.arrow([(1257,464),(1257,493)],both=True)
        s.rect(565,683,790,177,'white',BLUE)
        s.rect(566,684,788,40,PALE['blue'],'none',r=9)
        s.text(584,712,'Targeted revision',27,True)
        for i,bar in enumerate(['Bars 1–2','Bars 3–4','Bars 5–6','Bars 7–8','Bars 9–10']):
            x=580+i*151;s.rect(x,738,142,82,PALE['amber'] if i==2 else PALE['blue'],AMBER if i==2 else BLUE,sw=2.3 if i==2 else 1.4)
            if i==2:s.lines(x+71,763,['Revise selected','text'],18,23,True,anchor='middle')
            else:s.lock(x+61,747,16);s.text(x+71,787,'Protected context',17,anchor='middle')
            s.text(x+71,810,bar,17,anchor='middle')
        s.text(960,847,'Keep time · track · motif fixed',22,True,anchor='middle')
    s.arrow([(1590,125),(1590,136),(1120,136),(1120,217),(975,217),(975,239)])
    with s.group('whole-song-score'):
        s.panel(1400,150,380,730,'Whole-song textual score',fs=26)
        s.pill(1415,212,350,43,'Dance style is contemporary jazz.',PALE['teal'],INK,20)
        for i,(bar,motif,lines) in enumerate([
            ('Bars 1–2','A',['Step right; swing the','left arm across; open','the chest and settle','into a lunge.']),
            ('Bars 3–4','B',['Turn left; raise both','arms overhead; step','forward, then lower','with control.']),
            ('Bars 5–6','A′',['Vary the opening:','step right, sweep the','left arm wider, and','sink lower.'])]):
            y=278+i*178
            if i:s.line(1416,y-12,1765,y-12,LINE,1)
            s.pill(1418,y,111,32,bar,PALE['blue'],INK,19)
            s.pill(1542,y,56,32,motif,PALE['plum'] if i!=1 else PALE['teal'],PLUM,23)
            s.skeleton(1460,y+91,.60,pose=i,col=TEAL,joints=False)
            s.lines(1532,y+63,lines,20,26)
        s.pill(1416,829,348,35,'Inspect · edit · compose',PALE['gray'],INK,23,False)
    s.arrow([(1358,361),(1395,361)])
    with s.group('legend'):
        s.text(28,924,'Legend',22,True)
        for x,col,txt in [(125,'plum','Learned models'),(485,'teal','Control and scheduling'),(945,'amber','Deterministic tools'),(1395,'blue','Persistent state')]:
            s.rect(x,905,35,22,PALE[col],{'plum':PLUM,'teal':TEAL,'amber':AMBER,'blue':BLUE}[col],r=5)
            s.text(x+49,924,txt,21)
    return s.save('overview_method.editable.svg')

def data_benchmark():
    s=SVG(1800,930,'Data, benchmark, and baseline overview')
    with s.group('annotation'):
        s.panel(20,20,640,855,'(a) Evidence-linked annotation',dark=True,fs=28)
        # Source-media film strip: schematic vector figures, no raster photos.
        s.rect(40,85,355,140,'#eff3f7',INK,r=8,sw=3)
        for k in range(17):
            s.rect(47+k*20,91,11,8,'white',INK,r=1,sw=.6);s.rect(47+k*20,211,11,8,'white',INK,r=1,sw=.6)
        for k in range(3):
            s.rect(48+k*114,107,105,95,'white',LINE,r=3)
            s.skeleton(100+k*114,151,.51,pose=k,col=INK,joints=True)
        s.wave(425,154,193,78,TEAL,n=45)
        s.arrow([(622,156),(675,156)],'teal')
        s.lines(632,97,['Retain','aligned','audio'],17,20,True,anchor='middle')
        s.pill(40,252,182,32,'CURRENT LABEL',PALE['teal'],INK,20)
        for x,w,label in [(40,190,'1  2D rule events'),(282,142,'2  GPT-6.1-sol'),(468,170,'3  Linked captions')]:s.pill(x,310,w,42,label,PALE['plum'],INK,19)
        s.arrow([(234,331),(277,331)]);s.arrow([(428,331),(463,331)])
        s.rect(40,371,274,175,'white',LINE)
        s.skeleton(85,440,.56,joints=True,col=INK)
        for i,(lab,txt,col) in enumerate([('E1','Left wrist ↑',AMBER),('E2','Right wrist →',BLUE),('E3','Knee angle ↓',TEAL),('E4','Knee angle ↑',TEAL)]):
            y=395+i*36;s.pill(127,y-17,36,25,lab,PALE['blue'],col,15)
            s.text(171,y+1,txt,17,True);s.line(171,y+12,292-(i%2)*22,y+12,col,3)
        s.rect(330,371,114,175,'white',LINE)
        s.document(364,395,44,58,BLUE)
        s.lines(387,492,['Grounded','rewrite'],18,23,True,anchor='middle')
        s.text(387,532,'Text only',17,False,BLUE,anchor='middle')
        s.rect(461,371,177,175,'white',LINE)
        s.lines(474,402,['The left arm rises','as the knees bend.'],17,25,True,TEAL)
        s.line(500,438,500,475,AMBER,1.5);s.line(590,438,590,475,TEAL,1.5)
        s.pill(478,478,45,28,'E1',PALE['amber'],AMBER,18);s.pill(568,478,45,28,'E3',PALE['teal'],TEAL,18)
        s.arrow([(642,459),(675,459)],'plum')
        s.rect(40,574,598,67,PALE['plum'],PLUM)
        s.text(54,601,'Refinement checks',20,True)
        for x,y,t in [(245,597,'Coverage'),(414,597,'Body + direction'),(245,627,'Timing + repetition'),(456,627,'Grounded prose')]:
            s.rect(x-18,y-13,10,10,'white',PLUM,r=1,sw=1);s.text(x,y,t,17)
        s.panel(40,672,598,184,'3D geometry extension','blue',fs=23,dash='7 5')
        s.pill(57,730,95,31,'NLF',PALE['blue'],INK,19)
        s.arrow([(157,746),(190,746)]);s.pill(196,730,159,31,'Temporal events',PALE['plum'],INK,18)
        s.arrow([(360,746),(393,746)]);s.pill(399,730,221,31,'Evidence consolidation',PALE['plum'],INK,18)
        s.skeleton(91,803,.34,joints=True)
        # Palm and leg details correspond to different body locations.
        s.rect(142,779,65,30,'white',BLUE,r=5);s.rect(142,816,65,30,'white',BLUE,r=5)
        s.line(104,798,142,794,PLUM,1.2);s.line(97,824,142,831,PLUM,1.2)
        for j in range(4):s.line(160+j*7,800,155+j*9,786-j%2*3,TEAL,2)
        s.path('M157,825 L169,833 L189,823',TEAL,3)
        for i,lab in enumerate(['Bend','Extend','Bend','Extend']):
            x=234+i*97;s.pill(x,779,86,28,lab,PALE['teal'] if i%2==0 else PALE['amber'],INK,17)
            s.line(x+5,820,x+79,820,TEAL if i%2==0 else AMBER,4)
        s.text(428,845,'Summarize; retain source links',17,True,PLUM,anchor='middle')
        s.arrow([(620,671),(620,649)],'plum',dash='5 4')
    with s.group('paired-resource'):
        s.panel(680,20,430,615,'(b) Paired music–choreography',dark=True,fs=25)
        examples=[['The left arm rises','as the knees bend.'],['Step to the right','with an arm sweep.'],['Turn left with','a wide arm circle.'],['Make a small jump','and open the arms.']]
        for i,lines in enumerate(examples):
            y=88+i*98;s.rect(695,y,400,80,'white',LINE)
            s.wave(709,y+40,97,43,[TEAL,BLUE,PLUM,TEAL][i],n=28)
            s.text(821,y+44,'↔',25,True)
            s.rect(846,y+12,236,55,PALE[['teal','blue','plum','teal'][i]],'none',r=6)
            s.lines(856,y+33,lines,20,24,True,[TEAL,BLUE,PLUM,TEAL][i])
        s.rect(695,494,400,55,PALE['plum'],PLUM)
        s.text(895,517,'Record schema',22,True,anchor='middle')
        s.text(895,540,'Audio · text · time · provenance',20,anchor='middle')
        s.pill(696,575,183,37,'Training',PALE['teal'],INK,21)
        s.pill(898,575,197,37,'Evaluation',PALE['plum'],INK,21)
        s.arrow([(788,615),(788,661)],'teal')
        s.arrow([(1098,594),(1125,594)],'plum')
    with s.group('benchmark'):
        s.panel(1130,20,650,615,'(c) Music-to-text benchmark',dark=True,fs=28)
        s.rect(1148,90,455,105,PALE['teal'],TEAL)
        s.text(1163,115,'Paired evaluation inputs',21,True)
        s.wave(1163,153,165,40,TEAL,n=35)
        s.text(1347,163,'+',29)
        s.rect(1381,128,209,53,'white',BLUE)
        s.document(1392,139,23,31,BLUE)
        s.lines(1428,149,['Candidate','choreography'],19,23,True)
        s.rect(1620,90,142,105,PALE['amber'],AMBER)
        s.lines(1691,115,['Reference','agreement'],19,23,True,anchor='middle')
        s.lines(1691,162,['Complementary','diagnostic'],16,20,anchor='middle')
        s.text(1149,226,'Evidence process uses the raw audio',21,True)
        for x,w,lines in [(1148,156,['Listen']),(1333,213,['Quote movement','evidence']),(1576,186,['Rate','independently'])]:
            s.rect(x,243,w,61,'white',TEAL);s.lines(x+w/2,268,lines,20,24,True,anchor='middle')
        s.arrow([(1308,274),(1328,274)]);s.arrow([(1550,274),(1571,274)])
        for y,title,sub,col,tint in [(329,'Local movement content','Specificity · body / space · local timing',BLUE,'blue'),(400,'Music–movement response','Audio-supported movement choices',AMBER,'amber'),(471,'Whole-song composition','Continuity · structure · motif variation',PLUM,'plum')]:
            s.rect(1148,y,614,58,PALE[tint],col)
            s.text(1167,y+25,title,24,True)
            s.text(1167,y+49,sub,20)
        s.rect(1148,549,614,69,PALE['teal'],TEAL)
        s.text(1162,574,'Strict rubric',21,True)
        for x,num,lines,col in [(1300,'1',['Unsupported /','conflicting'],AMBER),(1472,'3',['Partially','supported'],AMBER),(1624,'5',['Evidence-','supported'],TEAL)]:
            s.pill(x,560,28,29,num,'white',col,19);s.lines(x+37,574,lines,16,20)
        s.text(1162,603,'Reasons + evidence',17,False,TEAL)
    with s.group('baseline-method'):
        s.panel(680,664,1100,211,'Our baseline: local expert + compositional harness',dark=True,fs=27)
        s.rect(699,732,220,119,PALE['plum'],PLUM)
        s.lines(809,762,['Our trained','Qwen3-Omni'],24,28,True,anchor='middle')
        s.lines(809,822,['Joint-aware clause','supervision'],18,22,anchor='middle')
        s.arrow([(923,787),(946,787)])
        s.rect(950,722,505,140,'white',PLUM)
        s.text(965,747,'Stateful harness',22,True)
        s.pill(1218,729,59,25,'JEV',PALE['teal'],TEAL,17)
        s.pill(1286,729,151,25,'Bar-aligned units',PALE['amber'],AMBER,16)
        for x,lab in [(967,'Planner'),(1139,'Controller'),(1311,'Tools')]:s.pill(x,771,127,35,lab,PALE['blue'] if lab!='Controller' else PALE['plum'],INK,20)
        s.arrow([(1099,788),(1134,788)],both=True);s.arrow([(1271,788),(1306,788)],both=True)
        s.pill(966,825,472,25,'Outline · motifs · committed history',PALE['plum'],INK,17,False)
        s.arrow([(1459,787),(1492,787)])
        s.rect(1498,723,264,139,PALE['blue'],BLUE)
        s.text(1630,749,'Timed choreography text',20,True,anchor='middle')
        s.text(1514,777,'Dance style is …',20,True)
        s.lines(1514,806,['Bars 1–2: …','Bars 3–4: …'],19,25)
        # Evaluation is outside the generation loop. Route along the outer edge.
        s.arrow([(1764,788),(1793,788),(1793,211),(1609,211),(1609,155),(1597,155)],'plum')
    s.text(900,910,'Shared evidence · controlled comparisons · explicit resource budgets',22,True,anchor='middle')
    return s.save('overview_data_benchmark.editable.svg')

if __name__ == '__main__':
    HERE.mkdir(parents=True,exist_ok=True)
    paths=[method(),data_benchmark()]
    import fitz
    for p in paths:
        doc=fitz.open(p)
        raster=fitz.open('pdf',doc.convert_to_pdf())
        raster[0].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False).save(p.with_suffix('.preview.png'))
        print(p)
