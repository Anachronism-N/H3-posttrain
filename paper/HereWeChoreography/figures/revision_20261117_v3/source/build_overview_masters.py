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
INK='#202C3C'; BORDER='#B7C6D2'
SLATE='#2474B5'; SAGE='#008F95'; PLUM='#7852A9'; OCHRE='#E69A28'
FILL={'slate':'#FAFCFD','sage':'#FAFCFD','plum':'#FBFAFE','ochre':'#FFFCF7'}
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
        # Real source-frame images are permitted; labels and diagrams remain native SVG.
        assert all(e.attrib.get('font-family')==FONT for e in r.findall('.//'+ns+'text'))
        return p


from overview_v3_layout import method_v3, benchmark_v3

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    paths=[method_v3(Fig,ROOT),benchmark_v3(Fig,ROOT)]
    import fitz
    audit=[]
    for p in paths:
        svg=fitz.open(p);doc=fitz.open('pdf',svg.convert_to_pdf())
        doc[0].get_pixmap(matrix=fitz.Matrix(.8,.8),alpha=False).save(p.with_suffix('.preview.png'))
        root=ET.parse(p).getroot();types={}
        for e in root.iter():
            tag=e.tag.rsplit('}',1)[-1];types[tag]=types.get(tag,0)+1
        audit.append(dict(file=p.name,font='Times New Roman',elements=types,embedded_images=types.get('image',0),native_text=True,scope='Vector master; image elements only genuine source-video stills'))
        print(p)
    (ROOT/'source/overview_master_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
