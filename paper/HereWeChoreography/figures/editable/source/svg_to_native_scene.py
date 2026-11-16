"""Flatten editable SVGs into native PowerPoint scene primitives.

No full-figure rasterization. Text remains text; photos remain independent
images; lines, chart marks, waveforms and skeletons become native freeforms.
"""
from pathlib import Path
from lxml import etree as ET
from fontTools.svgLib.path.parser import parse_path
from fontTools.pens.recordingPen import RecordingPen
from PIL import ImageFont
import re,math,json,base64

ROOT=Path(__file__).resolve().parents[1]
ORDER=['overview_method','overview_data_benchmark','annotation_pipeline','evaluation_rubric',
       'agent_policy_comparison','dataset_construction','annotation_distributions','choreography_example']
IDENT=(1.,0.,0.,1.,0.,0.)
def mul(a,b):
    A,B,C,D,E,F=a;x,y,z,w,u,v=b
    return (A*x+C*y,B*x+D*y,A*z+C*w,B*z+D*w,A*u+C*v+E,B*u+D*v+F)
def point(m,x,y):return (m[0]*x+m[2]*y+m[4],m[1]*x+m[3]*y+m[5])
def num(x,default=0.):
    try:return float(re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',str(x)).group())
    except (AttributeError,ValueError):return default
def nums(x):return [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',str(x))]
def transform(s):
    m=IDENT
    for kind,values in re.findall(r'(\w+)\s*\(([^)]*)\)',s or ''):
        n=nums(values)
        if kind=='matrix':t=tuple(n)
        elif kind=='translate':t=(1,0,0,1,n[0],n[1] if len(n)>1 else 0)
        elif kind=='scale':t=(n[0],0,0,n[1] if len(n)>1 else n[0],0,0)
        elif kind=='rotate':
            a=math.radians(n[0]);t=(math.cos(a),math.sin(a),-math.sin(a),math.cos(a),0,0)
            if len(n)>2:t=mul(mul((1,0,0,1,n[1],n[2]),t),(1,0,0,1,-n[1],-n[2]))
        else:continue
        m=mul(m,t)
    return m
def style(el,inherited):
    s=dict(inherited)
    for k in ['fill','stroke','stroke-width','font-size','font-family','font-weight','font-style',
              'text-anchor','dominant-baseline','marker-end','marker-start','opacity','fill-opacity','stroke-opacity','stroke-dasharray']:
        if k in el.attrib:s[k]=el.attrib[k]
    for item in el.get('style','').split(';'):
        if ':' in item:
            k,v=item.split(':',1);s[k.strip()]=v.strip()
    if 'font' in s:
        z=re.search(r'(?:(\d{3})\s+)?([\d.]+)px\s+(.+)',s['font'])
        if z:s.update({'font-size':z[2],'font-family':z[3],'font-weight':z[1] or 'normal'})
        s.pop('font',None)
    return s
def color(v,default='none'):
    if not v:return default
    if v.startswith('url('):return '#e7eef5'
    return v
def painted(s,channel,default='none'):
    c=color(s.get(channel),default)
    opacity=num(s.get('opacity'),1)*num(s.get(channel+'-opacity'),1)
    if opacity<1 and re.fullmatch(r'#[0-9a-fA-F]{6}',c):
        values=[int(c[i:i+2],16) for i in [1,3,5]]
        return '#'+''.join(f'{round(255*(1-opacity)+x*opacity):02x}' for x in values)
    return c
FONTS={}
def measure(t,size,bold,serif):
    fn=('timesbd.ttf' if bold else 'times.ttf') if serif else ('arialbd.ttf' if bold else 'arial.ttf')
    if fn not in FONTS:FONTS[fn]=ImageFont.truetype('C:/Windows/Fonts/'+fn,100)
    return FONTS[fn].getlength(t)*size/100

class Converter:
    def __init__(self,file):
        self.file=file;self.root=ET.parse(str(file)).getroot();self.ids={e.get('id'):e for e in self.root.iter() if e.get('id')};self.items=[];self.ignored=set()
        vb=nums(self.root.get('viewBox',''))
        self.vb=vb if len(vb)==4 else [0,0,num(self.root.get('width')),num(self.root.get('height'))]
        self.scale=min(1536/self.vb[2],836/self.vb[3]);self.ox=(1600-self.vb[2]*self.scale)/2;self.oy=(900-self.vb[3]*self.scale)/2
    def add_path(self,commands,s,m,name):
        mapped=[];xy=[]
        for op,pts in commands:
            if op in ['endPath']:continue
            q=[point(m,*p) for p in pts if p is not None];xy+=q;mapped.append((op,q))
        if not xy:return
        x0=min(x for x,y in xy);y0=min(y for x,y in xy);w=max(.1,max(x for x,y in xy)-x0);h=max(.1,max(y for x,y in xy)-y0)
        cmds=[]
        for op,pts in mapped:
            q=[(x-x0,y-y0) for x,y in pts]
            if op=='moveTo':cmds.append({'moveTo':dict(x=q[0][0],y=q[0][1])})
            elif op=='lineTo':
                for x,y in q:cmds.append({'lineTo':dict(x=x,y=y)})
            elif op=='curveTo':
                for i in range(0,len(q),3):
                    a,b,c=q[i:i+3];cmds.append({'cubicBezTo':dict(x1=a[0],y1=a[1],x2=b[0],y2=b[1],x=c[0],y=c[1])})
            elif op=='qCurveTo':
                a,b=q[:2];cmds.append({'quadBezTo':dict(x1=a[0],y1=a[1],x=b[0],y=b[1])})
            elif op=='closePath':cmds.append({'close':{}})
        self.items.append(dict(kind='path',name=name,x=x0,y=y0,w=w,h=h,commands=cmds,fill=painted(s,'fill'),stroke=painted(s,'stroke'),sw=num(s.get('stroke-width'),1)*math.hypot(m[0],m[1]),dash=bool(s.get('stroke-dasharray'))))
        if s.get('marker-end') and len(xy)>1:
            a,b=xy[-2:];ang=math.atan2(b[1]-a[1],b[0]-a[0]);l=max(6,6*num(s.get('stroke-width'),1)*self.scale);r=l*.45
            tri=[b,(b[0]-l*math.cos(ang)+r*math.sin(ang),b[1]-l*math.sin(ang)-r*math.cos(ang)),(b[0]-l*math.cos(ang)-r*math.sin(ang),b[1]-l*math.sin(ang)+r*math.cos(ang))]
            self.add_path([('moveTo',[tri[0]]),('lineTo',tri[1:]),('closePath',[])],{'fill':s.get('stroke','#20384b')},IDENT,name+' arrowhead')
    def walk(self,el,m=IDENT,st=None,group=''):
        tag=ET.QName(el).localname if isinstance(el.tag,str) else ''
        if tag in ['defs','metadata','title','desc','clipPath','style']:return
        s=style(el,st or {});m=mul(m,transform(el.get('transform','')));name=el.get('id') or (group+'/'+tag)
        if tag in ['svg','g']:
            for child in el:self.walk(child,m,s,name)
            return
        if tag=='use':
            link=el.get('{http://www.w3.org/1999/xlink}href') or el.get('href','')
            ref=self.ids.get(link.lstrip('#'))
            if ref is not None:self.walk(ref,mul(m,(1,0,0,1,num(el.get('x')),num(el.get('y')))),s,name)
            return
        if tag=='text':
            text=''.join(el.itertext())
            if not text.strip():return
            x,y=point(m,num(el.get('x')),num(el.get('y')));size=num(s.get('font-size'),16)*math.hypot(m[0],m[1]);bold=s.get('font-weight','normal') in ['bold','600','700','800','900'];serif='STIX' in s.get('font-family','') or 'Times' in s.get('font-family','')
            width=max(size*.25,measure(text,size,bold,serif));anchor=s.get('text-anchor','start');rotation=math.degrees(math.atan2(m[1],m[0]))
            shift=width/2 if anchor=='middle' else width if anchor=='end' else 0
            baseline=s.get('dominant-baseline','');height=size*1.24
            local_y=height/2-size*(.48 if baseline in ['middle','central'] else .88)
            factor=math.hypot(m[0],m[1]);ux,uy=m[0]/factor,m[1]/factor;vx,vy=m[2]/factor,m[3]/factor
            cx=x+ux*(width/2-shift)+vx*local_y;cy=y+uy*(width/2-shift)+vy*local_y
            self.items.append(dict(kind='text',name=name,text=text,x=cx-width/2-1,y=cy-height/2,w=width+max(3,size*.14),h=height,size=size,bold=bold,italic=s.get('font-style')=='italic',font='Times New Roman' if serif else 'Arial',fill=color(s.get('fill'),'#20384b'),rotation=rotation))
            return
        if tag=='image':
            x,y,w,h=[num(el.get(v)) for v in ['x','y','width','height']];a=point(m,x,y);b=point(m,x+w,y+h)
            href=el.get('{http://www.w3.org/1999/xlink}href') or el.get('href')
            self.items.append(dict(kind='image',name=name,x=min(a[0],b[0]),y=min(a[1],b[1]),w=abs(b[0]-a[0]),h=abs(b[1]-a[1]),data=href,flipV=m[3]<0,flipH=m[0]<0))
            return
        if tag in ['rect','circle','ellipse']:
            if tag=='rect':x,y,w,h=[num(el.get(v)) for v in ['x','y','width','height']];rx=num(el.get('rx'))
            else:
                rx=num(el.get('r') or el.get('rx'));ry=num(el.get('r') or el.get('ry'));x=num(el.get('cx'))-rx;y=num(el.get('cy'))-ry;w=2*rx;h=2*ry
            a=point(m,x,y);b=point(m,x+w,y+h)
            self.items.append(dict(kind='shape',name=name,geometry='ellipse' if tag!='rect' else 'roundRect' if rx else 'rect',x=min(a[0],b[0]),y=min(a[1],b[1]),w=max(.1,abs(b[0]-a[0])),h=max(.1,abs(b[1]-a[1])),radius=rx*math.hypot(m[0],m[1]) if tag=='rect' else 0,fill=painted(s,'fill','#000000'),stroke=painted(s,'stroke'),sw=num(s.get('stroke-width'),1)*math.hypot(m[0],m[1]),dash=bool(s.get('stroke-dasharray'))))
            return
        if tag in ['line','polyline','polygon','path']:
            if tag=='line':cmds=[('moveTo',[(num(el.get('x1')),num(el.get('y1')))]),('lineTo',[(num(el.get('x2')),num(el.get('y2')))])]
            elif tag in ['polyline','polygon']:
                v=nums(el.get('points'));pts=list(zip(v[::2],v[1::2]));cmds=[('moveTo',pts[:1]),('lineTo',pts[1:])]+([('closePath',[])] if tag=='polygon' else [])
            else:
                pen=RecordingPen();parse_path(el.get('d',''),pen);cmds=pen.value
            self.add_path(cmds,s,m,name);return
        self.ignored.add(tag)
    def run(self):
        m=(self.scale,0,0,self.scale,self.ox-self.vb[0]*self.scale,self.oy-self.vb[1]*self.scale)
        self.walk(self.root,m,{'fill':'#000000'})
        return dict(file=self.file.name,title=self.file.stem.replace('.editable','').replace('_',' '),items=self.items,ignored=sorted(self.ignored))

if __name__=='__main__':
    slides=[]
    for name in ORDER:
        f=ROOT/(name+'.editable.svg')
        if not f.is_file():raise FileNotFoundError(f)
        slide=Converter(f).run();slides.append(slide)
        print(name,len(slide['items']),'objects','ignored',slide['ignored'])
    target=ROOT/'source/native_scenes.json'
    target.write_text(json.dumps(dict(width=1600,height=900,slides=slides),ensure_ascii=False,separators=(',',':')),encoding='utf-8')
