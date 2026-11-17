"""Rebuild three paper diagrams as native, text-editable SVG objects.

No raster image is embedded. Preview waveforms and poses are schematic icons.
Times New Roman and restrained scientific panel styling, revision 2026-11-17.
Run this file with Python 3; output paths are relative to this source file.
"""
from pathlib import Path
from html import escape
import math
import re

OUT = Path(__file__).resolve().parents[1]
OUT.mkdir(parents=True, exist_ok=True)
INK='#242A33'; TEAL='#667A70'; BLUE='#65778A'; PURPLE='#85778C'; GOLD='#A18C69'
LIGHT_TEAL='#FBFCFA'; LIGHT_BLUE='#FAFBFD'; LIGHT_PURPLE='#FCFAFD'; LIGHT_GOLD='#FDFBFA'


class SVG:
    def __init__(self,w,h):
        self.w,self.h=w,h
        self.n=0
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
          '<title>Editable scientific diagram</title>',
          '<desc>Native text, shapes, paths and groups. Every label and connector can be edited individually. Schematic icons are not measured data.</desc>',
          '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>',
          '<rect width="100%" height="100%" fill="white"/>']
    def raw(self,s): self.parts.append(s)
    def group(self,name): self.raw(f'<g id="{escape(name,quote=True)}">')
    def end(self): self.raw('</g>')
    def box(self,x,y,w,h,fill='white',stroke=INK,r=14,sw=1.5,dash=None):
        r=min(r,5)
        sw=min(sw,1.25)
        if stroke!='none': stroke='#B7BFC7'
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def text(self,x,y,s,size=22,weight='normal',color=INK,anchor='start',italic=False):
        lines=s if isinstance(s,list) else str(s).split('\n')
        for i,line in enumerate(lines):
            self.raw(f'<text x="{x}" y="{y+i*size*1.22}" font-family="Times New Roman" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}"'+(' font-style="italic"' if italic else '')+f'>{escape(str(line))}</text>')
    def path(self,d,stroke=INK,sw=2,fill='none',arrow=False,dash=None):
        self.raw(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
        if arrow:
            # Explicit, individually editable triangle: works in SVG/PPT/PDF renderers.
            values=[float(v) for v in re.findall(r'[-+]?(?:\d*\.\d+|\d+)',d)]
            px,py,x,y=values[-4:]
            a=math.atan2(y-py,x-px); n=8+sw; half=n*.43
            bx,by=x-n*math.cos(a),y-n*math.sin(a)
            ax,ay=bx+half*math.sin(a),by-half*math.cos(a)
            cx,cy=bx-half*math.sin(a),by+half*math.cos(a)
            self.raw(f'<path d="M{x},{y} L{ax:.2f},{ay:.2f} L{cx:.2f},{cy:.2f} Z" fill="{stroke}" stroke="none"/>')
    def line(self,x1,y1,x2,y2,color=INK,sw=2,arrow=False,dash=None):
        self.path(f'M{x1},{y1} L{x2},{y2}',color,sw,arrow=arrow,dash=dash)
    def circle(self,x,y,r,fill=TEAL,stroke=None,sw=1.5):
        self.raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{sw}"/>')
    def badge(self,x,y,n,color,r=23):
        self.circle(x,y,r,color)
        self.text(x,y+9,str(n),29,'bold','white','middle')
    def chip(self,x,y,w,h,label,fill=LIGHT_TEAL,color=TEAL,fs=19,weight='normal'):
        self.box(x,y,w,h,fill,color,r=10,sw=1.2)
        self.text(x+w/2,y+h/2+fs*.34,label,fs,weight,color,'middle')
    def finish(self,name):
        self.raw('</svg>')
        target=OUT/name
        target.write_text('\n'.join(self.parts)+'\n',encoding='utf-8')
        return target


def wave(s,x,y,w,h,color=BLUE,seed=1):
    s.group(f'waveform-{seed}-{int(x)}-{int(y)}')
    for i in range(80):
        a=abs(math.sin(i*.71+seed)*math.sin(i*.137+seed*.25))
        envelope=.25+.75*abs(math.sin(i*.054+seed))
        v=(.1+.9*a)*envelope*h/2
        s.line(x+i*w/79,y-v,x+i*w/79,y+v,color,1.8)
    s.end()


def body(s,cx,top,scale=1,pose=0,joints=True,color=TEAL):
    """A body diagram with editable limb segments and joint markers."""
    pts={'head':(0,0),'neck':(0,19),'ls':(-19,27),'rs':(19,27),
         'le':(-28,54),'re':(30,51),'lw':(-39,79),'rw':(42,76),
         'hip':(0,72),'lh':(-13,76),'rh':(13,76),
         'lk':(-18,112),'rk':(22,111),'la':(-23,147),'ra':(29,147)}
    if pose==1:
        pts.update(le=(-36,15),lw=(-55,-1),re=(34,13),rw=(53,-8),lk=(-26,111),la=(-39,147))
    elif pose==2:
        pts.update(le=(-35,54),lw=(-5,63),re=(38,42),rw=(56,24),lk=(-31,105),la=(-21,137),rk=(33,106),ra=(43,136))
    elif pose==3:
        pts.update(le=(-31,34),lw=(-55,18),re=(35,17),rw=(49,-10),lk=(-13,111),la=(-26,147),rk=(35,99),ra=(13,124))
    p={k:(cx+a*scale,top+b*scale) for k,(a,b) in pts.items()}
    # Filled torso and tapered, rounded limb segments reproduce body mass,
    # while every contour and overlaid landmark remains editable.
    ls,rs,lh,rh=p['ls'],p['rs'],p['lh'],p['rh']
    s.path(f"M{ls[0]},{ls[1]} Q{cx},{top+13*scale} {rs[0]},{rs[1]} L{rh[0]+3*scale},{rh[1]} Q{cx},{top+84*scale} {lh[0]-3*scale},{lh[1]} Z",color,0.8,color)
    def segment(a,b,r1,r2):
        x1,y1=p[a];x2,y2=p[b];a0=math.atan2(y2-y1,x2-x1)
        nx,ny=-math.sin(a0),math.cos(a0)
        q=[(x1+nx*r1*scale,y1+ny*r1*scale),(x2+nx*r2*scale,y2+ny*r2*scale),(x2-nx*r2*scale,y2-ny*r2*scale),(x1-nx*r1*scale,y1-ny*r1*scale)]
        s.path('M'+' L'.join(f'{a},{b}' for a,b in q)+' Z',color,.5,color)
        s.circle(x1,y1,r1*scale,color);s.circle(x2,y2,r2*scale,color)
    for a,b,r1,r2 in [('ls','le',6.5,5.1),('le','lw',5.1,3.6),('rs','re',6.5,5.1),('re','rw',5.1,3.6),('lh','lk',8.2,5.8),('lk','la',5.8,3.7),('rh','rk',8.2,5.8),('rk','ra',5.8,3.7)]:segment(a,b,r1,r2)
    s.path(f'M{cx-4*scale},{top+8*scale} L{cx+4*scale},{top+8*scale} L{cx+5*scale},{top+24*scale} L{cx-5*scale},{top+24*scale} Z',color,.5,color)
    s.circle(*p['head'],9*scale,color)
    for key in ['la','ra']:
        xx,yy=p[key];s.path(f'M{xx-4*scale},{yy-2*scale} L{xx+3*scale},{yy-2*scale} L{xx+10*scale},{yy+5*scale} L{xx-5*scale},{yy+5*scale} Z',color,.5,color)
    for key in ['lw','rw']:
        xx,yy=p[key];s.path(f'M{xx-3.2*scale},{yy-2*scale} L{xx+3.2*scale},{yy-2*scale} L{xx+5*scale},{yy+9*scale} L{xx-4*scale},{yy+10*scale} Z',color,.5,color)
    if joints:
        for k in pts:
            if k!='head': s.circle(*p[k],3.1*scale,'#A18C69','white',.7)
    return p


def palm(s,x,y,scale=1,color=TEAL):
    s.group(f'palm-{int(x)}-{int(y)}')
    s.path(f'M{x},{y+42*scale} L{x-7*scale},{y+22*scale} Q{x-8*scale},{y+17*scale} {x-3*scale},{y+17*scale} L{x+5*scale},{y+30*scale} L{x+5*scale},{y+3*scale} Q{x+8*scale},{y-2*scale} {x+11*scale},{y+3*scale} L{x+11*scale},{y+25*scale} L{x+15*scale},{y-1*scale} Q{x+18*scale},{y-6*scale} {x+21*scale},{y-1*scale} L{x+21*scale},{y+25*scale} L{x+26*scale},{y+1*scale} Q{x+29*scale},{y-4*scale} {x+32*scale},{y+2*scale} L{x+31*scale},{y+28*scale} L{x+38*scale},{y+10*scale} Q{x+42*scale},{y+7*scale} {x+44*scale},{y+12*scale} L{x+38*scale},{y+42*scale} Q{x+18*scale},{y+59*scale} {x},{y+42*scale} Z',color,1,color)
    for a,b in [(7,34),(20,36),(31,32)]: s.circle(x+a*scale,y+b*scale,2.4*scale,'#A18C69','white',.7)
    s.end()


def limb(s,x,y,straight=False,color=TEAL,scale=1):
    pts=[(0,45),(18,12),(52,0)] if straight else [(0,45),(17,11),(44,33)]
    for (a,b),(c,d) in zip(pts,pts[1:]): s.line(x+a*scale,y+b*scale,x+c*scale,y+d*scale,color,6*scale)
    for a,b in pts: s.circle(x+a*scale,y+b*scale,4*scale,color)


def bullets(s,x,y,items,size=18,color=INK):
    for i,item in enumerate(items):
        s.circle(x,y-5+i*size*1.5,2.3,color)
        s.text(x+13,y+i*size*1.5,item,size,color=color)


def annotation():
    s=SVG(1920,1040)
    cols=[16+316*i for i in range(6)]
    titles=[['Dance Clips'],['3D Motion','Extraction'],['Temporal Event','Extraction'],['Motion Evidence','Consolidation'],['Evidence-Grounded','Refinement'],['Paired Annotation','Dataset']]
    colors=[TEAL,BLUE,BLUE,PURPLE,PURPLE,GOLD]
    fills=[LIGHT_TEAL,LIGHT_BLUE,LIGHT_BLUE,LIGHT_PURPLE,LIGHT_PURPLE,LIGHT_GOLD]
    for i,(x,title,color,fill) in enumerate(zip(cols,titles,colors,fills)):
        s.group(f'stage-{i+1}')
        s.box(x,16,300,444,'white',color,20)
        s.box(x+1,17,298,74,fill,'none',18)
        s.badge(x+33,52,i+1,color)
        s.text(x+182,46 if len(title)>1 else 61,title,23,'bold',INK,'middle')
        s.end()
        if i<5: s.line(x+301,245,x+314,245,INK,2.5,True)

    x=cols[0];s.group('source-media')
    s.box(x+14,121,272,137,'#242A33','#242A33',7)
    for row in [127,245]:
        for j in range(14): s.box(x+21+18*j,row,10,7,'white','none',1)
    for j,p in enumerate([2,1,3]):
        s.box(x+23+85*j,142,78,94,'#FBFCFA','none',0)
        body(s,x+61+85*j,155,.46,p,False)
    wave(s,x+21,310,257,68,seed=2)
    s.box(x+12,367,276,79,'#FDFBFA','none',12)
    bullets(s,x+30,395,['Video frames','Paired music'],20)
    s.end()

    x=cols[1];s.group('pose-and-detail')
    p=body(s,x+82,118,1.25)
    s.line(x+183,156,x+183,119,BLUE,2,True);s.text(x+188,116,'z',15,'bold',BLUE)
    s.line(x+183,156,x+215,156,'#A18C69',2,True);s.text(x+220,161,'x',15,'bold','#A18C69')
    s.line(x+183,156,x+168,170,'#667A70',2,True);s.text(x+159,184,'y',15,'bold','#667A70')
    s.box(x+173,184,112,73,'white',INK,10)
    palm(s,x+202,193,.75)
    s.text(x+229,249,'Hand ROI',16,'bold',anchor='middle')
    s.box(x+173,273,112,78,'white',INK,10)
    s.path(f'M{x+202},286 L{x+233},300 L{x+228},325',TEAL,9)
    for a,b in [(202,286),(233,300),(228,325)]: s.circle(x+a,b,3.5,'#A18C69','white',1)
    s.text(x+229,344,'Leg detail',16,'bold',anchor='middle')
    s.line(*p['rw'],x+174,218,INK,1.5,dash='4 4')
    s.line(*p['rk'],x+174,309,INK,1.5,dash='4 4')
    s.text(x+228,367,'Optional RGB landmarks',13,color='#65778A',anchor='middle',italic=True)
    s.box(x+12,379,276,68,'#FDFBFA','none',10)
    bullets(s,x+24,399,['NLF body pose','Support-aware trajectories','Normalize + resample'],15.5)
    s.end()

    x=cols[2];s.group('temporal-event-extraction')
    for j in range(3): s.box(x+12+j*94,116,88,129,'white',BLUE,11)
    limb(s,x+26,141,False,scale=.9);s.path(f'M{x+53},151 A35,35 0 0 1 {x+60},187',GOLD,2)
    s.text(x+56,229,'Angle',17,'bold',anchor='middle')
    palm(s,x+119,136,.46);palm(s,x+156,136,.46)
    s.line(x+137,172,x+152,172,PURPLE,2,True)
    s.text(x+150,229,'Distance',17,'bold',anchor='middle')
    body(s,x+244,133,.42,joints=False);s.path(f'M{x+213},177 C{x+259},197 {x+280},175 {x+263},166',GOLD,2,arrow=True)
    s.text(x+244,229,'Orientation',15,'bold',anchor='middle')
    s.line(x+20,314,x+280,314,INK,1.5,True);s.line(x+20,314,x+20,258,INK,1.5,True)
    d='M'+str(x+22)+',302 '
    for k in range(65):d+=f'L{x+22+k*3.92:.2f},{288-12*math.sin(k*.24)-6*math.sin(k*.62):.2f} '
    s.path(d,BLUE,2.4)
    for j,c in enumerate([PURPLE,'#667A70','#A18C69']):s.box(x+22+j*83,322,79,13,c,'none',0)
    s.text(x+282,339,'t',16,italic=True)
    s.chip(x+14,359,124,33,'Pose attributes',LIGHT_BLUE,BLUE,15)
    s.line(x+143,376,x+158,376,INK,1.4,True)
    s.chip(x+165,359,122,33,'Temporal change','#FBFCFA','#667A70',14)
    s.chip(x+14,407,124,33,'Temporal change',LIGHT_BLUE,BLUE,14)
    s.line(x+143,423,x+158,423,INK,1.4,True)
    s.chip(x+165,407,122,33,'Motion events',LIGHT_GOLD,GOLD,15)
    s.end()

    x=cols[3];s.group('evidence-consolidation-top')
    s.box(x+13,116,274,100,'white',PURPLE,11)
    for j,b in enumerate([False,True,False,True]):
        limb(s,x+23+j*65,133,b,scale=.75)
        if j<3:s.line(x+67+j*65,164,x+81+j*65,164,INK,1.4,True)
    s.path(f'M{x+28},210 Q{x+26},222 {x+48},222 L{x+259},222 Q{x+279},222 {x+277},210',PURPLE,2)
    s.chip(x+26,236,249,37,'Repeated alternation',LIGHT_PURPLE,PURPLE,18)
    for j,(t,c,f) in enumerate([('bend',TEAL,LIGHT_TEAL),('extend',GOLD,LIGHT_GOLD),('bend',TEAL,LIGHT_TEAL)]):s.chip(x+15+89*j,300,82,35,t,f,c,18)
    s.box(x+12,365,276,82,'#FDFBFA','none',12)
    bullets(s,x+24,387,['Validate event evidence','Merge overlaps','Summarize alternations'],17)
    s.end()

    x=cols[4];s.group('refinement-top')
    s.box(x+86,112,128,61,'white',INK,9)
    for j,c in enumerate([TEAL,GOLD,PURPLE]):
        s.circle(x+101,128+15*j,4,c);s.line(x+114,128+15*j,x+197-j*14,128+15*j,c,4)
    s.line(x+150,174,x+150,194,INK,1.7,True)
    s.box(x+26,204,248,54,'#FBFCFA',INK,10)
    s.text(x+150,239,'GPT-6.1-sol',25,'bold',anchor='middle')
    s.line(x+150,259,x+150,279,INK,1.7,True)
    s.box(x+62,290,176,57,LIGHT_PURPLE,PURPLE,10)
    for j in range(3):s.line(x+79,306+j*13,x+222-j*9,306+j*13,'#85778C',4)
    s.box(x+12,367,276,80,'#FDFBFA','none',12)
    bullets(s,x+25,393,['Ground motion wording','Preserve source links'],19)
    s.end()

    x=cols[5];s.group('paired-dataset')
    for j in range(3):
        yy=117+j*83;s.box(x+13,yy,274,69,'white',BLUE,10)
        wave(s,x+21,yy+34,78,42,seed=j+3)
        for k,c in enumerate([TEAL,'#A18C69',PURPLE]):
            s.line(x+113,yy+20+k*13,x+239-k*21,yy+20+k*13,c,5)
        s.text(x+266,yy+29,'↗',27,'bold',anchor='middle')
        s.box(x+248,yy+44,24,11,LIGHT_GOLD,GOLD,4)
    s.box(x+12,379,276,68,'#FDFBFA','none',12)
    bullets(s,x+24,400,['Music + movement captions','Source links + quality flags'],16.5)
    s.end()

    s.group('audio-bypass')
    s.path('M166,461 L166,480 Q166,497 184,497 L1746,497 Q1764,497 1764,480 L1764,463',TEAL,3,arrow=True)
    s.chip(735,475,450,43,'Preserve aligned audio',LIGHT_TEAL,TEAL,24)
    s.end()

    # Expanded stages, matched by their badges without crossing connectors.
    s.group('expanded-consolidation')
    s.box(16,548,937,469,'white',PURPLE,18)
    s.box(17,549,935,52,LIGHT_PURPLE,'none',17)
    s.badge(47,575,4,PURPLE,21);s.text(80,584,'Motion evidence consolidation',27,'bold')
    cards=[('Validate','evidence'),('Deduplicate','overlaps'),('Summarize','alternations'),('Preserve','provenance')]
    for j,t in enumerate(cards):
        xx=35+j*229;s.box(xx,618,211,59,LIGHT_PURPLE,PURPLE,10)
        s.text(xx+105,642,list(t),18,'bold',anchor='middle')
        if j<3:s.line(xx+213,648,xx+226,648,PURPLE,1.6,True)
    s.box(34,698,708,248,'#FAFBFD','#B7BFC7',12)
    s.box(35,699,175,246,'#FCFAFD','none',10)
    s.text(48,729,['Temporal','evidence ledger'],20,'bold')
    s.text(48,846,['Event (primary)','Event (overlap)'],17,'bold')
    s.text(48,914,['Source occurrences','(provenance)'],16,'bold')
    for j,p in enumerate([2,1,2,1]):
        body(s,271+117*j,715,.56,p)
        if j<3:s.line(307+117*j,767,328+117*j,767,INK,1.4,True)
    s.line(225,814,727,814,INK,1.5,True)
    for j,t in enumerate(['bend','extend','bend','extend']):
        xx=228+124*j
        s.chip(xx,824,111,30,t,LIGHT_TEAL if j%2==0 else LIGHT_GOLD,TEAL if j%2==0 else GOLD,17)
        s.line(xx+55,857,xx+55,933,PURPLE,1.1,dash='3 4')
        s.circle(xx+55,920,13,'white',PURPLE)
        s.text(xx+55,925,'o'+str(j+1),14,anchor='middle')
    s.box(228,861,111,26,LIGHT_PURPLE,PURPLE,8,dash='4 3');s.text(283,879,'bend (overlap)',12,anchor='middle',italic=True)
    s.box(762,698,173,248,'#FCFAFD','none',11)
    limb(s,777,721,False,scale=.9);limb(s,873,721,True,scale=.9)
    s.line(830,753,851,753,INK,1.4,True)
    s.text(849,810,['Repeated','alternation'],19,'bold',PURPLE,'middle')
    s.chip(774,839,149,76,['Repeated bending','and extending'] if False else 'Bend ↔ extend',LIGHT_GOLD,GOLD,16)
    s.chip(66,964,320,35,'Timing retained',LIGHT_BLUE,BLUE,19)
    s.chip(425,964,469,35,'Source occurrences retained',LIGHT_BLUE,BLUE,19)
    s.end()

    s.group('expanded-refinement')
    s.box(973,548,931,469,'white',GOLD,18)
    s.box(974,549,929,52,LIGHT_GOLD,'none',17)
    s.badge(1004,575,5,PURPLE,21);s.text(1038,584,'Evidence-grounded refinement',27,'bold')
    s.box(993,621,889,66,'white',BLUE,12)
    # Each phrase is an independent text box for easy recoloring and wording edits.
    words=[('One arm',124,TEAL,LIGHT_TEAL),('rises',75,GOLD,LIGHT_GOLD),('while',71,PURPLE,LIGHT_PURPLE),('the other',132,TEAL,LIGHT_TEAL),('lowers;',92,GOLD,LIGHT_GOLD),('both elbows',160,INK,'white'),('then',65,PURPLE,LIGHT_PURPLE),('extend.',100,GOLD,LIGHT_GOLD)]
    xx=1005
    for word,ww,c,f in words:
        s.box(xx,638,ww,32,f,'none',5);s.text(xx+ww/2,661,word,19,'bold',c,'middle');xx+=ww+2
    for center,label,ww in [(1125,'arm rise',180),(1390,'arm lower',200),(1686,'elbow extension',263)]:
        s.line(center,687,center,720,TEAL,1.6)
        s.circle(center,687,3,TEAL);s.circle(center,720,3,TEAL)
        s.chip(center-ww/2,730,ww,38,label,LIGHT_TEAL,TEAL,20)
    s.chip(1605,779,277,27,'Revise unsupported claims',LIGHT_PURPLE,PURPLE,15)
    s.path('M1882,793 L1893,793 L1893,655 L1884,655',PURPLE,1.7,arrow=True)
    s.chip(1127,814,617,31,'Refinement objectives',LIGHT_PURPLE,PURPLE,21)
    checks=[('Source coverage',1000,856),('Body + direction fidelity',1452,856),('Temporal + repetition consistency',1000,914),('Concise, grounded prose',1452,914)]
    for label,xx,yy in checks:
        s.box(xx,yy,430,44,LIGHT_GOLD,GOLD,10)
        s.circle(xx+22,yy+22,12,GOLD);s.text(xx+22,yy+28,'✓',19,'bold','white','middle')
        s.text(xx+43,yy+29,label,18,'bold')
    s.box(995,976,887,28,'#FAFBFD','none',9)
    s.text(1438,996,'Automatic checks; semantic validation is separate',18,anchor='middle',italic=True)
    s.end()
    return s.finish('annotation_pipeline.svg')



if __name__=='__main__':
    print(annotation())
