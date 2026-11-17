"""Rich native-vector harness reference layout. No generated bitmap backing."""
import math

BLUE='#2474B5'; TEAL='#008F95'; PURPLE='#7852A9'; ORANGE='#E69A28'; CORAL='#D45D64'; INK='#202C3C'; EDGE='#B8C8D6'
PALE='#FAFCFD'; LILAC='#FBFAFE'; CREAM='#FFFCF7'; SKIN='#F3C3A2'; NAVY='#315C86'

def ellipse(f,cx,cy,rx,ry,fill='white',stroke=BLUE,sw=1.6):
    f.parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    f.scene.append(dict(type='ellipse',cx=cx,cy=cy,rx=rx,ry=ry,fill=fill,stroke=stroke,sw=sw))

def document(f,x,y,w=36,h=46,col=BLUE,fill='white'):
    f.path(f'M{x},{y} H{x+w-10} L{x+w},{y+10} V{y+h} H{x} Z',col,1.9,fill)
    f.path(f'M{x+w-10},{y} V{y+10} H{x+w}',col,1.5)
    for k in range(3):f.line(x+7,y+19+k*8,x+w-7-(k==2)*7,y+19+k*8,col,1.6)

def database(f,x,y,w,h,col=BLUE,fill='white'):
    f.rect(x,y+10,w,h-20,fill,'none')
    f.line(x,y+10,x,y+h-10,col,1.9);f.line(x+w,y+10,x+w,y+h-10,col,1.9)
    ellipse(f,x+w/2,y+h-10,w/2,10,fill,col,1.8)
    ellipse(f,x+w/2,y+10,w/2,10,fill,col,1.8)
    if h>65:f.path(f'M{x},{y+h*.58} Q{x+w/2},{y+h*.58+19} {x+w},{y+h*.58}',col,1.2)

def gear(f,cx,cy,r=20,col=TEAL):
    points=[]
    for i in range(32):
        angle=i*math.pi/16;rr=r*(1 if i%4 in (0,1) else .76)
        points.append((cx+rr*math.cos(angle),cy+rr*math.sin(angle)))
    f.path('M'+' L'.join(f'{x},{y}' for x,y in points)+' Z',col,1,col)
    f.circle(cx,cy,r*.36,'white',col,1.1)

def lock(f,x,y,s=25,col=BLUE):
    f.path(f'M{x+s*.2},{y+s*.47} V{y+s*.23} A{s*.3},{s*.3} 0 0 1 {x+s*.8},{y+s*.23} V{y+s*.47}',col,2)
    f.rect(x,y+s*.4,s,s*.68,'white',col,1.8,4)
    f.circle(x+s*.5,y+s*.72,s*.06,col)
    f.line(x+s*.5,y+s*.73,x+s*.5,y+s*.9,col,1.5)

def pencil(f,x,y,s=30,col=ORANGE):
    f.path(f'M{x},{y+s} L{x+4},{y+s-12} L{x+s-7},{y+2} L{x+s+1},{y+10} L{x+11},{y+s-3} Z',col,1.9,'white')
    f.line(x+4,y+s-12,x+11,y+s-3,col,1.8)
    f.path(f'M{x},{y+s} l3,-8 l5,5 Z',col,1,col)

def calendar(f,x,y,s=34,col=BLUE):
    f.rect(x,y+5,s,s-4,'white',col,1.8,3)
    f.line(x,y+15,x+s,y+15,col,1.5)
    for i in [8,s-8]:f.line(x+i,y,x+i,y+10,col,2.8)
    for j in range(2):
        for i in range(3):f.rect(x+6+i*9,y+21+j*9,4,4,col,'none')

def tag(f,x,y,w=39,h=29,col=TEAL):
    f.path(f'M{x},{y+9} L{x+12},{y} H{x+w} V{y+h-9} L{x+w-12},{y+h} H{x} Z',col,1.9,'white')
    f.circle(x+10,y+10,2.7,'white',col,1.5)

def waveform_icon(f,x,y,w=37,h=32,col=BLUE):
    for k,z in enumerate([.25,.55,.95,.55,.35]):f.line(x+k*w/5,y-z*h/2,x+k*w/5,y+z*h/2,col,2.4)

def memory_layers(f,x,y,w=35,h=31,col=PURPLE):
    for dy in [14,7,0]:f.path(f'M{x},{y+dy+8} L{x+w/2},{y+dy} L{x+w},{y+dy+8} L{x+w/2},{y+dy+16} Z',col,1.5,'white')

def ledger(f,x,y,w=37,h=33,col=BLUE):
    f.line(x,y+h,x+w,y+h,col,1.5)
    for i,z in enumerate([12,21,30]):f.rect(x+4+i*11,y+h-z,6,z,col,'none',r=1)

def check(f,x,y,col=TEAL):f.path(f'M{x},{y} l7,7 l15,-18',col,2.6)

def flat_dancer(f,x,y,s=1,pose=0):
    """A clothed human icon built from editable filled paths and strokes."""
    ps=[{'head':(0,-60),'shL':(-14,-38),'shR':(15,-37),'elL':(4,-27),'elR':(37,-27),'wrL':(33,-24),'wrR':(50,-7),'hipL':(-10,8),'hipR':(14,8),'knL':(-39,28),'knR':(43,27),'anL':(-48,63),'anR':(48,63)},
        {'head':(0,-60),'shL':(-14,-38),'shR':(15,-37),'elL':(-30,-64),'elR':(32,-63),'wrL':(-10,-91),'wrR':(11,-89),'hipL':(-10,8),'hipR':(14,8),'knL':(-26,43),'knR':(30,42),'anL':(-38,78),'anR':(45,77)},
        {'head':(-2,-60),'shL':(-16,-38),'shR':(14,-37),'elL':(-36,-8),'elR':(39,-37),'wrL':(-22,13),'wrR':(62,-54),'hipL':(-10,8),'hipR':(14,8),'knL':(-33,36),'knR':(34,35),'anL':(-48,70),'anR':(54,66)},
        {'head':(0,-60),'shL':(-14,-38),'shR':(15,-37),'elL':(-43,-36),'elR':(45,-36),'wrL':(-70,-37),'wrR':(72,-39),'hipL':(-14,8),'hipR':(17,8),'knL':(-45,23),'knR':(45,23),'anL':(-58,58),'anR':(60,58)}][pose%4]
    def p(k):a,b=ps[k];return x+a*s,y+b*s
    def poly(points,fill,stroke='none',sw=0):f.path('M'+' L'.join(f'{x+a*s},{y+b*s}' for a,b in points)+' Z',stroke,sw,fill)
    # Bent trousers and distinct hip/knee structure, drawn before the upper body.
    for suffix,color in [('L',NAVY),('R',BLUE)]:
        h,k,a=p('hip'+suffix),p('kn'+suffix),p('an'+suffix)
        f.path(f'M{h[0]},{h[1]} L{k[0]},{k[1]} L{a[0]},{a[1]}',color,13*s)
        ax,ay=ps['an'+suffix];poly([(ax-6,ay-2),(ax+5,ay-1),(ax+14,ay+5),(ax-8,ay+5)],INK)
    poly([(-12,-1),(16,-1),(20,15),(-12,16)],NAVY)
    # Skin forearms and small hands are separate native shapes.
    for suffix in ['L','R']:
        a,b,c=p('sh'+suffix),p('el'+suffix),p('wr'+suffix)
        f.path(f'M{a[0]},{a[1]} L{b[0]},{b[1]} L{c[0]},{c[1]}',SKIN,8*s)
        f.circle(*c,4.3*s,SKIN)
    poly([(-16,-40),(16,-39),(18,-9),(14,2),(-12,3),(-18,-13)],BLUE)
    poly([(-14,-37),(-5,-36),(-7,-3),(-12,3),(-18,-13)],'#1D659E')
    for side in ['L','R']:
        a,b=p('sh'+side),p('el'+side)
        f.line(*a,a[0]+(b[0]-a[0])*.37,a[1]+(b[1]-a[1])*.37,BLUE,12*s)
    f.line(x,y-48*s,x,y-39*s,SKIN,7*s)
    hx,hy=p('head');f.circle(hx-3*s,hy-3*s,11*s,INK)
    f.circle(hx+1*s,hy+2*s,9*s,SKIN)
    f.path(f'M{hx-9*s},{hy+2*s} Q{hx-7*s},{hy-13*s} {hx+10*s},{hy-7*s} Q{hx+4*s},{hy-1*s} {hx-9*s},{hy+2*s} Z',INK,0,INK)
    f.circle(hx-11*s,hy-7*s,5*s,INK)
    f.circle(hx+7*s,hy+1*s,.8*s,INK)

def build(Fig):
    f=Fig('Reference-inspired native harness overview',2400,1180)
    # Source audio fans out to the trained model and independent specialist tools.
    with f.group('source-music'):
        f.rect(25,20,2350,110,PALE,EDGE,r=10)
        waveform_icon(f,52,68,36,42,BLUE)
        f.text(110,73,'Music',40,True)
        f.wave(245,56,2095,46,BLUE,n=245)
        for xx,label in [(290,'Bars 1–2'),(985,'Bars 3–4'),(1680,'Bars 5–6')]:
            f.line(xx,90,xx+570,90,BLUE,1.2)
            f.line(xx,84,xx,96,BLUE,1.2);f.line(xx+570,84,xx+570,96,BLUE,1.2)
            f.text(xx+285,119,label,24,True,BLUE,'middle')
        f.line(25,130,10,130,BLUE,1.6)
        f.line(10,130,10,631,BLUE,1.6)
    with f.group('trained-local-expert'):
        f.rect(25,151,560,957,LILAC,EDGE,r=12)
        f.line(25,151,585,151,PURPLE,3.2)
        document(f,46,169,30,38,PURPLE)
        f.text(93,200,'Our trained choreography expert',35,True,PURPLE)
        f.boxlabel(45,244,250,55,'Audio front end',32,col=BLUE)
        f.arrow([(10,271),(40,271)],BLUE)
        f.arrow([(170,303),(170,334)],BLUE)
        f.rect(45,339,250,141,'white',PURPLE,1.7,r=8)
        f.text(170,373,'Qwen3-Omni',34,True,PURPLE,'middle')
        f.text(170,411,'+ LoRA',32,True,PURPLE,'middle')
        for j in range(2):
            yy=433+j*25;f.rect(61,yy,156,17,'white',BLUE,1)
            f.line(70,yy+8,205,yy+8,BLUE,1)
            f.rect(245,yy,34,17,'white',PURPLE,1)
            f.text(262,yy+13,'ΔW',17,False,PURPLE,'middle')
            f.arrow([(240,yy+8),(221,yy+8)],PURPLE,1.1)
        f.rect(316,244,249,236,'white',PURPLE,1.5,r=8)
        f.lines(330,279,['Joint-aware','clause weights'],32,38,True,PURPLE)
        f.rect(326,347,140,76,'#F6F0FA','none',r=4)
        f.lines(330,378,['bend the','right knee'],31,37,True,PURPLE)
        f.skeleton(515,373,.48,joints=True,legs=True,col=BLUE)
        f.text(332,456,'w = 1.5',34,True,ORANGE)
        f.text(551,458,'clause CE',24,False,BLUE,'end',True)
        f.arrow([(170,484),(170,506)],PURPLE)
        f.rect(45,512,520,57,'white',PURPLE,1.7,r=7)
        document(f,60,525,26,32,PURPLE)
        f.text(317,550,'Local movement proposals',34,True,PURPLE,'middle')
        f.text(45,615,'Shared musical analysis',32,True,BLUE)
        for xx,title,desc,col,icon in [(45,'Beat This!','2–4 bars',BLUE,'calendar'),(224,'JEV','Global style',TEAL,'tag'),(403,'librosa','Acoustic cues',CORAL,'wave')]:
            f.rect(xx,644,162,126,'white',col,1.6,r=7)
            if icon=='calendar':calendar(f,xx+12,657,28,col)
            elif icon=='tag':tag(f,xx+10,657,31,24,col)
            else:waveform_icon(f,xx+13,670,27,30,col)
            f.text(xx+81,721,title,31,True,col,'middle')
            f.text(xx+81,758,desc,27,anchor='middle')
        # The raw-audio bus stays outside the expert and never enters the harness.
        f.line(10,631,484,631,BLUE,1.6)
        for xx in [126,305,484]:f.arrow([(xx,631),(xx,639)],BLUE)
        for xx in [126,305,484]:f.arrow([(xx,774),(xx,799)],BLUE)
        f.line(126,799,484,799,BLUE,1.7)
        f.arrow([(305,799),(305,827)],BLUE)
        # Cache contains text slips, never dancer illustrations.
        database(f,45,833,520,115,BLUE,PALE)
        f.text(305,881,'Frozen evidence + candidate cache',31,True,BLUE,'middle')
        for k,label in enumerate(['“step right”','“turn left”','“widen sweep”']):
            xx=58+k*167;f.rect(xx,897,155,32,'white',EDGE,1,r=3)
            f.text(xx+77.5,919,label,24,anchor='middle')
        # Model proposals are cached independently of acoustic analysis.
        f.arrow([(569,540),(575,540),(575,815),(539,815),(539,832)],PURPLE)
        f.text(305,1024,'Audio-conditioned movement content',28,False,PURPLE,'middle',True)
        f.text(305,983,'Read-only evidence',26,True,BLUE,'middle')
        f.text(305,1071,'Joint identity + action + relation',28,False,BLUE,'middle')
    with f.group('stateful-compositional-harness'):
        f.rect(625,151,1110,957,PALE,TEAL,3.3,r=13)
        gear(f,658,184,18,TEAL)
        f.text(696,201,'Stateful compositional harness',40,True,TEAL)
        # The task brief is contained in the harness boundary.
        f.rect(665,241,1030,67,CREAM,EDGE,1.2,r=7)
        document(f,680,257,26,34,ORANGE)
        f.text(723,278,'Choreographic brief',32,True,ORANGE)
        f.text(1080,278,'Goals · conditions · editable scope',30)
        # Global objectives illustrate inspectable agent state, not a learned critic.
        f.text(665,365,'Compose a whole-song outline',32,True,BLUE)
        for k,(xx,lab,col) in enumerate([(1180,'A',PURPLE),(1390,'B',TEAL),(1600,'A′',ORANGE)]):
            f.circle(xx,355,22,'white',col,1.8);f.text(xx,365,lab,32,True,col,'middle')
            if k<2:f.arrow([(xx+28,355),(xx+181,355)],BLUE)
        # Learned composer and deterministic control/execution are visibly distinct.
        f.rect(665,429,280,245,'white',PURPLE,1.9,r=9)
        document(f,684,447,28,36,PURPLE)
        f.text(728,478,'Composer',35,True,PURPLE)
        f.text(805,536,'Qwen3-VL',36,True,anchor='middle')
        f.lines(805,578,['Outline + motifs','Unit drafts + edits'],31,39,anchor='middle')
        lock(f,707,638,18,PURPLE)
        f.text(837,660,'Fixed at inference',23,False,PURPLE,'middle')
        f.rect(1065,429,270,245,'white',TEAL,1.9,r=9)
        gear(f,1091,462,18,TEAL)
        f.text(1125,478,'Controller',35,True,TEAL)
        f.lines(1200,535,['Schedule stages','Budget / permissions','Continue / stop'],30,47,anchor='middle')
        f.rect(1440,429,255,245,'white',BLUE,1.9,r=9)
        memory_layers(f,1457,447,29,24,BLUE)
        f.text(1500,478,'Tools',35,True,BLUE)
        ops=[('Read evidence','doc'),('Validate plan','check'),('Commit unit','lock'),('Revise text','pencil')]
        for k,(label,icon) in enumerate(ops):
            yy=523+k*42
            if icon=='doc':document(f,1457,yy-20,19,25,BLUE)
            elif icon=='check':check(f,1458,yy-7,TEAL)
            elif icon=='lock':lock(f,1457,yy-23,20,BLUE)
            else:pencil(f,1458,yy-23,21,ORANGE)
            f.text(1487,yy,label,29)
        f.arrow([(1060,529),(950,529)],PURPLE)
        f.lines(1005,482,['Context +','feedback'],26,30,col=PURPLE,anchor='middle')
        f.arrow([(950,615),(1060,615)],PURPLE)
        f.text(1005,650,'Proposal',28,False,PURPLE,'middle')
        f.arrow([(1340,529),(1435,529)],BLUE)
        f.text(1387,502,'Execute',26,False,BLUE,'middle')
        f.arrow([(1435,615),(1340,615)],BLUE)
        f.lines(1387,650,['Checks /','errors'],26,30,col=BLUE,anchor='middle')
        # A single labelled cache edge enters composer context, not persistent memory.
        f.arrow([(565,892),(603,892),(603,596),(660,596)],BLUE)
        f.circle(625,596,3.2,BLUE)
        f.text(665,729,'Persistent state',36,True,BLUE)
        f.rect(665,757,1030,147,'white',BLUE,1.6,r=8)
        cells=[(['Outline +','motifs'],'layers'),(['Current','draft'],'doc'),(['Committed','history'],'db'),(['Call / token','ledger'],'ledger')]
        for k,(labels,icon) in enumerate(cells):
            xx=665+k*257.5
            if k:f.line(xx,768,xx,892,EDGE)
            cx=xx+128.75
            if icon=='layers':memory_layers(f,cx-19,774,38,30,PURPLE)
            elif icon=='doc':document(f,cx-14,772,28,37,BLUE)
            elif icon=='db':database(f,cx-20,774,40,38,TEAL)
            else:ledger(f,cx-20,774,40,36,BLUE)
            f.lines(cx,850,labels,32,38,True,anchor='middle')
        f.arrow([(1200,680),(1200,752)],TEAL,both=True)
        f.arrow([(1567,680),(1567,752)],BLUE,both=True)
        f.text(665,955,'Targeted revision',36,True,ORANGE)
        f.text(1695,955,'Keep time · track · motif fixed',28,False,BLUE,'end')
        for k,bar in enumerate(['Bars 1–2','Bars 3–4','Bars 5–6','Bars 7–8','Bars 9–10']):
            xx=665+k*210
            f.rect(xx,985,190,100,CREAM if k==2 else PALE,ORANGE if k==2 else BLUE,2.7 if k==2 else 1.2,r=7)
            f.text(xx+95,1019,bar,30,True,anchor='middle')
            if k==2:pencil(f,xx+84,1029,24,ORANGE)
            else:lock(f,xx+83,1030,24,BLUE)
            f.text(xx+95,1073,'Revise text' if k==2 else 'Protected',26,True,ORANGE if k==2 else BLUE,'middle')
    with f.group('hierarchical-choreography'):
        f.rect(1775,151,600,697,PALE,EDGE,1.6,r=12)
        f.line(1775,151,2375,151,BLUE,3)
        document(f,1793,170,28,35,BLUE)
        f.text(1839,201,'Hierarchical choreography',35,True,BLUE)
        f.text(1800,263,'Dance style is jazz.',32,False,TEAL)
        for k,(xx,lab,role,col) in enumerate([(1850,'A','Introduce',PURPLE),(2080,'B','Contrast',TEAL),(2310,'A′','Resolve',ORANGE)]):
            f.circle(xx,326,25,'white',col,1.9);f.text(xx,336,lab,34,True,col,'middle')
            f.text(xx,385,role,30,anchor='middle')
            if k<2:f.arrow([(xx+31,326),(xx+199,326)],BLUE)
        f.line(1795,415,2355,415,EDGE)
        rows=[(463,'Bars 1–2 · A',['Step right; sweep the left arm','across the chest and settle low.'],0,PURPLE),
              (650,'Bars 9–10 · A′',['Return to A with a wider sweep','and a lower stance.'],3,ORANGE)]
        for yy,lab,lines,pose,col in rows:
            f.text(1798,yy,lab,32,True,col)
            flat_dancer(f,1849,yy+103,.76,pose)
            f.lines(1913,yy+55,lines,30,39)
        f.text(2355,829,'Schematic score and pose examples',26,False,BLUE,'end',True)
        f.arrow([(1700,596),(1770,596)],BLUE)
        f.text(1735,572,'Score',28,False,BLUE,'middle')
    with f.group('independent-local-global-evaluation'):
        f.rect(1775,885,600,223,CREAM,EDGE,1.6,r=12)
        f.line(1775,885,2375,885,CORAL,3)
        document(f,1794,901,25,32,CORAL)
        f.text(1833,928,'Local and global evaluation',32,True,CORAL)
        f.rect(1790,944,570,38,'white',EDGE,1,r=4)
        waveform_icon(f,1805,963,25,21,BLUE)
        f.text(1845,971,'Music + choreography text',28,False,BLUE)
        f.wave(2240,963,105,20,BLUE,n=22)
        f.arrow([(2075,853),(2075,880)],CORAL)
        document(f,1796,1005,22,28,CORAL)
        f.text(1831,1034,'Local units',26,True,CORAL)
        f.line(1957,997,1957,1055,EDGE)
        f.text(1970,1012,'Q1–Q3: text quality',28,True,BLUE)
        f.text(1970,1047,'M1–M3: music compatibility',28,True,TEAL)
        f.line(1790,1066,2360,1066,EDGE)
        ledger(f,1796,1072,25,27,CORAL)
        f.text(1828,1094,'Whole song',26,True,CORAL)
        f.text(1970,1094,'G1–G3:',26,True,PURPLE)
        f.text(2070,1094,'continuity · structure · motifs',24,False,PURPLE)
    with f.group('legend'):
        f.text(40,1158,'Models',27,True,PURPLE)
        f.line(137,1150,182,1150,PURPLE,3)
        f.text(223,1158,'Control',27,True,TEAL);f.line(321,1150,365,1150,TEAL,3)
        f.text(406,1158,'Tools / state',27,True,BLUE);f.line(564,1150,608,1150,BLUE,3)
        f.text(649,1158,'Selected revision',27,True,ORANGE);f.line(870,1150,914,1150,ORANGE,3)
        f.text(2375,1158,'Evidence and state are retained; evaluation stays outside generation.',27,False,BLUE,'end',True)
    return f.save('overview_method.svg')
