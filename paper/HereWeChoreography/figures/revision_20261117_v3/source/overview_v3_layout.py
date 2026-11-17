"""Third overview composition: vivid scientific foregrounds, pale backgrounds.

Designed as vector masters, not a trace of generated illustrations. Imported by
build_overview_masters.py; only the three genuine source frames are raster data.
"""
from pathlib import Path
import base64

BLUE='#2474B5'; TEAL='#008F95'; PURPLE='#7852A9'; ORANGE='#E69A28'; CORAL='#D45D64'; INK='#202C3C'; EDGE='#B7C6D2'
WHITE='#FFFFFF'; ICE='#FAFCFD'; LILAC='#FBFAFE'; CREAM='#FFFCF7'

def image(f,path,x,y,w,h):
    data=base64.b64encode(Path(path).read_bytes()).decode('ascii')
    f.parts.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{data}"/>')
    f.scene.append(dict(type='image',x=x,y=y,w=w,h=h,source=str(path),scope='Unmodified real source-video still; illustration only'))

def banner(f,x,y,w,label,accent,fs=35):
    f.line(x,y,x+w,y,accent,3)
    f.text(x,y+43,label,fs,True,accent)

def check(f,x,y,col=TEAL):
    f.path(f'M{x},{y} l7,7 l15,-18',col,3)

def method_v3(Fig,root):
    f=Fig('From joint-aware local prediction to stateful long-form composition',2400,1150)
    with f.group('music-source'):
        f.text(35,60,'Music',40,True)
        f.wave(175,57,2190,76,BLUE,n=270)
        f.line(35,113,2365,113,EDGE,1.2)
    with f.group('trained-expert-architecture'):
        f.rect(30,174,440,926,LILAC,EDGE)
        banner(f,50,177,400,'Our trained local expert',PURPLE,35)
        f.boxlabel(60,246,380,58,'Audio front end',32,col=BLUE)
        f.arrow([(250,309),(250,338)],BLUE)
        f.rect(60,343,380,166,WHITE,BLUE,1.7)
        f.text(250,380,'Qwen3-Omni + LoRA',36,True,PURPLE,'middle')
        for k in range(3):
            y=399+k*33
            f.rect(80,y,238,23,ICE,BLUE,1)
            f.line(96,y+11.5,298,y+11.5,BLUE,1.5)
            f.rect(351,y,68,23,LILAC,PURPLE,1.4)
            f.text(385,y+18,'ΔW',24,True,PURPLE,'middle')
            f.arrow([(346,y+11.5),(323,y+11.5)],PURPLE,1.4)
        f.arrow([(250,514),(250,541)],BLUE)
        f.boxlabel(60,546,380,64,'Movement proposals',34,col=PURPLE)
        f.text(60,683,'Joint-aware clause weights',34,True,PURPLE)
        f.text(60,738,'“Raise the left arm;',32)
        f.rect(57,755,330,45,'#F3ECFA','none')
        f.text(60,787,'bend the right knee.”',32,True,PURPLE)
        f.line(60,801,384,801,PURPLE,2)
        f.skeleton(126,906,.62,joints=True,legs=True,col=BLUE)
        f.lines(213,874,['Action + relation','weighted together'],32,39,True,PURPLE)
        f.text(213,980,'w = 1.5',36,True,ORANGE)
        f.lines(60,1050,['Clause-level text supervision'],28,col=BLUE,italic=True)
    # Raw music goes to the model, independently of the analysis specialists.
    f.arrow([(35,113),(17,113),(17,275),(60,275)],BLUE)
    with f.group('shared-musical-analysis'):
        for x,w,title,desc,col in [(530,350,'Beat This!','Bar-aligned windows',BLUE),(920,300,'JEV','Global dance style',TEAL),(1260,390,'librosa','Energy · onset · chroma',CORAL)]:
            f.rect(x,174,w,112,ICE,EDGE)
            f.line(x,174,x+w,174,col,3)
            f.text(x+18,215,title,35,True,col)
            f.text(x+18,260,desc,32)
            f.arrow([(x+w/2,114),(x+w/2,169)],col)
            f.arrow([(x+w/2,291),(x+w/2,326)],col)
        f.line(705,326,1694,326,BLUE,2)
        f.circle(1070,326,3,BLUE);f.circle(1455,326,3,BLUE)
        f.arrow([(1694,326),(1694,578),(1645,578)],BLUE)
    with f.group('compositional-harness'):
        f.rect(510,402,1170,698,ICE,EDGE)
        banner(f,530,405,1130,'Stateful compositional harness',TEAL,38)
        # The brief is task state inside the harness, not another audio expert.
        f.line(545,479,1645,479,EDGE)
        f.text(545,519,'Choreographic brief',32,True,ORANGE)
        f.text(900,519,'Goals · constraints · editable scope',30)
        f.rect(545,546,1100,64,WHITE,BLUE,1.5)
        f.text(566,586,'Read-only cache',32,True,BLUE)
        f.line(840,558,840,598,EDGE)
        f.text(861,586,'Candidates · bar units · style · acoustic cues',32)
        f.arrow([(445,578),(540,578)],PURPLE)
        f.arrow([(1100,615),(1100,672)],BLUE)
        # A decision loop, rather than another chain of prose cards.
        f.rect(545,700,300,198,WHITE,PURPLE,1.8)
        f.text(695,740,'Composer agent',36,True,PURPLE,'middle')
        f.text(695,788,'Qwen3-VL',34,True,anchor='middle')
        f.lines(695,837,['Outline + motifs','Draft / revise units'],32,38,anchor='middle')
        f.circle(1100,790,118,WHITE,TEAL,2.8)
        f.text(1100,777,'Controller',36,True,TEAL,'middle')
        f.lines(1100,816,['Stage / scope','Budget / stop'],32,38,anchor='middle')
        f.rect(1360,700,285,198,WHITE,BLUE,1.8)
        f.text(1502,740,'Tool execution',34,True,BLUE,'middle')
        f.lines(1502,801,['Read / validate','Commit / revise'],32,43,anchor='middle')
        f.arrow([(999,745),(850,745)],PURPLE)
        f.text(925,722,'Context',28,False,PURPLE,'middle')
        f.arrow([(850,845),(1004,845)],PURPLE)
        f.text(925,880,'Proposal',28,False,PURPLE,'middle')
        f.arrow([(1201,745),(1355,745)],BLUE)
        f.text(1278,722,'Call',28,False,BLUE,'middle')
        f.arrow([(1355,845),(1195,845)],BLUE)
        f.lines(1278,880,['Checks /','observations'],26,30,col=BLUE,anchor='middle')
        f.text(545,946,'Persistent task memory',34,True,TEAL)
        f.arrow([(1100,905),(1100,965)],TEAL,both=True)
        f.rect(545,970,1100,103,WHITE,TEAL,1.5)
        for k,(a,b) in enumerate([('Brief / scope','Task conditions'),('Outline / motifs','Global intentions'),('Current draft','Mutable unit'),('Committed plan','Accepted history'),('Budget ledger','Calls / tokens')]):
            x=545+k*220
            if k:f.line(x,981,x,1063,EDGE)
            f.text(x+110,1010,a,32,True,TEAL,'middle')
            f.text(x+110,1050,b,28,anchor='middle')
    with f.group('hierarchical-score'):
        f.rect(1730,174,635,636,ICE,EDGE)
        banner(f,1750,177,595,'Hierarchical choreography score',BLUE,35)
        f.text(1750,276,'Dance style is jazz.',32,False,TEAL)
        for k,(x,lab,role,col) in enumerate([(1800,'A','Introduce',PURPLE),(2048,'B','Contrast',TEAL),(2295,'A′','Resolve',ORANGE)]):
            f.circle(x,339,27,WHITE,col,2)
            f.text(x,350,lab,34,True,col,'middle')
            f.text(x,399,role,30,anchor='middle')
            if k<2:f.arrow([(x+33,339),(x+214,339)],BLUE)
        f.line(1750,430,2345,430,EDGE)
        rows=[(472,'Bars 1–2 · A',['Step right; sweep the left arm','across the torso and settle low.'],0,PURPLE),
              (648,'Bars 9–10 · A′',['Widen the arm sweep; turn left','and finish with the arms low.'],2,ORANGE)]
        for y,lab,lines,pose,col in rows:
            f.text(1750,y,lab,32,True,col)
            f.skeleton(1794,y+86,.57,pose=pose,col=BLUE,joints=False)
            f.lines(1865,y+51,lines,32,39)
        f.text(2345,795,'Illustrative score excerpts',26,False,BLUE,'end',True)
        f.arrow([(1650,793),(1725,793)],BLUE)
    with f.group('scoped-targeted-revision'):
        # Revision is a detail of the current tool operation; the scaffold is fixed.
        f.rect(30,1120,2335,1,WHITE,'none')
        f.text(545,662,'Targeted revision',32,True,ORANGE)
        labels=[('A','keep',PURPLE),('B','edit',ORANGE),('A′','keep',PURPLE)]
        for k,(lab,verb,col) in enumerate(labels):
            x=805+k*100
            f.line(x,641,x+70,641,col,5 if verb=='edit' else 2)
            f.text(x+35,680,f'{verb.title()} {lab}',26,False,col,'middle')
        f.text(1200,662,'time / track / motif fixed',28,False,BLUE)
    with f.group('independent-local-global-evaluation'):
        f.rect(1730,860,635,240,CREAM,EDGE)
        banner(f,1750,863,595,'Independent evaluation',CORAL,34)
        f.text(2340,904,'audio + score',28,False,BLUE,'end')
        f.arrow([(2048,815),(2048,855)],CORAL)
        cols=[(1750,'Q1–Q3',['Local text','quality'],BLUE),(1955,'M1–M3',['Local music','compatibility'],TEAL),(2160,'G1–G3',['Global','composition'],PURPLE)]
        for x,code,lines,col in cols:
            f.text(x,965,code,34,True,col)
            f.lines(x,1011,lines,32,39,True)
        f.text(2048,1090,'Quoted evidence · strict rubrics',28,False,BLUE,'middle',True)
    return f.save('overview_method.svg')

def benchmark_v3(Fig,root):
    f=Fig('Evidence-linked data, local and long-form tasks, and a compact baseline',2400,1150)
    with f.group('data-construction'):
        f.rect(30,30,1370,760,ICE,EDGE)
        banner(f,50,33,1330,'A  Evidence-linked music–choreography supervision',BLUE,38)
        f.text(65,120,'Dance video',34,True,BLUE)
        for k,name in enumerate(['display_01.png','display_03.png','display_06.png']):
            x=66+k*116;f.rect(x-2,143,106,197,WHITE,EDGE)
            image(f,root/'source/case_frames'/name,x,145,102,190)
        f.text(65,378,'Genuine source frames',26,False,BLUE,italic=True)
        f.wave(66,462,330,91,TEAL,n=70)
        f.text(66,535,'Paired music',32,True,TEAL)
        f.arrow([(426,258),(465,258)],BLUE)
        f.rect(470,145,300,240,WHITE,BLUE,1.8)
        f.text(620,185,'2D rule events',34,True,BLUE,'middle')
        for i,(label,col) in enumerate([('Wrist ↑',BLUE),('Elbow ∠',ORANGE),('Knee ∠',TEAL)]):
            y=237+i*53;f.text(488,y,label,32,True,col)
            f.path(f'M630,{y-7} l18,-8 l21,10 l25,-14 l44,7',col,2.2)
        f.rect(820,145,270,240,LILAC,PURPLE,1.8)
        f.text(955,185,'GPT-6.1-sol',34,True,PURPLE,'middle')
        f.text(955,221,'Text-only refinement',28,False,PURPLE,'middle',True)
        for i,label in enumerate(['Coverage','Body / direction','Time / repeats']):
            y=266+i*45;check(f,838,y-7);f.text(870,y,label,32)
        f.arrow([(775,258),(815,258)],PURPLE)
        f.arrow([(1095,258),(1125,258)],PURPLE)
        f.rect(1130,145,245,560,WHITE,BLUE,1.8)
        f.lines(1252,186,['Paired','resource'],34,39,True,BLUE,'middle')
        for k,col in enumerate([BLUE,TEAL,PURPLE]):
            y=275+k*95
            f.wave(1148,y+27,64,48,col,n=20)
            for j,length in enumerate([133,111,120]):f.line(1230,y+j*18,1230+length,y+j*18,col,4)
        f.line(1148,559,1357,559,EDGE)
        f.lines(1252,604,['Audio · text','Time · source'],32,42,anchor='middle')
        f.text(1252,688,'Supervision + tasks',26,False,BLUE,'middle',True)
        # The audio bypass is explicit and never touches the text-only refiner.
        f.arrow([(66,462),(44,462),(44,130),(1252,130),(1252,140)],TEAL)
        f.text(690,121,'Aligned music retained alongside text',26,False,TEAL,'middle',True)
        f.dashed(470,439,620,151,EDGE)
        f.text(490,479,'NLF geometry extension',32,True,BLUE)
        for x,w,lab in [(490,130,'NLF'),(680,180,'3D events'),(920,150,'Merge')]:f.boxlabel(x,503,w,48,lab,32,col=BLUE)
        f.arrow([(625,527),(675,527)],BLUE);f.arrow([(865,527),(915,527)],BLUE)
        f.text(490,578,'Support masks · source provenance',28,False,BLUE)
        f.text(470,638,'Evidence-grounded movement text',32,True,PURPLE)
        f.lines(470,688,['“The left arm rises as the knees bend.”'],32,bold=True,col=PURPLE)
        f.text(470,735,'Source-event links preserve body, direction and order.',28)
        f.lines(65,650,['Source illustration;','not a scored prediction.'],26,34,col=BLUE,italic=True)
    with f.group('local-and-whole-song-benchmark'):
        f.rect(1450,30,920,760,CREAM,EDGE)
        banner(f,1470,33,880,'B  Music-to-text choreography benchmark',CORAL,38)
        # The paired resource defines cases; no reference is fed back into a candidate.
        f.arrow([(1380,220),(1475,220)],BLUE)
        for x,w,title,col in [(1480,410,'Local task',BLUE),(1930,410,'Whole-song task',PURPLE)]:
            f.rect(x,145,w,157,WHITE,EDGE)
            f.text(x+20,182,title,34,True,col)
        f.wave(1498,230,172,57,BLUE,n=40)
        f.lines(1700,229,['Bar unit','Action text'],32,39)
        for k,(lab,col) in enumerate([('A',PURPLE),('B',TEAL),('A′',ORANGE)]):
            x=1995+k*133;f.circle(x,237,22,WHITE,col,1.8);f.text(x,248,lab,32,True,col,'middle')
            if k<2:f.arrow([(x+27,237),(x+105,237)],BLUE)
        f.text(1480,354,'Local and global criteria',34,True)
        cols=[(1480,265,'Q','Local text quality',[('Q1','Specificity'),('Q2','Body / space'),('Q3','Local timing')],BLUE),
              (1780,265,'M','Music compatibility',[('M1','Rhythm'),('M2','Dynamics'),('M3','Affective')],TEAL),
              (2080,260,'G','Global composition',[('G1','Continuity'),('G2','Structure'),('G3','Motif variation')],PURPLE)]
        for x,w,letter,title,rows,col in cols:
            f.line(x,381,x+w,381,col,3)
            f.text(x,427,letter,40,True,col)
            f.wrap(x+52,414,title,w-52,32,36,True,col)
            for k,(code,lab) in enumerate(rows):
                f.text(x,491+k*41,code,32,True,col)
                f.text(x+62,491+k*41,lab,32)
            if letter=='M':f.text(x+62,609,'compatibility',32)
        f.line(1480,626,2340,626,EDGE)
        f.text(2340,130,'Reference agreement: diagnostic',26,False,BLUE,'end',True)
        f.text(1480,668,'Listen · quote · rate',32,True,CORAL)
        for x,num,word in [(1830,'1','Unsupported'),(2030,'3','Partial'),(2190,'5','Supported')]:
            f.text(x,668,num,32,True,CORAL);f.text(x+30,668,word,28)
        f.boxlabel(1480,704,330,66,['Music audio'],32,col=TEAL)
        f.boxlabel(1960,704,380,66,['Candidate score'],34,col=BLUE)
        f.text(1875,751,'+',37,True)
        f.line(1645,691,2150,691,CORAL,1.8)
        f.line(1645,699,1645,691,CORAL,1.8);f.line(2150,699,2150,691,CORAL,1.8)
        f.arrow([(1670,691),(1670,677)],CORAL)
        # The input bundle is distinct from rubric anchors and training references.
    with f.group('paired-example-and-provenance'):
        f.rect(30,850,920,240,LILAC,EDGE)
        banner(f,50,853,880,'A paired supervision record',PURPLE,34)
        f.wave(55,980,272,90,TEAL,n=60)
        f.lines(378,956,['Left arm rises','while the knees bend.'],34,43,True,PURPLE)
        for x,lab,col in [(378,'E1: arm rise',BLUE),(610,'E3: knee bend',ORANGE)]:
            f.line(x,1020,x+200,1020,col,4);f.text(x,1064,lab,30,True,col)
        f.text(50,1129,'Media and authored text illustrate the record format.',26,False,BLUE,italic=True)
    with f.group('compact-baseline'):
        f.rect(1010,850,1360,240,ICE,EDGE)
        f.text(1450,888,'Our baseline: trained expert + stateful harness',34,True,BLUE)
        f.boxlabel(1040,930,340,135,['Our Qwen3-Omni','+ LoRA','Joint-aware clauses'],32,col=PURPLE)
        f.arrow([(1252,710),(1252,925)],PURPLE)
        f.text(1280,830,'SFT',28,True,PURPLE)
        f.rect(1460,930,570,135,WHITE,TEAL,1.6)
        for x,w,lab in [(1477,128,'Planner'),(1661,152,'Controller'),(1869,143,'Tools')]:f.boxlabel(x,950,w,50,lab,32,col=TEAL)
        f.arrow([(1610,975),(1656,975)],BLUE,both=True);f.arrow([(1818,975),(1864,975)],BLUE,both=True)
        f.text(1745,1044,'Motifs · memory · scoped revision',30,anchor='middle')
        f.arrow([(1385,996),(1455,996)],BLUE)
        f.boxlabel(2110,930,230,135,['Hierarchical','score'],34,col=BLUE)
        f.arrow([(2035,996),(2105,996)],BLUE)
        f.arrow([(2225,925),(2225,775)],CORAL)
    return f.save('overview_benchmark.svg')
