"""Native editable annotation statistics, from a frozen source-only audit.

No inference or new quality annotation. All six panels use the same paired
population; events, words, source windows and producer flags stay distinct.
"""
from pathlib import Path
import hashlib
import json
from build_annotation_pipeline import SVG

SOURCE=Path(__file__).resolve().parent
DATA=SOURCE/'annotation_audit.source.json'
INK='#242A33';SLATE='#65778A';SAGE='#667A70';PLUM='#85778C';OCHRE='#A18C69'
EDGE='#B7BFC7';GRID='#E5E8EB'


def panel(s,x,y,title,subtitle,fill='#FAFBFD'):
    s.box(x,y,616,365,fill,EDGE,4,1)
    s.text(x+20,y+37,title,29,'bold')
    s.text(x+20,y+68,subtitle,22,color=SLATE)


def histogram(s,x,y,labels,counts,n,color,unit,maximum=60):
    left=x+60;right=x+592;top=y+105;bottom=y+282
    s.group('bars-'+str(x)+'-'+str(y))
    for v in [0,25,50]:
        yy=bottom-(bottom-top)*v/maximum
        s.line(left,yy,right,yy,GRID,1)
        s.text(left-12,yy+7,str(v),21,color=SLATE,anchor='end')
    s.line(left,top,left,bottom,EDGE,1.2);s.line(left,bottom,right,bottom,EDGE,1.2)
    step=(right-left)/len(labels);bw=step*.68
    shares=[]
    for i,(lab,count) in enumerate(zip(labels,counts)):
        value=count/n*100;shares.append(value);h=(bottom-top)*value/maximum;xx=left+step*(i+.5)
        s.box(xx-bw/2,bottom-h,bw,h,color,'none',0)
        s.text(xx,bottom-h-10,f'{value:.1f}',22,anchor='middle')
        s.text(xx,bottom+28,lab,21,anchor='middle')
    s.text((left+right)/2,y+343,unit,23,anchor='middle')
    s.text(x+13,y+98,'Clips (%)',18,color=SLATE)
    s.end()
    return shares


def main():
    d=json.loads(DATA.read_text(encoding='utf-8'));n=d['n']
    assert n==22352
    assert d['source_sha256']=='bef81f195f1fc88372ee6b3a8206818d043e28721fff97f9ab4a077254637052'
    assert sum(d['original_events']['counts'])==sum(d['caption_words']['counts'])==n
    assert sum(d['source_bodypart_counts'].values())==d['source_event_total']
    assert sum(d['style_counts'].values())==sum(d['producer_gate'].values())==n
    assert d['pairing']['strict_paired_unique_ids']==n
    assert d['pairing']['all_ids_used_not_quality_selected'] is True
    s=SVG(1920,810)
    x1,x2,x3=16,652,1288;y1,y2=12,393
    s.group('source-event-count')
    panel(s,x1,y1,'(a) Source event count',f"Mean {d['event_summary']['mean']:.2f} statements per clip")
    events=histogram(s,x1,y1,['0–9','10–19','20–39','40–59','60–79','≥80'],d['original_events']['counts'],n,SAGE,'Rule-event statements')
    s.end()
    s.group('caption-length')
    panel(s,x2,y1,'(b) Refined caption length',f"Mean {d['caption_word_summary']['mean']:.2f} words; style header excluded",'#FCFAFD')
    words=histogram(s,x2,y1,['0–59','60–99','100–139','140–179','≥180'],d['caption_words']['counts'],n,PLUM,'English body words')
    s.end()

    s.group('source-bodypart-bias')
    panel(s,x3,y1,'(c) Body-part information','Shares of source event occurrences','#FBFCFA')
    bp=[]
    for j,(name,color) in enumerate(zip(['wrist','elbow','knee'],[SLATE,SAGE,OCHRE])):
        yy=y1+130+j*65;value=d['source_bodypart_counts'][name]/d['source_event_total']*100;bp.append(value)
        s.text(x3+25,yy+7,name.title(),25)
        s.box(x3+133,yy-17,value/70*365,28,color,'none',0)
        s.text(x3+143+value/70*365,yy+7,f'{value:.1f}%',25,'bold')
    s.text(x3+308,y1+332,'Event share is not motion correctness.',22,anchor='middle',italic=True)
    s.end()

    s.group('source-temporal-granularity')
    wc=d['window_counts'];win=[wc.get(str(k),0) for k in range(1,5)]
    win.append(sum(v for k,v in wc.items() if int(k)>=5));assert sum(win)==n
    panel(s,x1,y2,'(d) Source temporal granularity',f"Mean {d['window_summary']['mean']:.2f} windows per 5.0625 s clip")
    windows=histogram(s,x1,y2,['1','2','3','4','≥5'],win,n,SLATE,'Source rule-event windows')
    s.end()

    s.group('source-style-metadata')
    sc=d['style_counts'];styles=[sc['unknown'],sc['contemporary'],n-sc['unknown']-sc['contemporary']]
    panel(s,x2,y2,'(e) Source-style metadata','Existing labels; no independent style relabeling','#FDFBFA')
    sty=[]
    for j,(name,count,color) in enumerate(zip(['Unknown','Contemporary','Other labels'],styles,[SLATE,SAGE,OCHRE])):
        yy=y2+130+j*65;value=count/n*100;sty.append(value)
        s.text(x2+23,yy+7,name,24)
        s.box(x2+166,yy-17,value/70*306,28,color,'none',0)
        s.text(x2+177+value/70*306,yy+7,f'{value:.1f}%',25,'bold')
    s.text(x2+308,y2+332,'Unknown labels remain in the audit.',22,anchor='middle',italic=True)
    s.end()

    s.group('automatic-and-structural-audit')
    panel(s,x3,y2,'(f) Annotation audit','Producer-reported flags and identity checks','#FCFAFD')
    s.line(x3+20,y2+93,x3+596,y2+93,EDGE,1)
    s.text(x3+24,y2+124,'Automatic check: pass',25)
    s.text(x3+586,y2+124,f"{d['producer_gate']['true']/n*100:.2f}%",25,'bold',anchor='end')
    s.text(x3+24,y2+162,'Automatic check: flagged',25)
    s.text(x3+586,y2+162,f"{d['producer_gate']['false']/n*100:.2f}%",25,'bold',anchor='end')
    s.line(x3+20,y2+181,x3+596,y2+181,EDGE,1)
    s.text(x3+24,y2+215,'Paired record IDs',25)
    s.text(x3+586,y2+215,f'{n:,} / {n:,}',25,anchor='end')
    s.text(x3+24,y2+253,'Paired metadata mismatches',25)
    s.text(x3+586,y2+253,str(d['pairing']['source_or_split_mismatch']),25,anchor='end')
    s.text(x3+24,y2+291,'Style-field changes',25)
    s.text(x3+586,y2+291,str(d['style_field_changed_count']),25,anchor='end')
    s.text(x3+308,y2+338,'Automatic flags are not human correctness.',22,anchor='middle',italic=True)
    s.end()
    s.text(960,794,f'All {n:,} paired records retained. Events, words, windows, and automatic flags are different units.',23,anchor='middle')
    output=s.finish('annotation_audit.svg')
    audit={'source_json':DATA.name,'source_json_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
           'raw_audit_sha256':d['source_sha256'],'n':n,'quality_filter_applied':False,
           'event_bin_percentages':events,'word_bin_percentages':words,'bodypart_event_percentages':bp,
           'source_window_bin_percentages':windows,'style_percentages':sty,
           'producer_gate_percentages':{k:v/n*100 for k,v in d['producer_gate'].items()},
           'all_chart_bars_native':True,'all_text_native':True,'font_family':'Times New Roman',
           'unsupported_quality_rates_plotted':False}
    (SOURCE/'annotation_audit.derived.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':main()
