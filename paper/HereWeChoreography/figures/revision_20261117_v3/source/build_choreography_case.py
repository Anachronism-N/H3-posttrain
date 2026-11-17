"""F3: equally sampled real frames, measured waveform and readable score analysis.

All text and geometry are native SVG. Embedded photographs are unchanged
source-frame pixels with a background-only display crop. Text is an authored
illustration, not a scored model prediction or a verified performance.
"""
from pathlib import Path
from html import escape
from PIL import Image, ImageFont
import base64
import hashlib
import io
import json
import math
import subprocess
import wave
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
FIG=OUT.parent
CASE=FIG/'source/case_example'
CASE_ID='wJPptuhclZmKAEWs'
W,H=1800,960
INK='#202C3C'; MUTED='#687385'; LINE='#C7D0DA'
SAGE='#008F95'; SLATE='#2474B5'; PLUM='#7852A9'; OCHRE='#E69A28'; CORAL='#D45D64'
PALE=['#FAFCFD','#FBFAFE','#FFFCF7','#FAFCFD']
CROP=(120,280,640,1250)
TIMES=[0,1,2,3,4,5]
INDICES=[16*t for t in TIMES]
X0,X1=145,1655
TEXTS=[
 'Lower the raised foot; sweep both hands in compact side-to-side arcs in front of the upper torso, keeping the elbows bent.',
 'Continue the same hand pattern with bent elbows; near the end, raise one knee.',
]
parts=[];bounds=[]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def font(size,bold=False):
 return ImageFont.truetype('C:/Windows/Fonts/'+('timesbd.ttf' if bold else 'times.ttf'),round(size))
def text(x,y,s,size=28,col=INK,bold=False,anchor='start',italic=False):
 width=font(size,bold).getlength(str(s))
 start=x-width/2 if anchor=='middle' else x-width if anchor=='end' else x
 bounds.append((start,y-size,start+width,y+size*.25,str(s)))
 parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="Times New Roman" font-size="{size}" font-weight="{700 if bold else 400}" fill="{col}" text-anchor="{anchor}"'+(' font-style="italic"' if italic else '')+f'>{escape(str(s))}</text>')
def line(points,color=LINE,sw=1.6,dash=None):
 parts.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
def rect(x,y,w,h,fill='white',stroke=LINE,r=0,sw=1.4):
 parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def badge(x,y,n,color):
 parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="16" fill="white" stroke="{color}" stroke-width="2.1"/>')
 text(x,y+7,str(n),24,color,True,'middle')
def arrow(x1,y1,x2,y2,color=INK,sw=1.7):
 line([(x1,y1),(x2,y2)],color,sw)
 ang=math.atan2(y2-y1,x2-x1);n=8;r=3.4
 q=[(x2,y2),(x2-n*math.cos(ang)+r*math.sin(ang),y2-n*math.sin(ang)-r*math.cos(ang)),(x2-n*math.cos(ang)-r*math.sin(ang),y2-n*math.sin(ang)+r*math.cos(ang))]
 parts.append('<path d="M'+' L'.join(f'{x:.2f},{y:.2f}' for x,y in q)+f' Z" fill="{color}" stroke="none"/>')
def rich(x,y,s,width,size=30,leading=37):
 phrases={
  'the raised foot':CORAL,'both hands':SAGE,'compact side-to-side arcs':SLATE,'in front of the upper torso':SLATE,
  'elbows bent':PLUM,'same hand pattern':SAGE,'bent elbows':PLUM,'near the end':OCHRE,'raise one knee':CORAL,
 }
 spans=[]
 for phrase,col in phrases.items():
  p=s.lower().find(phrase)
  if p>=0:spans.append((p,p+len(phrase),col))
 import re
 cx=x;cy=y
 for m in re.finditer(r'\S+\s*',s):
  word=m.group();col=next((c for a,b,c in spans if a<=m.start()<b),INK);bold=col!=INK
  word_width=font(size,bold).getlength(word)
  if cx>x and cx+word_width>x+width:cx=x;cy+=leading
  # Pale, local emphasis supports the words rather than an abstract color legend.
  if bold:rect(cx-1,cy-size*.85,word_width+2,size*1.03,PALE[1] if col==PLUM else PALE[0],'none',2,0)
  text(cx,cy,word,size,col,bold)
  cx+=word_width
 return cy

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 frame_dir=HERE/'case_frames';frame_dir.mkdir(exist_ok=True)
 video=CASE/(CASE_ID+'.mp4');wav=CASE/(CASE_ID+'.wav')
 beat=json.loads((CASE/(CASE_ID+'.beat_audit.json')).read_text(encoding='utf-8'))
 ann=json.loads((CASE/(CASE_ID+'.audit.json')).read_text(encoding='utf-8'))
 source_frames=json.loads((CASE/(CASE_ID+'.frame_audit.json')).read_text(encoding='utf-8'))
 assert sha(video)==source_frames['video_sha256']==ann['video']['sha256']
 assert sha(wav)==source_frames['audio_sha256']==ann['audio']['sha256']
 with wave.open(str(wav),'rb') as f:
  sr,n=f.getframerate(),f.getnframes()
  assert f.getnchannels()==1 and f.getsampwidth()==2
  audio=np.frombuffer(f.readframes(n),dtype='<i2').astype(float)/32768
 duration=n/sr
 assert duration==5.0625 and sr==24000
 assert beat['raw_prediction_record']['downbeats']==[.2,1.96,3.7]
 assert source_frames['video_probe']['fps_numerator']==16
 expected=[frame_dir/f'frame_{i+1:02}.png' for i in range(6)]
 if not all(f.is_file() for f in expected):
  expr='+'.join(f'eq(n,{i})' for i in INDICES)
  subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),'-vf',f"select='{expr}'",'-fps_mode','passthrough',str(frame_dir/'frame_%02d.png')],check=True)
 tx=lambda t:X0+(X1-X0)*t/duration
 parts.extend([f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
  '<title>A body-specific, spatially clear and connected choreography score</title>',
  '<desc>Six real video frames at one-second intervals, real audio and predicted bars. Colored fields parse authored instructions at window scope, not measured action timing or motion trajectories.</desc>',
  f'<rect width="{W}" height="{H}" fill="white"/>'])
 text(50,43,'Dance style: Contemporary',36,INK,True)
 text(1750,42,'Real video and music · illustrative score',26,MUTED,False,'end')

 # True frames lead the graphic, equally spaced in source time and on the page.
 parts.append('<g id="uniform-real-video-filmstrip">')
 photo_top=70;photo_h=278;photo_w=photo_h*(CROP[2]-CROP[0])/(CROP[3]-CROP[1])
 rows=[]
 for i,(f,t,idx) in enumerate(zip(expected,TIMES,INDICES)):
  pixels=Image.open(f);assert pixels.size==(704,1280)
  crop=pixels.crop(CROP);saved=frame_dir/f'display_{i+1:02}.png';crop.save(saved)
  payload=base64.b64encode(saved.read_bytes()).decode('ascii')
  x=tx(t)-photo_w/2
  parts.append(f'<g id="real-frame-{i+1}"><image x="{x:.2f}" y="{photo_top}" width="{photo_w:.2f}" height="{photo_h}" xlink:href="data:image/png;base64,{payload}"/>')
  rect(x,photo_top,photo_w,photo_h,'none',LINE,0,.95)
  text(tx(t),381,f'{t:.1f} s',28,INK,True,'middle')
  parts.append('</g>')
  rows.append(dict(original_frame_file=str(f.relative_to(OUT)),original_frame_sha256=sha(f),display_frame_file=str(saved.relative_to(OUT)),display_sha256=sha(saved),video_frame=idx,seconds=t))
 parts.append('</g>')

 # Measured PCM envelope; no local gain or invented accents.
 parts.append('<g id="measured-waveform-and-predicted-bars">')
 text(50,427,'Music',28,INK,True)
 step=48;starts=np.arange(0,n,step);ends=np.minimum(starts+step,n)
 centers=(starts+ends)/(2*sr);peak=float(np.abs(audio).max());mid=419;amp=24
 upper=[(tx(float(t)),mid-float(audio[a:b].max())/peak*amp) for t,a,b in zip(centers,starts,ends)]
 lower=[(tx(float(t)),mid-float(audio[a:b].min())/peak*amp) for t,a,b in zip(centers,starts,ends)]
 path='M'+' L'.join(f'{x:.2f},{y:.2f}' for x,y in upper+lower[::-1])+' Z'
 parts.append(f'<path d="{path}" fill="{SAGE}" stroke="none"/>')
 for t in [.2,1.96,3.7]:line([(tx(t),390),(tx(t),446)],OCHRE,1.6,'4 4')
 for a,b,label,color in [(.2,1.96,'Bar 1 · 0.20–1.96 s',SLATE),(1.96,3.7,'Bar 2 · 1.96–3.70 s',SLATE),(3.7,duration,'Ending · 3.70–5.06 s',OCHRE)]:
  line([(tx(a)+2,450),(tx(b)-2,450)],color,2.2)
  text((tx(a)+tx(b))/2,479,label,26,color,True,'middle')
 parts.append('</g>')

 # Unit-level language fields, explicitly not detected action intervals.
 parts.append('<g id="unit-level-instruction-fields">')
 text(50,525,'Instruction fields',29,INK,True)
 text(1750,525,'Unit-level schematic; no measured motion trace',26,MUTED,False,'end')
 boundary=tx(3.7)
 for y,label,col in [(563,'Hands',SAGE),(615,'Pose',PLUM),(667,'Legs',CORAL)]:
  text(50,y+8,label,28,col,True)
  rect(X0,y-23,X1-X0,44,'#FAFCFD',LINE,3,.9)
 rect(X0,540,boundary-X0-6,44,'#FAFCFD',SAGE,3,1.5)
 rect(boundary+6,540,X1-boundary-6,44,'#FAFCFD',SAGE,3,1.5)
 text(X0+18,572,'both hands · compact side-to-side arcs · in front of the torso',28,SAGE,True)
 text(boundary+24,572,'continue the same pattern',27,SAGE,True)
 rect(X0,592,X1-X0,44,'#FBFAFE',PLUM,3,1.5)
 text((X0+X1)/2,624,'BENT ELBOWS · retained across the window boundary',28,PLUM,True,'middle')
 line([(boundary,587),(boundary,593)],PLUM,2)
 line([(boundary,635),(boundary,643)],PLUM,2)
 rect(X0,644,boundary-X0-6,44,'#FFFCF7',CORAL,3,1.5)
 rect(boundary+6,644,X1-boundary-6,44,'#FFFCF7',CORAL,3,1.5)
 text(X0+18,676,'lower the raised foot',28,CORAL,True)
 text(boundary+24,676,'near-end knee lift',28,CORAL,True)
 parts.append('</g>')

 # Plain-language windows map directly to the fields above. No abstract footer.
 parts.append('<g id="natural-language-window-captions">')
 a_x,a_w=50,1090;b_x,b_w=1166,584
 top,bottom=722,923
 rect(a_x,top,a_w,bottom-top,'#FAFCFD',LINE,4,1.1)
 rect(b_x,top,b_w,bottom-top,'#FBFAFE',LINE,4,1.1)
 rect(a_x,top,7,bottom-top,SAGE,'none',0,0)
 rect(b_x,top,7,bottom-top,PLUM,'none',0,0)
 text(a_x+24,762,'Window 1 · Bars 1–2',31,SLATE,True)
 text(a_x+a_w-22,762,'0–3.70 s',27,MUTED,False,'end')
 text(b_x+24,762,'Ending window',31,PLUM,True)
 text(b_x+b_w-18,762,'3.70–5.06 s',27,MUTED,False,'end')
 end1=rich(a_x+24,813,TEXTS[0],a_w-50,31,39)
 end2=rich(b_x+24,813,TEXTS[1],b_w-48,30,37)
 assert end1<901 and end2<906
 parts.append('</g>')
 parts.append('</svg>')
 svg=OUT/'choreography_case.svg';svg.write_text('\n'.join(parts)+'\n',encoding='utf-8')
 assert all(-1<=x0<=W and 0<=y0 and x1<=W+1 and y1<=H for x0,y0,x1,y1,_ in bounds)
 audit=dict(schema='herewechoreography.case_revision.20261117_v3',case_id=CASE_ID,
  source_video_sha256=sha(video),source_audio_sha256=sha(wav),display_crop_xyxy=CROP,
  crop_scope='Unchanged pixels, identical background-only crop; full body retained',
  frame_selection=dict(source_fps=16,indices=INDICES,times_seconds=TIMES,equal_time_step_seconds=1,rows=rows),
  waveform=dict(kind='2 ms PCM minimum/maximum envelope',global_peak=peak,global_scaling_only=True,sample_rate=sr,sample_count=n,synthetic=False),
  predicted_downbeats=[.2,1.96,3.7],planning_units_seconds=[[0,3.7],[3.7,duration]],
  leading_fragment_merged=[0,.2],ending_is_incomplete_fragment=True,
  text_origin='Author-composed illustrative instructions, not model output, GT, or quality rating',
  instructions=TEXTS,quality_ratings=None,independent_av_alignment_verified=False,
  layout='Equal-time filmstrip, measured audio/bar clock, unit-level body/action fields, two natural-language windows',
  instruction_field_rows=['Hands: unit-scoped hand action and range','Pose: written bent-elbow state carried across both units','Legs: unit-scoped lower-foot / near-end knee instruction'],
  field_bands_are_schematic_not_measured_action_intervals=True,
  changes_from_previous='Reordered real frames before waveform; replaced three long explanatory footer blocks with native body/action/state fields; saturated foreground colors on pale panels',
  no_AI_modification_to_frames=True,output_svg_sha256=sha(svg),
  bounds_checked=True,native_text_count=len(bounds),all_text_font='Times New Roman')
 (HERE/'choreography_case.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(svg=str(svg),native_text_count=len(bounds),frame_times=TIMES,text_last_baselines=[end1,end2]),indent=2))

if __name__=='__main__':main()
