"""Frozen measured onset-envelope features, not verified musical beat or tempo."""
import argparse,json,hashlib,math
from pathlib import Path
import librosa,numpy as np,soundfile as sf
from scipy.signal import find_peaks,resample_poly
from captionlab.audio_observations import observe


def evidence(audio,rate):
 div=math.gcd(rate,24000);x=resample_poly(audio,24000//div,rate//div).astype(np.float32) if rate!=24000 else np.asarray(audio,dtype=np.float32)
 env=librosa.onset.onset_strength(y=x,sr=24000,hop_length=240,n_fft=1024)
 peaks,_=find_peaks(env,distance=15,prominence=.25*max(env)) if len(env) and max(env)>0 else (np.array([],dtype=int),{})
 times=peaks*.01;diff=np.diff(times);median=float(np.median(diff)) if len(diff) else None;cv=float(np.std(diff)/np.mean(diff)) if len(diff) else None
 regular=bool(len(times)>=4 and cv is not None and cv<=.2)
 return dict(waveform=observe(x,24000),spectral_flux=dict(peak_times_seconds=times.tolist(),peak_count=len(times),
 median_peak_interval_seconds=median,interval_cv=cv,regular_onset_pattern=regular,onset_repetition_per_minute=60/median if median else None,
 peak_distance_seconds=.15,relative_prominence=.25,hop_seconds=.01),
 interpretation='Measured spectral-flux peak activity, NOT verified musical BPM/beat grid; multiples, subdivisions, vocal syllables and missing peaks can differ from musical beat.')


def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 data=json.loads(a.manifest.read_text());rows=data['cases'];out=[]
 for row in rows:
  info=row['audio'];x,rate=sf.read(info['path'],dtype='float32',always_2d=True)
  if x.shape[1]!=1 or len(x)!=info['num_samples'] or rate!=info['sample_rate']:raise ValueError('Registered audio geometry changed')
  out.append(dict(case_id=row['case_id'],source_group_id=row['source_group_id'],audio_sha256=info['sha256'],evidence=evidence(x[:,0],rate)))
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(records=out,manifest_sha=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parameters_frozen_on_synthetic_dev=True,not_quality_score=True),indent=2)+'\n')
 print(json.dumps(dict(completed=True,records=len(out))),flush=True)


if __name__=='__main__':main()
