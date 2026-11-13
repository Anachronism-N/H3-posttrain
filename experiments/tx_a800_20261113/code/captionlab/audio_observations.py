"""Measured waveform envelope evidence; not music genre, beat truth or dance fit."""
import numpy as np


def observe(waveform,rate):
    x=np.asarray(waveform,dtype=np.float64)
    frame=max(1,round(.02*rate));rms=np.array([np.sqrt(np.mean(x[i:i+frame]**2)) for i in range(0,len(x),frame)])
    peak=float(rms.max()) if len(rms) else 0.0
    threshold=.03*peak
    active=rms>threshold if peak else np.zeros(len(rms),dtype=bool)
    runs=np.flatnonzero(active & ~np.r_[False,active[:-1]]) if len(active) else np.array([],dtype=int)
    seconds=[float(i*.02) for i in runs]
    return dict(duration_seconds=len(x)/rate,frame_seconds=.02,rms=float(np.sqrt(np.mean(x*x))) if len(x) else 0.0,
        peak_frame_rms=peak,relative_activity_threshold=.03,low_energy_fraction=float(1-active.mean()) if len(active) else 1.0,
        active_energy_run_count=len(seconds),active_energy_run_starts_seconds=seconds,
        interpretation='Thresholded amplitude-envelope activity runs, NOT detected beats or musical tempo; continuous tracks may have one run despite many beats',
        rms_per_second=[float(np.sqrt(np.mean(x[i:i+rate]**2))) for i in range(0,len(x),rate)])
