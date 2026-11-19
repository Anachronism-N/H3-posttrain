"""Current progress is only the corrected English merged-window motion revision."""
from motion_revision_progress_v1 import show,run
import argparse
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--watch",action="store_true");p.add_argument("--json",action="store_true");p.add_argument("--service",action="store_true");a=p.parse_args()
    if a.service:run()
    else:show(a.watch,a.json)
