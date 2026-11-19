"""Current approved mainline: original MS plus32B text-only wording."""
import argparse
from ms_llm32_progress_live_v4 import show
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--json',action='store_true');a=p.parse_args();show(a.watch,a.json)
