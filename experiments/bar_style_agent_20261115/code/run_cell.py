"""One (song, arm) cell via run_composition; emits a registry row with logical results."""
import json, sys
from pathlib import Path

JOB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOB))
from choreobench.io import read_json
from choreobench.composition_workflows import run_composition

METHOD = {'W': 'workflow', 'SF': 'self_refine', 'RA': 'react', 'O': 'music_structured'}
LIMITS = dict(seed=42, max_output_tokens=4096, outline_max_output_tokens=4096,
              max_request_characters=400000, max_input_tokens=4000000,
              max_total_output_tokens=300000, repeat_groups=5)


def main():
    catalog, song, arm, provider_path, out_dir = sys.argv[1:6]
    flags = set(sys.argv[6:])
    extra = dict()
    args = sys.argv[6:]
    for i, value in enumerate(args):
        if value == '--music': extra['music'] = args[i+1]
        if value == '--style': extra['style'] = args[i+1]
        if value == '--call-cap': extra['cap'] = int(args[i+1])
    if '--long-output' in flags:
        LIMITS['max_output_tokens']=8192; LIMITS['outline_max_output_tokens']=8192
    if '--xlong-output' in flags:
        LIMITS['max_output_tokens']=16384; LIMITS['outline_max_output_tokens']=16384
    print('CELL_START %s %s' % (song, arm), flush=True)
    out = Path(out_dir)
    result = run_composition(catalog, song, read_json(provider_path), out,
                             method=METHOD[arm], controller_version='v2',
                             music_summary_path=extra.get('music', str(JOB / ('assets/music/%s.json' % song))),
                             style_prior=None if '--no-style' in flags else extra.get('style', str(JOB / 'assets/global_style_prior_v2.json')),
                             acoustic_guidance=arm == 'O' and '--no-acoustic' not in flags,
                             rewrite_repeats=arm in ('W', 'O'),
                             max_calls_cap=extra.get('cap'),
                             snap_schedule='--snap' in flags, continuity_prompt='--continuity' in flags, fallback_unit='--fallback' in flags,
                             **LIMITS)
    complete = result.get('full_song_succeeded') is True
    costs = {k: result.get(k) for k in ('calls_used', 'input_tokens', 'output_tokens')}
    final = dict(arm=arm, song_id=song, complete=complete,
                 plan=str(out / 'plan.json') if complete else None,
                 physical_run=str(out), costs=costs, status=result.get('stop_reason'))
    logical = [final]
    if arm in ('W', 'O'):
        before = out / 'before_rewrite.json'
        before_costs = out / 'before_rewrite_costs.json'
        if before.is_file() and before_costs.is_file():
            prefix = read_json(before_costs)
            logical.insert(0, dict(final, arm='W' if arm == 'W' else 'O-noR',
                                   complete=prefix.get('full_song_succeeded') is True,
                                   plan=str(before),
                                   costs={k: prefix.get(k) for k in costs},
                                   status='pre_rewrite_snapshot'))
        else:
            logical.insert(0, dict(final, arm='W-plusR' if arm == 'W' else 'O-noR',
                                   complete=False, plan=None, costs=None,
                                   status='pre_rewrite_plan_unavailable'))
        if arm == 'W':
            logical[-1]['arm'] = 'W-plusR'
    print('CELL_DONE %s %s complete=%s calls=%s stop=%s' % (song, arm, complete, result.get('calls_used'), result.get('stop_reason')), flush=True)
    print('CELL ' + json.dumps(dict(song_id=song, arm=arm, status='terminal', error=None,
                                    result=result, logical_results=logical), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        print('CELL_ERROR ' + traceback.format_exc(), flush=True)
        raise
