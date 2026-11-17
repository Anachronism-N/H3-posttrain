"""Build the paper's grouped local-evaluation table from frozen results.

No inference, editing of predictions, or choice of a best scoring version.
Reference agreement, Q ratings and M ratings retain distinct cohorts.
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import collections
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
MODELS = [
    ('qwen3-omni-instruct', 'Qwen3-Omni-Instruct'),
    ('moss-audio', 'MOSS-Audio'), ('moss-music', 'MOSS-Music'),
    ('music-flamingo', 'Music-Flamingo'), ('audio-flamingo3', 'Audio-Flamingo3'),
    ('audio-flamingo-next', 'Audio-Flamingo-Next'), ('qwen25-omni', 'Qwen2.5-Omni'),
    ('SFT_v3_rp105', 'ChoreoScribe (ours)'),
]
RULES = 'SFT_v3_rp105+music-rules-v5'
QD = ['D1', 'D2', 'D3']
MD = ['D4', 'D5', 'D6']

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def score_ok(r):
    return r.get('status') == 'valid_structure' and type(r.get('score')) is int and 1 <= r['score'] <= 5
def fmt(v): return str(Decimal(str(v)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
def id_hash(ids): return hashlib.sha256(('\n'.join(sorted(ids))+'\n').encode()).hexdigest()

def main():
    source = read(RESULTS/'e2_full2000.source.json')
    q_receipt = read(RESULTS/'d123_refine_v1/table.source.json')
    ours_manifest = read(RESULTS/'d123_refine_v1/manifest.json')
    base_manifest = read(RESULTS/'d123_baselines/manifest.json')
    for name,h in ours_manifest['artifacts_sha256'].items():
        assert sha(RESULTS/'d123_refine_v1'/name)==h,name
    for name,item in base_manifest['files'].items():
        assert sha(RESULTS/'d123_baselines'/name)==item['sha256'],name
    q_scores = {method:{d:{} for d in QD} for method,_ in MODELS}
    for r in rows(RESULTS/'d123_baselines/judgments_r4_holdout.jsonl'):
        q_scores[r['method']][r['dimension']][r['case_id']] = r
    holdout = set(q_scores[MODELS[0][0]]['D1'])
    for r in rows(RESULTS/'d123_refine_v1/judgments.jsonl'):
        if r['case_id'] in holdout:
            for d in QD:q_scores['SFT_v3_rp105'][d][r['case_id']]=r['scores']['r4'][d]
    q_common={d:set.intersection(*[{c for c,r in q_scores[m][d].items() if score_ok(r)} for m,_ in MODELS]) for d in QD}
    q_means={m:{d:statistics.mean(q_scores[m][d][c]['score'] for c in q_common[d]) for d in QD} for m,_ in MODELS}
    for m,_ in MODELS:
        old='Ours' if m=='SFT_v3_rp105' else m
        for d in QD:assert math.isclose(q_means[m][d],q_receipt['means'][old][d],abs_tol=1e-12)

    music=RESULTS/'music_compatibility_v3';provenance=read(music/'provenance.json')
    for name,item in provenance['files'].items():assert sha(music/name)==item['sha256'],name
    records=rows(music/'results.jsonl');published=read(music/'summary.json')
    assert len(records)==6200 and sha(music/'results.jsonl')==published['results_sha256']
    observed={r['case_id']:r for r in records if r['stage']=='observation'}
    assert len(observed)==200 and all(r['status']=='valid_observation' for r in observed.values())
    full_ids={r['case_id'] for r in rows(RESULTS/'d123_refine_v1/outputs.jsonl')}
    assert set(observed)<=full_ids
    m_scores=collections.defaultdict(lambda:collections.defaultdict(dict))
    versions=collections.Counter();statuses=collections.Counter();models=collections.Counter()
    for r in records:
        if r['stage']!='judge':continue
        c,m,d=r['case_id'],r['method'],r['dimension']
        assert c not in m_scores[m][d] and d in MD
        if r['status']=='valid_structure':assert r.get('raw_audio_supplied') is True
        assert r['audio_processing']['input_sha256']==observed[c]['audio_processing']['input_sha256']
        item=dict(r);item['score']=r.get('judgment',{}).get('score')
        if score_ok(item):assert item['score']==r['content']['score']
        m_scores[m][d][c]=item
        statuses[r['status']]+=1;versions[r.get('judgment',{}).get('protocol_version')]+=1;models[r['model']]+=1
    available={}
    for m,dims in m_scores.items():
        available[m]={}
        for d,items in dims.items():
            assert set(items)==set(observed)
            vals=[r['score'] for r in items.values() if score_ok(r)]
            available[m][d]={'n':len(vals),'mean':statistics.mean(vals)}
            p=published['per_method_dimension'][m][d]
            assert len(vals)==p['n'] and math.isclose(statistics.mean(vals),p['mean'],abs_tol=.00051)
    compared=[m for m,_ in MODELS]+[RULES]
    common={d:set.intersection(*[{c for c,r in m_scores[m][d].items() if score_ok(r)} for m in compared]) for d in MD}
    m_means={m:{d:statistics.mean(m_scores[m][d][c]['score'] for c in common[d]) for d in MD} for m in compared}
    metrics={}
    for m,_ in MODELS:
        if m not in source['methods']:metrics[m]=[None]*6
        else:
            x=source['methods'][m]
            metrics[m]=[x['corpus_bleu4_0_100'],100*x['rougeL_f1'],100*source['common_bert_f1'][m],100*x['token_f1'],100*x['action_tag_micro_f1'],100*x['body_tag_micro_f1']]
        metrics[m] += [q_means[m][d] for d in QD]+[m_means[m][d] for d in MD]
    best=[max(v[i] for v in metrics.values() if v[i] is not None) for i in range(12)]
    tex=r'''% Generated by results/build_unified_caption_table.py. Frozen cohorts remain separate.
\begin{table*}[t]
\centering
\caption{\textbf{Local music-to-choreography evaluation.} Reference measures are scaled to 0--100; Q/M criteria each use 1--5. Bold marks the best direct-model score in each column. The deterministic music-rule control is separate from direct SFT and uses no additional LLM. Groups follow their own evaluation protocols; no cross-group total is formed.}
\label{tab:local-results}
\label{tab:e2-full2000-v2}
\label{tab:refinement-d123}
\begingroup\scriptsize
\setlength{\tabcolsep}{2.3pt}
\renewcommand{\arraystretch}{1.16}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrrrrr@{}}
\toprule
& \multicolumn{6}{c}{Reference agreement $\uparrow$} & \multicolumn{3}{c}{Text quality $\uparrow$} & \multicolumn{3}{c}{Music compatibility $\uparrow$} \\
\cmidrule(lr){2-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
Method & BLEU-4 & R-L & BERT-F1$^\dagger$ & Tok-F1 & KW-F1 & BP-F1 & Q1 & Q2 & Q3 & M1 & M2 & M3 \\
\midrule
'''
    for m,name in MODELS:
        if m=='SFT_v3_rp105':tex+='\\midrule\n';name=r'\textbf{ChoreoScribe (ours)}'
        cells=[name]
        for i,v in enumerate(metrics[m]):
            cell='--' if v is None else fmt(v)
            if v is not None and math.isclose(v,best[i],abs_tol=1e-12):cell=r'\textbf{'+cell+'}'
            cells.append(cell)
        tex+=' & '.join(cells)+r' \\'+'\n'
    tex+=r'\midrule'+'\n'+r'ChoreoScribe + rules & '+ ' & '.join(['--']*9+[fmt(m_means[RULES][d]) for d in MD])+r' \\'+'\n'
    tex+=r'''\bottomrule
\end{tabular*}
\par\vspace{3pt}
\begin{minipage}{\textwidth}\scriptsize
R-L: ROUGE-L; Tok-F1: token overlap; KW/BP-F1: motion/body lexicon agreement.
Q1--Q3 assess specificity, body/space, and local timing; M1--M3 assess rhythm,
movement dynamics, and affective compatibility. $^\dagger$Common-case BERTScore.
Missing evaluations are ``--''. Coverage and configuration are in Appendix~\ref{sec:experimental-details}.
\end{minipage}
\endgroup
\end{table*}
'''
    (ROOT/'tables/captioner_unified.tex').write_text(tex,encoding='utf-8')
    receipt={'schema':'paper.grouped_caption_results.v1','source_commit_music':provenance['source_commit'],
      'reference_source_sha256':sha(RESULTS/'e2_full2000.source.json'),'Q_source_sha256':sha(RESULTS/'d123_refine_v1/table.source.json'),
      'Q_protocol':'r4','Q_registered':len(holdout),'Q_common_valid_counts':{d:len(x) for d,x in q_common.items()},
      'Q_common_ids_sha256':{d:id_hash(x) for d,x in q_common.items()},
      'M_release_declared_protocol':'sol61.music-style-affect.v3.0-local',
      'M_record_wrapper_versions':dict(versions),'M_wrapper_matches_declared_version':False,
      'M_registered':len(observed),'M_common_valid_counts':{d:len(x) for d,x in common.items()},
      'M_common_ids_sha256':{d:id_hash(x) for d,x in common.items()},'M_judgment_statuses':dict(statuses),
      'M_judge_models':dict(models),'M_available_case':available,'M_common_case':m_means,
      'columns':['BLEU4','ROUGE_L','BERT_F1','TOKEN_F1','KW_F1','BP_F1','Q1','Q2','Q3','M1','M2','M3'],
      'direct_model_values':metrics,'music_rules_values':m_means[RULES],
      'rounding':'Decimal ROUND_HALF_UP to two decimals','missing_imputed':False,
      'cohorts_or_protocols_pooled':False,'cross_group_aggregate':None,'rules_control_joined_to_SFT_row':False,
      'captioner_inference':'Direct SFT; no additional LLM editor, per author clarification.',
      'frozen_result_files_modified':False,'independent_human_calibration_completed':False,
      'M_full_judge_prompt_hash_released':False}
    (RESULTS/'unified_caption_results.source.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'table':'tables/captioner_unified.tex','music_common_counts':receipt['M_common_valid_counts'],
       'music_scores':{m:{d:fmt(v) for d,v in dims.items()} for m,dims in m_means.items()}},indent=2))

if __name__=='__main__':main()
