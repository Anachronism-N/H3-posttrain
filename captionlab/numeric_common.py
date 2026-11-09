"""Fixed-source numeric snapshots; no generation, training, or acceptance by fiat."""
import hashlib
import json
from pathlib import Path, PureWindowsPath
import unicodedata
import torch


def digest_tensor(value):
    x=value.detach().cpu().contiguous()
    h=hashlib.sha256(str((str(x.dtype),list(x.shape))).encode())
    h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _numeric_filename_id(value):
    """Keep the original Unicode ID, but require one portable path component."""
    if not isinstance(value,str) or not value or not value.strip():
        raise ValueError('Numeric source id must be nonempty text for a portable filename component')
    if value in ('.','..') or value.endswith('.') or value[-1].isspace():
        raise ValueError('Numeric source id cannot be . or .. or end in a dot or whitespace')
    if any(c in '<>:"/\\|?*' or unicodedata.category(c)=='Cc' for c in value):
        raise ValueError('Numeric source id contains a path separator, reserved filename character or control character')
    try:value.encode('utf-8')
    except UnicodeError as error:raise ValueError('Numeric source id must be valid Unicode text') from error
    if PureWindowsPath(value).is_reserved() or value.split('.')[0].rstrip(' ').upper()=='CLOCK$':
        raise ValueError('Numeric source id is a reserved Windows device name')
    return value


def rows_from_manifest(path):
    m=json.loads(Path(path).read_text(encoding='utf-8'))
    if len(m['cases'])!=3:raise ValueError('Exactly three fixed source cases required')
    seen=set()
    for case in m['cases']:
        identifier=_numeric_filename_id(case['target_source']['id'])
        if identifier.casefold() in seen:
            raise ValueError('Duplicate numeric source id under case-insensitive filename comparison')
        seen.add(identifier.casefold())
    rows=[]
    for case in m['cases']:
        s=case['target_source'];audio=s['audio'];duration=audio['num_samples']/audio['sample_rate']
        rows.append(dict(id=s['id'],source_group_id=s['source_group_id'],audio_path=audio['path'],
            audio_sha256=audio['sha256'],audio_duration_seconds=duration,audio_window_seconds=[0.,duration],
            target_text=s['raw_caption'],target_windows=case['target_windows']))
    return rows


def fixed_positions(adapter,batch):
    n=int(batch['labels'].ne(-100).long().argmax(dim=1)[0]);seq=batch['input_ids'][0].tolist()
    positions={n-1,n,min(n+7,len(seq)-2),len(seq)-2}
    for word in [' wrist',' left',' right',' knee',' increases',' decreases']:
        tokens=adapter.tokenizer.encode(word,add_special_tokens=False)
        j=next((j for j in range(n,len(seq)-len(tokens)) if seq[j:j+len(tokens)]==tokens),None)
        if j is not None:positions.add(j+len(tokens)-2)
    return n,sorted(positions)


def snapshot(logits,batch,positions):
    n=int(batch['labels'].ne(-100).long().argmax(dim=1)[0])
    labels=batch['labels'][0,n:].to(logits.device);scores=[]
    for start in range(0,len(labels),32):
        end=min(start+32,len(labels));x=logits[0,n+start-1:n+end-1].float()
        scores.append((-torch.nn.functional.cross_entropy(x,labels[start:end],reduction='none')).cpu())
    return dict(input_ids=batch['input_ids'].detach().cpu().clone(),labels=batch['labels'].detach().cpu().clone(),
        positions=positions,selected_logits=logits[0,positions].detach().float().cpu().clone(),
        target_logprobs=torch.cat(scores),prefix_tokens=n,
        input_ids_sha256=digest_tensor(batch['input_ids']),labels_sha256=digest_tensor(batch['labels']),
        input_feature_sha256=digest_tensor(batch['input_features']),
        feature_mask_sha256=digest_tensor(batch['feature_attention_mask']))


def compare(reference,candidate,target_ids=None):
    reference=reference.float();candidate=candidate.float()
    if reference.shape!=candidate.shape or reference.ndim!=2:raise ValueError('Aligned [positions,vocab] required')
    r=reference.log_softmax(-1);c=candidate.log_softmax(-1);rp=r.exp();cp=c.exp()
    delta=(candidate-reference).abs();centered=((candidate-candidate.mean(-1,keepdim=True))-(reference-reference.mean(-1,keepdim=True))).abs()
    top=reference.topk(2,-1);ctop=candidate.topk(2,-1)
    records=[]
    for i in range(len(reference)):
        rec=dict(max_logit_abs_difference=float(delta[i].max()),mean_logit_abs_difference=float(delta[i].mean()),
            max_centered_logit_abs_difference=float(centered[i].max()),
            total_variation=float((rp[i]-cp[i]).abs().sum()/2),
            kl_reference_candidate=float((rp[i]*(r[i]-c[i])).sum()),
            reference_top1=int(top.indices[i,0]),candidate_top1=int(ctop.indices[i,0]),
            top1_equal=bool(top.indices[i,0]==ctop.indices[i,0]),reference_top1_margin=float(top.values[i,0]-top.values[i,1]),
            candidate_top1_margin=float(ctop.values[i,0]-ctop.values[i,1]))
        if target_ids is not None:rec.update(target_id=int(target_ids[i]),target_logprob_difference=float(c[i,target_ids[i]]-r[i,target_ids[i]]))
        records.append(rec)
    return records


def save_new(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve previous numeric evidence')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+chr(10))


def fixed_sequence_cache_logits(adapter,batch,positions):
    model=adapter.model
    n=int(batch['labels'].ne(-100).long().argmax(dim=1)[0])
    prefix={k:v for k,v in batch.items() if k!='labels'}
    for key in ['input_ids','attention_mask','token_type_ids','mm_token_type_ids']:
        if key in prefix:prefix[key]=prefix[key][:,:n]
    prefix=adapter._move(prefix)
    output=model(**prefix,use_cache=True);past=output.past_key_values
    selected={n-1:output.logits[0,-1].detach().float().cpu().clone()};del output
    native=model.get_base_model() if hasattr(model,'get_base_model') else model
    delta=native.rope_deltas.detach().clone()
    for position in range(n,batch['input_ids'].shape[1]-1):
        token=batch['input_ids'][:,position:position+1].to(next(model.parameters()).device)
        pos=torch.full((3,1,1),position,dtype=torch.long,device=token.device)+delta.to(token.device).unsqueeze(0)
        output=model(input_ids=token,position_ids=pos,attention_mask=batch['attention_mask'][:,:position+1].to(token.device),past_key_values=past,use_cache=True)
        past=output.past_key_values
        if position in positions:selected[position]=output.logits[0,-1].detach().float().cpu().clone()
        del output
    del past
    return torch.stack([selected[p] for p in positions])
