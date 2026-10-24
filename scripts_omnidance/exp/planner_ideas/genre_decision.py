"""Independent genre-decision experiment; the fair planner protocol is unchanged.

The cache stores contextualized audio-token states from INPUT-ONLY native prefill,
not raw music-encoder features and never teacher-forced answer states.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts_omnidance.exp.ms.music_planner_adapters import (
    FAMILIES, TASK_INSTRUCTION, PlannerAdapter, load_planner_adapter,
)

CACHE_SCHEMA = 'planner-genre-audio-prefill-v1'
HEAD_SCHEMA = 'planner-genre-head-v1'
CONDITION_PROTOCOL = 'planner-selected-genre-prompt-v1'
SUPPORTED_CONDITION_FAMILIES = ('moss', 'music_flamingo')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding='utf-8')


def validate_classes(classes):
    if (not isinstance(classes, list) or len(classes) < 2 or
            any(not isinstance(x, str) or not x.strip() or '\n' in x or
                '<' in x or '>' in x or len(x) > 80 for x in classes) or
            len(set(classes)) != len(classes)):
        raise ValueError('Classes must be at least two unique plain genre names, <=80 characters')
    return classes


def validate_model_identity(value):
    if (not isinstance(value, str) or not value.strip() or
            value.strip().lower() in ('latest', 'main', 'master', 'head', 'default', 'preview') or
            any(part in value.lower() for part in ('-latest', ':latest', '/latest', '@main', '@head'))):
        raise ValueError('model-id must declare an immutable base+adapter identity, never latest/main')
    return value


def adapter_code_sha():
    return sha256(sys.modules[PlannerAdapter.__module__].__file__)


class GenreHead(nn.Module):
    """Masked pooling followed by an FP32 classification head."""
    def __init__(self, feature_dim, n_classes, hidden_dim=128):
        super().__init__()
        if min(feature_dim, n_classes, hidden_dim) <= 0 or n_classes < 2:
            raise ValueError('Invalid genre-head dimensions')
        self.config = dict(feature_dim=feature_dim, n_classes=n_classes, hidden_dim=hidden_dim)
        self.network = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
                                     nn.GELU(), nn.Linear(hidden_dim, n_classes))

    def forward(self, features, mask):
        if (features.ndim != 3 or mask.shape != features.shape[:2] or
                mask.dtype != torch.bool or features.shape[-1] != self.config['feature_dim']):
            raise ValueError('Require features [B,T,D] and boolean valid-token mask [B,T]')
        if not mask.any(-1).all() or not torch.isfinite(features[mask]).all():
            raise ValueError('Each sample needs finite valid features and a nonempty mask')
        # masked_fill also ignores NaN/Inf padding; NaN*0 would not.
        pooled = features.float().masked_fill(~mask[..., None], 0).sum(1)
        pooled = pooled / mask.sum(1, keepdim=True)
        return self.network(pooled)


def genre_loss(logits, targets):
    """Hard CE or soft CE; targets must be honest mutually exclusive distributions."""
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('Logits must be finite [B,K]')
    if targets.ndim == 1:
        if targets.dtype != torch.long or targets.shape[0] != logits.shape[0]:
            raise ValueError('Hard targets must be int64 [B]')
        if ((targets < 0) | (targets >= logits.shape[-1])).any():
            raise ValueError('Hard target outside class dictionary')
        return F.cross_entropy(logits, targets)
    if (targets.shape != logits.shape or not torch.isfinite(targets).all() or
            (targets < 0).any() or not torch.allclose(targets.sum(-1),
            torch.ones_like(targets.sum(-1)), atol=1e-5, rtol=0)):
        raise ValueError('Soft targets must be finite nonnegative [B,K] summing to one')
    return -(targets * F.log_softmax(logits, -1)).sum(-1).mean()


def fit_temperature(logits, targets):
    """Fit one temperature on a disjoint calibration split, after model selection."""
    logits, targets = logits.detach().float(), targets.detach()
    genre_loss(logits, targets)
    log_t = torch.zeros((), requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([log_t], lr=.2, max_iter=80, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = genre_loss(logits / log_t.clamp(-4.6, 4.6).exp(), targets)
        loss.backward()
        return loss

    before = genre_loss(logits, targets).item()
    optimizer.step(closure)
    temperature = float(log_t.detach().clamp(-4.6, 4.6).exp())
    if genre_loss(logits / temperature, targets).item() > before:
        return 1.0
    return temperature


def metrics(logits, targets, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    probs = (logits / temperature).softmax(-1)
    target = F.one_hot(targets, logits.shape[-1]).float() if targets.ndim == 1 else targets
    confidence, predicted = probs.max(-1)
    # Soft-label argmax accuracy is only a diagnostic, not music/choreography quality.
    correct = predicted.eq(target.argmax(-1)).float()
    # For a soft target, expected correctness is q[predicted], not a hard argmax label.
    expected_correct = target.gather(1, predicted[:, None]).squeeze(1)
    ece = 0.
    for index in range(10):
        selected = (confidence > index / 10) & (confidence <= (index + 1) / 10)
        if selected.any():
            ece += float(selected.float().mean() *
                         (confidence[selected].mean() - expected_correct[selected].mean()).abs())
    return dict(nll=float(genre_loss(logits / temperature, targets)),
                brier=float(((probs - target) ** 2).sum(-1).mean()),
                argmax_accuracy=float(correct.mean()), ece_10_bins=ece,
                sources=len(logits), temperature=temperature)


def audio_only_row(row):
    """Whitelist input fields so no genre, target or video can reach prefill."""
    keys = ('id', 'source_group_id', 'audio_path', 'audio_sha256', 'audio_window_seconds')
    result = {key: row[key] for key in keys}
    if not isinstance(result['id'], str) or not isinstance(result['source_group_id'], str):
        raise ValueError('Source identity fields must be strings')
    return result


@torch.no_grad()
def export_prefill_features(adapter, row):
    batch = adapter.prefix(audio_only_row(row))
    if 'labels' in batch:
        raise ValueError('Feature extraction forbids teacher-forced labels')
    candidates = []
    config = getattr(adapter.model, 'config', None)
    thinker = getattr(config, 'thinker_config', None)
    for owner in (adapter.processor, config, thinker):
        for key in ('audio_token_id', 'audio_token_index'):
            value = getattr(owner, key, None)
            if value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError('Invalid native audio token identity')
                candidates.append(value)
    token = getattr(adapter.processor, 'audio_token', None)
    if token is not None:
        value = adapter.tokenizer.convert_tokens_to_ids(token)
        if type(value) is not int or value == getattr(adapter.tokenizer, 'unk_token_id', None):
            raise ValueError('Native audio token resolves to an unknown token')
        candidates.append(value)
    if not candidates or len(set(candidates)) != 1:
        raise ValueError('Missing or conflicting native audio token identity; no guessed pooling')
    token_id = candidates[0]
    mask = batch['input_ids'].eq(token_id) & batch['attention_mask'].bool()
    if not mask.any() or mask.shape[0] != 1:
        raise ValueError('Require exactly one native audio prefix with visible audio tokens')
    adapter.model.eval()
    parameter = next(adapter.model.parameters())
    with torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                        enabled=parameter.dtype == torch.bfloat16):
        result = adapter.model(**adapter._move(batch), output_hidden_states=True,
                               return_dict=True, use_cache=False)
    states = getattr(result, 'hidden_states', None)
    if states is None or not len(states):
        raise ValueError('Native model did not expose hidden_states; extraction unsupported')
    hidden = states[-1]
    if hidden.shape[:2] != mask.shape:
        raise ValueError('Native expanded hidden/input alignment differs; refuse guessed token mask')
    features = hidden[0, mask[0].to(hidden.device)].detach().float().cpu()
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError('Invalid audio-prefill features')
    return features, dict(prefix_tokens=batch['input_ids'].shape[1],
                          audio_tokens=len(features), target_consumed=False,
                          feature_kind='decoder-contextualized-native-audio-token-states',
                          native_instruction_present=adapter.family != 'omni_captioner')


class GenreConditionedPlanner(PlannerAdapter):
    """Explicit experiment protocol, never a mutation of the fair baseline adapter."""
    def __init__(self, base, genre, protocol):
        if protocol != CONDITION_PROTOCOL:
            raise ValueError('Explicit selected-genre protocol opt-in required')
        if base.family not in SUPPORTED_CONDITION_FAMILIES:
            raise ValueError('omni_captioner native audio-only template does not support genre prompts; '
                             'a separately trained conditioning-token protocol is not implemented')
        validate_classes([genre, '__validation_sentinel__'])
        super().__init__(base.family, base.model, base.processor, base.max_length)
        self.genre = genre
        self.trainable_info = base.trainable_info
        self._audio_identities = base._audio_identities

    def _prompt(self):
        instruction = (TASK_INSTRUCTION + '\nRequested dance style: ' + self.genre +
                       '. Perform the movements in this style. Output movement descriptions only; '
                       'the program will prepend the dance style separately.')
        if self.family == 'moss':
            return self.processor._build_default_prompt(instruction, has_audio=True)
        return self.processor.apply_chat_template([{'role': 'user', 'content': [
            {'type': 'audio', 'audio': 'native-waveform'}, {'type': 'text', 'text': instruction}]}],
            tokenize=False, add_generation_prompt=True)


def conditioned_collate(base, rows, selected_genre, protocol):
    """SFT integration: same prefix as conditioned generation, targets are actions only."""
    return GenreConditionedPlanner(base, selected_genre, protocol).collate(rows)


def cache_manifest(paths):
    result, identity = {}, None
    for path in paths:
        path = Path(path).resolve()
        for row in read_jsonl(path):
            current = row.get('extractor')
            if not isinstance(current, dict) or current.get('schema') != CACHE_SCHEMA:
                raise ValueError('Feature cache schema differs')
            if identity is None:
                identity = current
            if current != identity:
                raise ValueError('Mixed family/model/protocol feature caches are forbidden')
            if row['id'] in result:
                raise ValueError('Duplicate feature source id')
            item = dict(row)
            item['feature_path'] = str((path.parent / row['feature_path']).resolve())
            result[row['id']] = item
    if not result:
        raise ValueError('Empty feature cache')
    return result, identity


def join_labels(cache, path, classes, split):
    result, seen = [], set()
    for label in read_jsonl(path):
        if label.get('split') != split:
            raise ValueError(f'Expected a pure {split} label manifest')
        if not label.get('label_provenance') or not label.get('label_version'):
            raise ValueError('Independent genre labels require provenance and version; no caption guessing')
        if label['id'] in seen or label['id'] not in cache:
            raise ValueError('Duplicate label id or missing feature source')
        row = cache[label['id']]
        for key in ('source_group_id', 'audio_sha256', 'split'):
            if row.get(key) != label.get(key):
                raise ValueError(f'Label/feature identity differs: {key}')
        if ('genre' in label) == ('genre_distribution' in label):
            raise ValueError('Specify exactly one hard genre or soft genre_distribution')
        if 'genre' in label:
            if label['genre'] not in classes:
                raise ValueError('Unknown genre')
            target = [float(name == label['genre']) for name in classes]
        else:
            distribution = label['genre_distribution']
            if not isinstance(distribution, dict) or not set(distribution) <= set(classes):
                raise ValueError('Unknown genre in soft distribution')
            target = [distribution.get(name, 0.) for name in classes]
        target = torch.tensor(target, dtype=torch.float32)
        genre_loss(torch.zeros(1, len(classes)), target[None])
        result.append(dict(row, target=target))
        seen.add(label['id'])
    if not result:
        raise ValueError(f'Empty {split} labels')
    return result


def check_isolation(splits):
    identities = {}
    for name, rows in splits.items():
        identities[name] = {key: sorted({r[key] for r in rows})
                            for key in ('id', 'source_group_id', 'audio_sha256')}
    names = list(identities)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            for key in identities[left]:
                if set(identities[left][key]) & set(identities[right][key]):
                    raise ValueError(f'{left}/{right} split leakage: {key}')
    return identities


def load_features(row):
    if sha256(row['feature_path']) != row['feature_sha256']:
        raise ValueError('Feature file bytes changed')
    with np.load(row['feature_path'], allow_pickle=False) as data:
        features = torch.from_numpy(data['features'].copy()).float()
        mask = torch.from_numpy(data['mask'].copy())
    if features.ndim != 2 or mask.shape != features.shape[:1] or mask.dtype != torch.bool:
        raise ValueError('Invalid cached feature/mask shape')
    return features, mask


def padded_batch(rows):
    loaded = [load_features(row) for row in rows]
    dims = {x.shape[-1] for x, _ in loaded}
    if len(dims) != 1:
        raise ValueError('Mixed feature dimensions')
    features = torch.zeros(len(rows), max(len(x) for x, _ in loaded), dims.pop())
    mask = torch.zeros(features.shape[:2], dtype=torch.bool)
    for i, (values, valid) in enumerate(loaded):
        features[i, :len(values)] = values
        mask[i, :len(values)] = valid
    return features, mask


@torch.no_grad()
def evaluate_rows(head, rows, batch_size=32):
    head.eval()
    parts = [head(*padded_batch(rows[i:i + batch_size]))
             for i in range(0, len(rows), batch_size)]
    return torch.cat(parts), torch.stack([row['target'] for row in rows])


def save_head(path, head, classes, extractor, temperature, evidence):
    validate_classes(classes)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid temperature')
    torch.save(dict(schema=HEAD_SCHEMA, config=head.config, classes=classes,
                    extractor=extractor, temperature=temperature,
                    evidence=evidence, state_dict=head.state_dict()), path)


def load_head(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema') != HEAD_SCHEMA:
        raise ValueError('Genre checkpoint schema differs')
    validate_classes(saved['classes'])
    if saved['config']['n_classes'] != len(saved['classes']):
        raise ValueError('Checkpoint class dimensions differ')
    if not math.isfinite(saved['temperature']) or saved['temperature'] <= 0:
        raise ValueError('Invalid checkpoint temperature')
    head = GenreHead(**saved['config'])
    head.load_state_dict(saved['state_dict'], strict=True)
    return head.eval(), saved


@torch.no_grad()
def decide(head, features, mask, classes, temperature):
    probabilities = (head(features, mask) / temperature).softmax(-1)
    return [dict(selected_genre=classes[int(p.argmax())],
                 probabilities=dict(zip(classes, p.tolist())),
                 max_probability=float(p.max())) for p in probabilities]


def run_export(args):
    validate_model_identity(args.model_id)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'features.jsonl'
    if manifest_path.exists():
        raise ValueError('Refuse to overwrite an existing feature manifest')
    adapter = load_planner_adapter(args.family, args.model_path, checkpoint=args.checkpoint,
        device=args.device, dtype=getattr(torch, args.dtype))
    identity = dict(schema=CACHE_SCHEMA, family=args.family, model_identity=args.model_id,
                    prefix_policy='native-input-only-no-answer-or-genre',
                    code_sha256=sha256(__file__), adapter_code_sha256=adapter_code_sha())
    seen, records = set(), []
    for row in read_jsonl(args.sources):
        source = audio_only_row(row)
        if source['id'] in seen or row.get('split') not in ('train', 'val', 'cal', 'test', 'predict'):
            raise ValueError('Duplicate source or unknown explicit split')
        if not Path(source['audio_path']).is_absolute():
            source['audio_path'] = str((args.sources.resolve().parent / source['audio_path']).resolve())
        features, info = export_prefill_features(adapter, source)
        name = hashlib.sha256(source['id'].encode()).hexdigest() + '.npz'
        path = out / name
        if path.exists():
            raise ValueError('Refuse to overwrite existing feature payload')
        np.savez_compressed(path, features=features.numpy(), mask=np.ones(len(features), dtype=np.bool_))
        records.append(dict(id=source['id'], source_group_id=source['source_group_id'],
                            audio_sha256=source['audio_sha256'], split=row['split'],
                            feature_path=name, feature_sha256=sha256(path),
                            extractor=identity, extraction=info))
        seen.add(source['id'])
    if not records:
        raise ValueError('Empty export source list')
    manifest_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')


def run_train(args):
    if args.epochs <= 0 or args.batch_size <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Positive epochs/batch-size/lr required')
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    classes = validate_classes(json.loads(args.classes.read_text(encoding='utf-8')))
    cache, extractor = cache_manifest(args.features)
    splits = {name: join_labels(cache, getattr(args, name + '_labels'), classes, name)
              for name in ('train', 'val', 'cal')}
    isolation = check_isolation(splits)
    first, _ = load_features(splits['train'][0])
    head = GenreHead(first.shape[-1], len(classes), args.hidden_dim)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    best, best_state, history = float('inf'), None, []
    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(len(splits['train'])).tolist()
        for start in range(0, len(order), args.batch_size):
            rows = [splits['train'][i] for i in order[start:start + args.batch_size]]
            loss = genre_loss(head(*padded_batch(rows)), torch.stack([r['target'] for r in rows]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logits, target = evaluate_rows(head, splits['val'], args.batch_size)
        report = metrics(logits, target)
        history.append(dict(epoch=epoch + 1, validation=report))
        if report['nll'] < best:
            best = report['nll']
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    logits, target = evaluate_rows(head, splits['cal'], args.batch_size)
    temperature = fit_temperature(logits, target)
    evidence = dict(seed=args.seed, history=history, isolation=isolation,
        calibration_before=metrics(logits, target), calibration_after=metrics(logits, target, temperature),
        label_manifests={name: sha256(getattr(args, name + '_labels')) for name in splits},
        test_used=False, gpu_training_performed=False,
        limitation='Genre-head experiment only; no choreography or renderer quality claim')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    save_head(args.out_dir / 'genre_head.pt', head, classes, extractor, temperature, evidence)
    write_json(args.out_dir / 'training_report.json', evidence)


def run_predict(args):
    head, saved = load_head(args.head)
    cache, identity = cache_manifest(args.features)
    if identity != saved['extractor']:
        raise ValueError('Prediction feature extractor differs from training')
    rows = list(cache.values())
    if args.final_test:
        if any(row['split'] != 'test' for row in rows):
            raise ValueError('Final test accepts a pure test feature manifest')
        for previous in saved['evidence']['isolation'].values():
            for key in ('id', 'source_group_id', 'audio_sha256'):
                if set(previous[key]) & {r[key] for r in rows}:
                    raise ValueError(f'Final-test leakage: {key}')
    records = []
    for row in rows:
        features, mask = padded_batch([row])
        prediction = decide(head, features, mask, saved['classes'], saved['temperature'])[0]
        records.append(dict(id=row['id'], split=row['split'], **prediction))
    args.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')


def run_generate(args):
    validate_model_identity(args.model_id)
    if args.family not in SUPPORTED_CONDITION_FAMILIES:
        raise ValueError('omni_captioner genre-conditioning protocol is unsupported')
    head, saved = load_head(args.head)
    if (saved['extractor']['family'] != args.family or
            saved['extractor']['model_identity'] != args.model_id or
            saved['extractor']['code_sha256'] != sha256(__file__) or
            saved['extractor'].get('adapter_code_sha256') != adapter_code_sha()):
        raise ValueError('Generation model/code identity differs from genre-feature training')
    base = load_planner_adapter(args.family, args.model_path, checkpoint=args.checkpoint,
                               device=args.device, dtype=getattr(torch, args.dtype))
    records = []
    for row in read_jsonl(args.sources):
        source = audio_only_row(row)
        if not Path(source['audio_path']).is_absolute():
            source['audio_path'] = str((args.sources.resolve().parent / source['audio_path']).resolve())
        features, extraction = export_prefill_features(base, source)
        prediction = decide(head, features[None], torch.ones(1, len(features), dtype=torch.bool),
                            saved['classes'], saved['temperature'])[0]
        adapter = GenreConditionedPlanner(base, prediction['selected_genre'], args.protocol)
        generation = adapter.generate(source, max_new_tokens=args.max_new_tokens)
        records.append(dict(id=source['id'], protocol=CONDITION_PROTOCOL, extractor=saved['extractor'], **prediction,
            caption_text='Dance style: ' + prediction['selected_genre'] + '.\n' + generation['raw_text'],
            action_generation=generation, extraction=extraction,
            conditioning_sft_performed=False))
    args.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    for name, runner in (('export', run_export), ('generate', run_generate)):
        sub = commands.add_parser(name)
        sub.set_defaults(run=runner)
        sub.add_argument('--family', choices=FAMILIES, required=True)
        sub.add_argument('--model-path', type=Path, required=True)
        sub.add_argument('--checkpoint', type=Path)
        sub.add_argument('--model-id', required=True, help='Immutable base+adapter revision identity; never latest')
        sub.add_argument('--sources', type=Path, required=True)
        sub.add_argument('--device', default='cuda')
        sub.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
        if name == 'export':
            sub.add_argument('--out-dir', type=Path, required=True)
        else:
            sub.add_argument('--head', type=Path, required=True)
            sub.add_argument('--out', type=Path, required=True)
            sub.add_argument('--protocol', choices=(CONDITION_PROTOCOL,), required=True)
            sub.add_argument('--max-new-tokens', type=int, default=1024)
    train = commands.add_parser('train')
    train.set_defaults(run=run_train)
    train.add_argument('--features', type=Path, nargs='+', required=True)
    train.add_argument('--classes', type=Path, required=True)
    for split in ('train', 'val', 'cal'):
        train.add_argument('--' + split + '-labels', type=Path, required=True)
    train.add_argument('--out-dir', type=Path, required=True)
    train.add_argument('--epochs', type=int, default=20)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--hidden-dim', type=int, default=128)
    train.add_argument('--lr', type=float, default=1e-3)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--cpu-threads', type=int, default=2)
    predict = commands.add_parser('predict')
    predict.set_defaults(run=run_predict)
    predict.add_argument('--features', type=Path, nargs='+', required=True)
    predict.add_argument('--head', type=Path, required=True)
    predict.add_argument('--out', type=Path, required=True)
    predict.add_argument('--final-test', action='store_true')
    return result


if __name__ == '__main__':
    arguments = parser().parse_args()
    arguments.run(arguments)
