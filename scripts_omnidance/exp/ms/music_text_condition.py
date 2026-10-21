"""Raw H3 text-tail replacement; oracle captions are supervision, not music inference."""
import hashlib
import json
import os
from pathlib import Path

import torch

from diffsynth.models.minimax_h3_text_encoder import _text_ids
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder


PURE_TEXT_TEACHER_KIND = 'pure_text_h3_raw_tokens'
PURE_TEXT_BINDING_KIND = 'pure_text_teacher_case_binding'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prefix_length(tokenizer, prompt, input_ids):
    tail_ids = torch.tensor(_text_ids(tokenizer, prompt), dtype=torch.long)
    ids = input_ids.detach().cpu().reshape(-1)
    if not len(tail_ids) or len(ids) <= len(tail_ids):
        raise ValueError('Require a nonempty presentation prefix and prompt tail')
    if not torch.equal(ids[-len(tail_ids):], tail_ids):
        raise ValueError('Prompt tokens are not an exact suffix of presentation')
    return len(ids) - len(tail_ids)


def replace_tail(prefix, prefix_tags, tail):
    if prefix.ndim != 2 or tail.ndim != 2 or prefix.shape[1] != 5120 or tail.shape[1] != 5120:
        raise ValueError('Expected raw [tokens,5120] prefix and tail')
    if not len(prefix) or not len(tail) or prefix_tags.shape != (len(prefix),):
        raise ValueError('Empty or inconsistent text-tail layout')
    if not torch.isfinite(prefix).all() or not torch.isfinite(tail).all():
        raise ValueError('Nonfinite raw text condition')
    if not torch.all((prefix_tags == 0) | (prefix_tags == 1)):
        raise ValueError('Invalid prefix presentation tags')
    hidden = torch.cat((prefix, tail.to(prefix)), dim=0)
    tags = torch.cat((prefix_tags, torch.ones(len(tail), dtype=prefix_tags.dtype, device=prefix_tags.device)))
    return dict(prompt_embeds=hidden, text_token_tags=tags)


def difference(first, second):
    if first.shape != second.shape:
        raise ValueError('Cannot compare different condition shapes')
    a, b = first.float(), second.float()
    delta = a - b
    rms = delta.square().mean().sqrt()
    return dict(exact=torch.equal(first, second), max_abs=float(delta.abs().max()),
                rms=float(rms), relative_rms=float(rms / b.square().mean().sqrt().clamp_min(1e-12)))


def write_new(path, value, binary=False):
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    # Exclusive output creation also rejects concurrent writers and failed partial outputs.
    with path.open('xb' if binary else 'x') as stream:
        if binary:
            torch.save(value, stream)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


class OracleTextTailEmbedder(MiniMaxH3Unit_PromptEmbedder):
    def __init__(self, cache_dir, prefix_tolerance=0.002):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(self.cache_dir)
        self.prefix_tolerance = prefix_tolerance
        self.teacher_prompt = None
        self.case = None
        self.teacher_output = None
        self.report = None
        self.report_path = None

    def set_case(self, teacher_prompt, metadata):
        if not isinstance(teacher_prompt, str) or not teacher_prompt.strip():
            raise ValueError('Oracle requires a declared supervision caption')
        self.teacher_prompt = teacher_prompt
        self.case = metadata
        self.teacher_output = None
        self.report = None

    def process(self, pipe, prompt, keyframes=None, ref_blocks=None, height=None, width=None, text_embedding=None):
        if prompt != ' ' or text_embedding is not None or keyframes:
            raise ValueError('Oracle student requires only reference image structure, no semantic prompt')
        if not ref_blocks or self.case is None:
            raise ValueError('Oracle requires reference blocks and case metadata')
        student = super().process(pipe, prompt, ref_blocks=ref_blocks, height=height, width=width)
        student_ids = self.preprocess_ref_blocks(pipe, prompt, ref_blocks)[0]
        lp = prefix_length(pipe.tokenizer, prompt, student_ids)
        teacher = super().process(pipe, self.teacher_prompt, ref_blocks=ref_blocks, height=height, width=width)
        teacher_ids = self.preprocess_ref_blocks(pipe, self.teacher_prompt, ref_blocks)[0]
        teacher_lp = prefix_length(pipe.tokenizer, self.teacher_prompt, teacher_ids)
        if lp != teacher_lp or not torch.equal(student_ids[:lp], teacher_ids[:lp]):
            raise ValueError('Teacher/student presentation prefixes differ')
        prefix_tags = teacher['text_token_tags'][:lp]
        if not torch.equal(prefix_tags, student['text_token_tags'][:lp]):
            raise ValueError('Teacher/student prefix tags differ')
        tail = teacher['prompt_embeds'][lp:]
        rebuilt = replace_tail(teacher['prompt_embeds'][:lp], prefix_tags, tail)
        identity = difference(rebuilt['prompt_embeds'], teacher['prompt_embeds'])
        if not identity['exact'] or not torch.equal(rebuilt['text_token_tags'], teacher['text_token_tags']):
            raise ValueError('Same-forward identity text reconstruction failed')
        prefix_check = difference(student['prompt_embeds'][:lp], teacher['prompt_embeds'][:lp])
        self.report = dict(version=1, kind='oracle_teacher_hidden_not_music_prediction', case=self.case,
                           semantic_student_prompt=prompt, teacher_prompt=self.teacher_prompt,
                           prefix_length=lp, tail_length=len(tail), identity_reconstruction=identity,
                           independent_prefix=prefix_check, prefix_tolerance=self.prefix_tolerance,
                           teacher_hidden_sha256=tensor_sha256(teacher['prompt_embeds']),
                           teacher_tail_sha256=tensor_sha256(tail), teacher_ids=teacher_ids.tolist(),
                           interface='raw_h3_last_hidden_5120_tail_replacement', packed_checked=False,
                           dit_prediction_checked=False, native_audio_condition='fixed_silence')
        stem = self.case['sample_id'] + '_' + self.case['audio_id']
        if Path(stem).name != stem:
            raise ValueError('Invalid case identifiers')
        self.report_path = self.cache_dir / (stem + '.json')
        if prefix_check['relative_rms'] > self.prefix_tolerance:
            self.report['status'] = 'failed_prefix_equivalence'
            write_new(self.report_path, self.report)
            raise ValueError(f"Independent structural prefix differs: {prefix_check}")
        output = replace_tail(student['prompt_embeds'][:lp], prefix_tags, tail)
        self.teacher_output = teacher
        cache = dict(metadata=self.report.copy(), prefix_context=student['prompt_embeds'][lp - 1].detach().cpu(),
                     tail=tail.detach().cpu(), prefix_tags=prefix_tags.detach().cpu(),
                     token_ids=teacher_ids.detach().cpu())
        write_new(self.cache_dir / (stem + '.pt'), cache, binary=True)
        print('[music text oracle] ' + json.dumps(self.report, ensure_ascii=False, allow_nan=False), flush=True)
        return output

    def finish_audit(self, packed_check, prediction_check, native_audio):
        if self.report is None or self.report_path is None:
            raise RuntimeError('No oracle output to audit')
        self.report.update(packed_checked=bool(packed_check), dit_prediction_checked=True,
                           dit_prediction=prediction_check, native_audio=native_audio)
        passed = packed_check and prediction_check['relative_rms'] <= self.prefix_tolerance
        self.report['status'] = 'passed' if passed else 'failed_condition_equivalence'
        write_new(self.report_path, self.report)
        self.teacher_output = None
        if not passed:
            raise ValueError('Oracle packed/DiT equivalence gate failed')


class ProjectedTeacherTailEmbedder(MiniMaxH3Unit_PromptEmbedder):
    """Explicit same-length teacher-space diagnostic, never a trained music path."""
    def __init__(self, manifest_path):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.manifest = json.loads(self.manifest_path.read_text())
        if (self.manifest.get('complete') is not True or self.manifest.get('diagnostic_only') is not True
                or self.manifest.get('format_version') != 1
                or self.manifest.get('kind') != 'projected_teacher_not_music_prediction'
                or self.manifest.get('teacher_length_for_oracle_only') is not True
                or self.manifest.get('changes_student_deployment') is not False
                or len(self.manifest.get('cases', {})) != 3):
            raise ValueError('Require a complete explicit three-person projected-teacher manifest')
        self.record = None
        self.original_output = None
        self.report = None

    def set_case(self, sample_id, image_path, audio_path):
        entry = self.manifest['cases'][sample_id]
        if (str(Path(image_path)) != entry['image_path'] or str(Path(audio_path)) != entry['audio_path']
                or hashlib.sha256(Path(image_path).read_bytes()).hexdigest() != entry['image_sha256']):
            raise ValueError('Projected teacher is not bound to the current reference and music case')
        path = Path(entry['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['file_sha256']:
            raise ValueError('Projected raw tail file changed')
        source = Path(entry['oracle_source'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry['oracle_sha256']:
            raise ValueError('Source oracle changed since projection')
        record = torch.load(path, map_location='cpu', weights_only=True)
        if (record['sample_id'] != sample_id or record['variant'] != entry['variant']
                or record['tail'].shape != record['original_tail'].shape
                or len(record['tail']) != entry['teacher_tokens']
                or tensor_sha256(record['original_tail']) != entry['original_teacher_tail_sha256']
                or hashlib.sha256(record['tail'].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    != entry['projected_tail_sha256']):
            raise ValueError('Projected teacher tail length, identity or source hash mismatch')
        self.record = record
        self.report = dict(sample_id=sample_id, variant=entry['variant'], manifest_sha256=self.manifest_sha256,
            projected_file_sha256=entry['file_sha256'], oracle_source_sha256=entry['oracle_sha256'],
            projection_tail_sha256=tensor_sha256(record['tail']), teacher_tokens=entry['teacher_tokens'],
            image_path=str(image_path), audio_path=str(audio_path), teacher_length_for_diagnostic_only=True,
            music_prediction=False, semantic_student_prompt=' ', original_packed_layout_checked=False)
        self.original_output = None

    def process(self, pipe, prompt, keyframes=None, ref_blocks=None, height=None, width=None, text_embedding=None):
        if (prompt != ' ' or self.record is None or not ref_blocks or keyframes or text_embedding is not None):
            raise ValueError('Projection diagnostic requires empty semantic text, current reference and a declared case')
        encoded = super().process(pipe, prompt, ref_blocks=ref_blocks, height=height, width=width)
        ids, tags = self.preprocess_ref_blocks(pipe, prompt, ref_blocks)[:2]
        lp = prefix_length(pipe.tokenizer, prompt, ids)
        if (not torch.equal(ids[:lp].cpu(), self.record['prefix_ids'])
                or not torch.equal(tags[:lp].cpu(), self.record['prefix_tags'])
                or not torch.equal(encoded['prompt_embeds'][lp - 1].cpu(), self.record['prefix_context'])):
            raise ValueError('Actual empty reference prefix differs from the source oracle')
        prefix = encoded['prompt_embeds'][:lp]
        prefix_tags = encoded['text_token_tags'][:lp]
        self.original_output = replace_tail(prefix, prefix_tags, self.record['original_tail'])
        output = replace_tail(prefix, prefix_tags, self.record['tail'])
        if (output['prompt_embeds'].shape != self.original_output['prompt_embeds'].shape
                or not torch.equal(output['text_token_tags'], self.original_output['text_token_tags'])):
            raise ValueError('Projected teacher changed length or modality tags')
        self.report.update(prefix_ids_tags_context_exact=True, prefix_length=lp,
                           prefix_sha256=tensor_sha256(prefix))
        return output

    def finish_layout_audit(self, packed_equal, native_audio):
        if not packed_equal or self.original_output is None:
            raise ValueError('Projected teacher changed the original packed layout')
        self.report.update(original_packed_layout_checked=True, original_packed_exact=True,
                           native_audio=native_audio,
                           scope='Teacher projection with identical length/layout, not identity-prediction equivalence or music deployment')
        print('[music text projection] ' + json.dumps(self.report, allow_nan=False), flush=True)


def neutralize_gender_pronouns(text):
    import re
    possessive_nouns = {'hand', 'hands', 'arm', 'arms', 'elbow', 'elbows', 'shoulder', 'shoulders',
                        'foot', 'feet', 'leg', 'legs', 'knee', 'knees', 'ankle', 'ankles',
                        'hip', 'hips', 'torso', 'body', 'head', 'neck', 'face', 'waist', 'chest',
                        'left', 'right', 'both', 'back', 'pose', 'position', 'stance', 'weight'}
    changes = []

    def replace(match):
        original = match.group()
        word = original.lower()
        following = re.match(r'\s+([A-Za-z]+)', text[match.end():])
        possessive = word == 'his' or (word == 'her' and following is not None
                                       and following[1].lower() in possessive_nouns)
        replacement = "the dancer's" if possessive else 'the dancer'
        if original[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        changes.append(dict(start=match.start(), end=match.end(), original=original, replacement=replacement))
        return replacement

    normalized = re.sub(r'\b(?:he|she|his|her|him|himself|herself)\b', replace, text, flags=re.IGNORECASE)
    return normalized, changes


def build_oracle_captions(exp_root, out, neutral_teacher=False):
    import csv
    root = Path(exp_root)
    genres = {entry['sid']: entry['label']['genre'] for entry in
              (json.loads(line) for line in (root / 'MS/vlm/eval3_32b.jsonl').read_text().splitlines() if line.strip())}
    vlm_path = root / 'E10/val47_motion_captions.csv'
    with vlm_path.open(newline='') as stream:
        vlm = {row['sample_id']: row['motion_caption'] for row in csv.DictReader(stream)}
    captions = {}
    people = None
    for condition in ('joint', 'aswap'):
        with (root / 'E2' / ('full_' + condition + '.csv')).open(newline='') as stream:
            rows = list(csv.DictReader(stream))[:3]
        ids = [row['sample_id'] for row in rows]
        if len(ids) != 3 or (people is not None and ids != people):
            raise ValueError('Require the same three source people in both conditions')
        people = ids
        for row in rows:
            audio_id = Path(row['audio_path']).stem
            ms_path = root / 'MS/ms_orig_eval3' / (audio_id + '.json')
            if ms_path.is_file():
                original = json.loads(ms_path.read_text())
                caption = ' '.join(sentence for sentence in original['ms'] if sentence)
                source = dict(kind='original_MotionScript', path=str(ms_path),
                              sha256=hashlib.sha256(ms_path.read_bytes()).hexdigest())
            else:
                caption = vlm[audio_id]
                source = dict(kind='video_observed_VLM_fallback_missing_original_MS', path=str(vlm_path),
                              sha256=hashlib.sha256(vlm_path.read_bytes()).hexdigest())
            if not caption.strip():
                raise ValueError('Empty supervision caption: ' + audio_id)
            if audio_id in genres:
                caption = f'Dance style: {genres[audio_id]}. ' + caption
            original_caption = caption
            changes = []
            if neutral_teacher:
                caption, changes = neutralize_gender_pronouns(caption)
            captions[audio_id] = dict(prompt=caption, source=source, original_prompt=original_caption,
                                      teacher_transform='gender_pronouns_to_dancer_v1' if neutral_teacher else 'none',
                                      replacements=changes,
                                      caveat='Paired dance description, not unique choreography required by music')
    if len(captions) != 6:
        raise ValueError('Expected six distinct original/swap music sources')
    write_new(out, captions)
    print(json.dumps(dict(oracle_music_sources=len(captions), source_people=people,
                          output=str(out), supervision_only=True)), flush=True)


def validate_pure_text_record(record, sample_id=None):
    if (not isinstance(record, dict) or not isinstance(record.get('metadata'), dict)
            or set(record) != {'metadata', 'tail', 'source_token_ids', 'teacher_length'}):
        raise ValueError('Pure-text teacher requires exact raw-tail/token-ID/length fields and no prefix context')
    metadata, tail, ids = record['metadata'], record['tail'], record['source_token_ids']
    if (metadata.get('kind') != PURE_TEXT_TEACHER_KIND or metadata.get('text_only') is not True
            or metadata.get('image_conditioned') is not False
            or type(metadata.get('prefix_length')) is not int or metadata['prefix_length'] != 0
            or (sample_id is not None and metadata.get('sample_id') != sample_id)
            or any(name in metadata for name in ('prefix_context', 'prefix_tags', 'prepared_image_size', 'context_sha256'))):
        raise ValueError('Pure-text teacher must declare caption-only encoding with zero reference prefix/context')
    length = record['teacher_length']
    if (type(length) is not int or length < 1 or type(metadata.get('teacher_length')) is not int
            or metadata['teacher_length'] != length or type(metadata.get('tail_length')) is not int
            or metadata['tail_length'] != length or not torch.is_tensor(tail)
            or tail.dtype != torch.bfloat16 or tail.shape != (length, 5120)
            or not torch.isfinite(tail).all() or not torch.is_tensor(ids)
            or ids.dtype != torch.long or ids.shape != (length,) or torch.any(ids < 0)):
        raise ValueError('Pure-text teacher raw BF16 tail and source token IDs must exactly match teacher length')
    prompt = metadata.get('prompt')
    if (not isinstance(prompt, str) or not prompt.strip()
            or hashlib.sha256(prompt.encode()).hexdigest() != metadata.get('caption_sha256')
            or tensor_sha256(tail) != metadata.get('tail_sha256')
            or tensor_sha256(ids) != metadata.get('source_token_ids_sha256')
            or tensor_sha256(ids) != metadata.get('token_ids_sha256')):
        raise ValueError('Pure-text teacher caption, raw-tail or source-token-ID SHA mismatch')
    if metadata.get('case') is not None and tensor_sha256(tail) != metadata.get('teacher_tail_sha256'):
        raise ValueError('Bound pure-text teacher tail SHA mismatch')
    return metadata


def bind_pure_text_cases(cache_dir, exp_root, out):
    import csv
    cache, root, output = Path(cache_dir), Path(exp_root), Path(out)
    if output.exists() or not output.parent.is_dir() or not cache.is_dir():
        raise ValueError('Binding requires a completed source directory and a new output with an existing parent')
    csv_path = root / 'E2/full_joint.csv'
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))[:3]
    if len(rows) != 3 or len({row['sample_id'] for row in rows}) != 3:
        raise ValueError('Require the fixed three distinct source-person joint cases')
    summary_path, protocol_path = cache / 'finished.json', cache / 'plan.json'
    summary = json.loads(summary_path.read_text())
    source = summary.get('source', {})
    protocol_sha = file_sha256(protocol_path)
    if (summary.get('complete') is not True or source.get('kind') != PURE_TEXT_TEACHER_KIND
            or source.get('text_only') is not True or source.get('image_conditioned') is not False
            or type(source.get('prefix_length')) is not int or source['prefix_length'] != 0
            or summary.get('source_protocol_sha256') != protocol_sha
            or json.loads(protocol_path.read_text()) != source):
        raise ValueError('Require completed pure-text source cache with exact manifest/protocol provenance')
    entries = summary.get('rows', [])
    declared = {entry['sample_id']: entry for entry in entries}
    if len(declared) != len(entries) or len(entries) != summary.get('expected'):
        raise ValueError('Duplicate or inconsistent pure-text manifest records')
    prepared = []
    for row in rows:
        sid, audio_id = row['sample_id'], Path(row['audio_path']).stem
        stem = sid + '_' + audio_id
        if Path(stem).name != stem or sid in ('.', '..') or sid not in declared:
            raise ValueError('Invalid or undeclared pure-text teacher case IDs')
        path = cache / (sid + '.pt')
        entry = declared[sid]
        file_sha = file_sha256(path)
        if Path(entry['path']).resolve() != path.resolve() or entry['file_sha256'] != file_sha:
            raise ValueError('Pure-text source case path or file SHA changed')
        record = torch.load(path, map_location='cpu', weights_only=True)
        metadata = validate_pure_text_record(record, sid).copy()
        if (metadata.get('source_protocol_sha256') != protocol_sha
                or Path(metadata['source_protocol_path']).resolve() != protocol_path.resolve()
                or source.get('lengths', {}).get(sid) != record['teacher_length']
                or source.get('token_ids_sha256', {}).get(sid) != metadata['source_token_ids_sha256']
                or entry.get('teacher_length') != record['teacher_length']):
            raise ValueError('Pure-text source token IDs/length/protocol differ from the completed manifest')
        case = dict(sample_id=sid, audio_id=audio_id, image_path=row['image_path'],
                    audio_path=row['audio_path'], condition='joint')
        metadata.update(case=case, teacher_tail_sha256=tensor_sha256(record['tail']),
                        source_cache_file=str(path), source_cache_file_sha256=file_sha,
                        source_scope='caption_only_no_reference_context', teacher_only_diagnostic=True,
                        teacher_length_for_diagnostic_only=True, reference_image_binding_only=True,
                        source_manifest_path=str(summary_path), source_manifest_sha256=file_sha256(summary_path))
        prepared.append((stem, dict(record, metadata=metadata), case))
    output.mkdir()
    bound_cache = output / 'cache'
    bound_cache.mkdir()
    bound_rows = []
    for stem, record, case in prepared:
        path = bound_cache / (stem + '.pt')
        write_new(path, record, binary=True)
        bound_rows.append(dict(case=case, path=str(path), file_sha256=file_sha256(path),
                               teacher_length=record['teacher_length'],
                               teacher_tail_sha256=record['metadata']['teacher_tail_sha256']))
    report = dict(kind=PURE_TEXT_BINDING_KIND, complete=True, expected=3, rows=bound_rows,
                  text_only=True, image_conditioned=False, prefix_length=0,
                  source_scope='caption_only_no_reference_context', source=dict(kind=PURE_TEXT_TEACHER_KIND,
                      cache_dir=str(cache), manifest_path=str(summary_path), manifest_sha256=file_sha256(summary_path),
                      protocol_sha256=protocol_sha), fixed_joint_csv_path=str(csv_path),
                  fixed_joint_csv_sha256=file_sha256(csv_path), image_io=False, actual_lm_forward=False,
                  scope='Case binding only; no reference image enters the caption-only teacher LM, no video-quality claim')
    write_new(output / 'finished.json', report)
    print(json.dumps(dict(event='pure_text_cases_bound', **report), allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build_oracle_captions', action='store_true')
    mode.add_argument('--bind_pure_text_cases', action='store_true', help='Bind caption-only raw teacher tokens to the fixed three joint cases; no image encoding')
    parser.add_argument('--cache', type=Path, help='Completed pure-text source cache directory')
    parser.add_argument('--exp_root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--neutral_teacher', action='store_true', help='Only teacher pronouns change; actions and student input unchanged')
    args = parser.parse_args()
    if args.exp_root is None or args.out is None:
        parser.error('Require --exp_root --out')
    if args.bind_pure_text_cases:
        if args.cache is None or args.neutral_teacher:
            parser.error('Binding requires --cache and cannot transform teacher captions')
        bind_pure_text_cases(args.cache, args.exp_root, args.out)
    else:
        if args.cache is not None:
            parser.error('--cache is only valid with --bind_pure_text_cases')
        build_oracle_captions(args.exp_root, args.out, neutral_teacher=args.neutral_teacher)
