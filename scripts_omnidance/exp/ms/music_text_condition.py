"""Raw H3 text-tail replacement; oracle captions are supervision, not music inference."""
import hashlib
import json
import os
from pathlib import Path

import torch

from diffsynth.models.minimax_h3_text_encoder import _text_ids
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build_oracle_captions', action='store_true')
    parser.add_argument('--exp_root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--neutral_teacher', action='store_true', help='Only teacher pronouns change; actions and student input unchanged')
    args = parser.parse_args()
    if not args.build_oracle_captions or args.exp_root is None or args.out is None:
        parser.error('Require --build_oracle_captions --exp_root --out')
    build_oracle_captions(args.exp_root, args.out, neutral_teacher=args.neutral_teacher)
