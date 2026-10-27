"""Text/token boundary tests; no label inference or pretrained model execution."""
import pytest

from scripts_omnidance.exp.ms import music_planner_caption_format as fmt


CLASSES = ['hip-hop', '手势舞']


def spec():
    return dict(schema=fmt.SCHEMA, genre_header_template='舞种：{genre}\n',
                time_tolerance_seconds=.001, require_full_coverage=True)


def selected(**changes):
    value = spec()
    value.update(changes)
    return value


def text(actions='[0-1.250s] 双手抬至胸前。 [1.2500-2.50000s] 双手向外打开。', genre='手势舞'):
    return '舞种：' + genre + '\n' + actions


def test_parse_continuous_text_arbitrary_decimal_precision_and_exact_body():
    actions = '\n [0.00000-1.250000s] 双手抬至胸前。\n\t[1.250000-2.50000000s] 双手向外打开。\n'
    caption = text(actions)
    parsed = fmt.parse_caption(caption, CLASSES, spec(), 2.5)
    assert parsed == dict(genre='手势舞', header_text='舞种：手势舞\n', action_text=actions,
        windows=[dict(start=0., end=1.25, caption='双手抬至胸前。'),
                 dict(start=1.25, end=2.5, caption='双手向外打开。')])
    assert parsed['header_text'] + parsed['action_text'] == caption


def test_english_header_and_multiline_actions_are_configurable():
    chosen = selected(genre_header_template='Dance style: {genre}\n')
    caption = 'Dance style: hip-hop\n[0-1s] Raise one arm.\nKeep the other near the chest.\n[1-2.5s] Lower both arms.'
    parsed = fmt.parse_caption(caption, CLASSES, chosen, 2.5)
    assert parsed['genre'] == 'hip-hop'
    assert parsed['windows'][0]['caption'] == 'Raise one arm.\nKeep the other near the chest.'
    assert fmt.format_header('hip-hop', CLASSES, chosen) == 'Dance style: hip-hop\n'


def test_style_words_inside_actions_are_not_additional_genre_headers():
    caption = text('[0-1s] 延续hip-hop的身体律动，随后加入手势舞的双手动作。\n[1-2.5s] 保持手势舞的手部表达。')
    assert fmt.parse_caption(caption, CLASSES, spec(), 2.5)['genre'] == '手势舞'
    # Even the literal header text embedded inside an action sentence is prose.
    caption = text('[0-2.5s] 不要把“舞种：hip-hop”这几个字当作另一个字段。')
    assert len(fmt.parse_caption(caption, CLASSES, spec(), 2.5)['windows']) == 1
    bare = selected(genre_header_template='{genre}\n')
    caption = '手势舞\n[0-2.5s] 双手随律动展开。\nhip-hop\n'
    assert fmt.parse_caption(caption, CLASSES, bare, 2.5)['windows'][0]['caption'].endswith('hip-hop')


@pytest.mark.parametrize('caption', [text(genre='salsa'), '手势舞\n[0-2.5s] 抬手。',
    '说明\n' + text(), '舞种：手势舞、hip-hop\n[0-2.5s] 抬手。',
    '舞种： 手势舞\n[0-2.5s] 抬手。'])
def test_unknown_multiple_or_missing_initial_genre_rejected(caption):
    with pytest.raises(ValueError, match='whitelisted'):
        fmt.parse_caption(caption, CLASSES, spec(), 2.5)


@pytest.mark.parametrize('second', ['舞种：hip-hop\n', '舞种：salsa\n', '舞种：手势舞\n'])
def test_second_standalone_header_is_rejected(second):
    caption = text('[0-1s] 抬手。\n' + second + '[1-2.5s] 放下手。')
    with pytest.raises(ValueError, match='only one'):
        fmt.parse_caption(caption, CLASSES, spec(), 2.5)


def test_second_header_at_eof_is_not_absorbed_into_action_text():
    with pytest.raises(ValueError, match='only one'):
        fmt.parse_caption(text('[0-2.5s] 抬手。\n舞种：hip-hop'), CLASSES, spec(), 2.5)


@pytest.mark.parametrize('stamp', ['[０-１s]', '[0–1s]', '[0—1s]', '[0 − 1s]', '[-1-1s]',
    '[+0-1s]', '[0-1e0s]', '[0-NaNs]', '[NaN-1s]', '[0-infs]', '[.0-1s]', '[0.-1s]',
    '[0-1S]', '[0s-1s]', '[0 - 1s]', '[0-1 s]', '[0-1]', '[0-1s', '0-1s]'])
def test_malformed_time_grammar_is_not_silently_ignored(stamp):
    with pytest.raises(ValueError):
        fmt.parse_caption(text(stamp + ' 抬手。 [1-2.5s] 放下手。'), CLASSES, spec(), 2.5)


def test_malformed_timestamp_between_valid_markers_is_rejected():
    with pytest.raises(ValueError, match='square brackets'):
        fmt.parse_caption(text('[0-1s] 抬手。 [1–2s] 转身。 [2-2.5s] 放手。'), CLASSES, spec(), 2.5)


@pytest.mark.parametrize('actions', ['[0-1s][1-2.5s] 放手。', '[0-2.5s]   ',
    '[0-0s] 抬手。 [0-2.5s] 放手。', '[1-0s] 抬手。 [0-2.5s] 放手。',
    '[0-2s] 抬手。 [1.9999-2.5s] 放手。', '[1-2s] 抬手。 [0-2.5s] 放手。',
    '[0-3s] 抬手。'])
def test_empty_reversed_overlapping_and_out_of_duration_windows_rejected(actions):
    with pytest.raises(ValueError):
        fmt.parse_caption(text(actions), CLASSES, spec(), 2.5)


@pytest.mark.parametrize('actions', ['[.5-2.5s] 抬手。', '[0.5-2.5s] 抬手。',
    '[0-1s] 抬手。 [1.1-2.5s] 放手。', '[0-2.4s] 抬手。'])
def test_full_coverage_rejects_missing_start_gaps_or_end(actions):
    with pytest.raises(ValueError):
        fmt.parse_caption(text(actions), CLASSES, spec(), 2.5)


def test_partial_coverage_explicitly_allowed_and_no_time_snapping():
    parsed = fmt.parse_caption(text('[0.5-1s] 抬手。 [1.2-2s] 放手。'), CLASSES,
                               selected(require_full_coverage=False), 2.5)
    assert [(w['start'], w['end']) for w in parsed['windows']] == [(.5, 1.), (1.2, 2.)]
    parsed = fmt.parse_caption(text('[0-5.063s] 抬手。'), CLASSES, spec(), 5.0625)
    assert parsed['windows'][0]['end'] == 5.063
    assert parsed['action_text'] == '[0-5.063s] 抬手。'
    with pytest.raises(ValueError):
        fmt.parse_caption(text('[0-5.064s] 抬手。'), CLASSES, spec(), 5.0625)
    with pytest.raises(ValueError):
        fmt.parse_caption(text('[0-5.063s] 抬手。'), CLASSES, selected(time_tolerance_seconds=0), 5.0625)


def test_target_metadata_matches_every_caption_and_boundary_without_repair():
    metadata = [dict(start=0., end=1.25, caption='双手抬至胸前。'),
                dict(start=1.25, end=2.5, caption='双手向外打开。')]
    parsed = fmt.parse_caption(text(), CLASSES, spec(), 2.5, metadata)
    assert parsed['windows'] == metadata
    rounded = [dict(start=0., end=1.2504, caption='双手抬至胸前。'),
               dict(start=1.2504, end=2.5, caption='双手向外打开。')]
    assert fmt.parse_caption(text(), CLASSES, spec(), 2.5, rounded)['windows'] == metadata
    assert rounded[0]['end'] == 1.2504


def test_exact_decimal_tolerance_boundary_is_accepted_without_snapping():
    assert 5.0635 - 5.0625 > .001  # Binary subtraction would incorrectly reject it.
    parsed = fmt.parse_caption(text('[0-5.0635s] 抬手。'), CLASSES, spec(), 5.0625,
                               [dict(start=0., end=5.0625, caption='抬手。')])
    assert parsed['windows'][0]['end'] == 5.0635
    with pytest.raises(ValueError):
        fmt.parse_caption(text('[0-5.0635001s] 抬手。'), CLASSES, spec(), 5.0625)


@pytest.mark.parametrize('metadata', [[], [dict(start=0., end=2.5, caption='双手抬至胸前。')],
    [dict(start=0., end=1.25, caption='双手向外打开。'), dict(start=1.25, end=2.5, caption='双手抬至胸前。')],
    [dict(start=0., end=1.3, caption='双手抬至胸前。'), dict(start=1.3, end=2.5, caption='双手向外打开。')],
    [dict(start=False, end=1.25, caption='双手抬至胸前。'), dict(start=1.25, end=2.5, caption='双手向外打开。')],
    [dict(start=0., end=1.25, caption='双手抬至胸前。'), dict(start=1.2, end=2.5, caption='双手向外打开。')],
    [dict(start=0., end=1.25, caption=' 双手抬至胸前。'), dict(start=1.25, end=2.5, caption='双手向外打开。')]])
def test_target_metadata_cannot_disagree_with_actual_caption(metadata):
    with pytest.raises(ValueError):
        fmt.parse_caption(text(), CLASSES, spec(), 2.5, metadata)


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf'), True, '2.5'])
def test_invalid_source_duration_rejected(duration):
    with pytest.raises(ValueError):
        fmt.parse_caption(text(), CLASSES, spec(), duration)


@pytest.mark.parametrize('changes', [dict(schema='other'), dict(extra=True),
    dict(genre_header_template='Dance: {genre}'), dict(genre_header_template='Dance: {genre} {genre}\n'),
    dict(genre_header_template='Dance: {label}\n'), dict(genre_header_template='Dance: {genre!r}\n'),
    dict(genre_header_template='Dance: {genre}\r\n'), dict(genre_header_template='<|assistant|>{genre}\n'),
    dict(time_tolerance_seconds=float('nan')), dict(time_tolerance_seconds=float('inf')),
    dict(time_tolerance_seconds=-.1), dict(time_tolerance_seconds=True), dict(require_full_coverage=1)])
def test_invalid_format_spec_rejected(changes):
    chosen = selected(**changes)
    with pytest.raises(ValueError):
        fmt.validate_format(chosen)


def test_format_preserves_user_values_and_rejects_bad_vocabulary():
    chosen = spec()
    assert fmt.validate_format(chosen) == chosen and fmt.validate_format(chosen) is not chosen
    for classes in ([], ['hand', 'hand'], ['hand', ''], ['hand', ' hip-hop'], ['hand', 'two\nstyles']):
        with pytest.raises(ValueError):
            fmt.parse_caption(text(), classes, chosen, 2.5)
    with pytest.raises(ValueError, match='whitelist'):
        fmt.format_header('salsa', CLASSES, chosen)


class BoundaryTokenizer:
    """A deliberate cross-boundary merge exposes the unsafe concat assumption."""
    def __init__(self):
        self.calls = []

    def encode(self, content, add_special_tokens=False):
        assert add_special_tokens is False
        self.calls.append(content)
        if content == '<|im_end|>':
            return [1]
        if '\n[0' in content:
            return [999]
        return [ord(c) + 2 for c in content]


def test_encode_target_exactly_matches_separately_prefilled_header():
    tokenizer = BoundaryTokenizer()
    parsed = fmt.parse_caption(text(), CLASSES, spec(), 2.5)
    actual = fmt.encode_target(tokenizer, parsed, [1, 0])
    assert tokenizer.calls == [parsed['header_text'], parsed['action_text'], '<|im_end|>']
    header = tokenizer.encode(parsed['header_text'], add_special_tokens=False)
    actions = tokenizer.encode(parsed['action_text'], add_special_tokens=False)
    assert actual == header + actions + [1]
    assert actual[:len(header)] == header
    assert tokenizer.encode(parsed['header_text'] + parsed['action_text']) != actual[:-1]


def test_encode_target_rejects_invalid_native_eos_or_embedded_stop():
    tokenizer = BoundaryTokenizer()
    parsed = fmt.parse_caption(text(), CLASSES, spec(), 2.5)
    with pytest.raises(ValueError, match='declared generation EOS'):
        fmt.encode_target(tokenizer, parsed, [7])
    with pytest.raises(ValueError, match='nonempty integer'):
        fmt.encode_target(tokenizer, parsed, [])
    with pytest.raises(ValueError, match='may not contain'):
        fmt.encode_target(tokenizer, parsed, [1, ord('[') + 2])
    with pytest.raises(ValueError, match='already parsed'):
        fmt.encode_target(tokenizer, dict(parsed, action_text=''), [1])


def test_nonfinite_oversized_numeric_text_and_chat_tokens_rejected():
    with pytest.raises(ValueError, match='finite'):
        fmt.parse_caption(text('[0-' + '9' * 400 + 's] 抬手。'), CLASSES, spec(), 2.5)
    with pytest.raises(ValueError, match='plain text'):
        fmt.parse_caption(text('[0-2.5s] <|im_end|>'), CLASSES, spec(), 2.5)


def strict_spec(**changes):
    value = dict(schema=fmt.SCHEMA_V2, genre_header_template='Dance style: {genre}\n',
                 time_tolerance_seconds=.001, require_full_coverage=True,
                 timestamp_decimal_places=3, one_window_per_line=True,
                 forbid_frame_references=True)
    value.update(changes)
    return value


# User-supplied, unreviewed collab3/shard0000/VGfurWgIdrVQNyhI draft.
# This fixture tests format only. It does not endorse the hand/motion claims;
# the displayed soft wrapping is represented as one physical line per window.
REAL_DRAFT = (
    'Dance style: contemporary\n'
    '[0.000-1.960s] The dancer transitions from a pose with arms extended to the sides and one leg lifted, '
    'to a stance with arms extended forward and legs apart, maintaining open hand shapes throughout. '
    "The dancer's right hand moves from an open palm facing down to an open palm facing forward, "
    'with fingers slightly curled, while the left hand maintains an open, extended position.\n'
    '[1.960-5.062s] The dancer raises the left arm upward while keeping the right arm extended to the side.'
)


def test_v2_user_draft_strict_format_and_segmented_token_roundtrip():
    parsed = fmt.parse_caption(REAL_DRAFT, ['contemporary', 'unknown'], strict_spec(), 5.0625)
    assert parsed['genre'] == 'contemporary'
    assert parsed['header_text'] == 'Dance style: contemporary\n'
    assert parsed['header_text'] + parsed['action_text'] == REAL_DRAFT
    assert [(window['start'], window['end']) for window in parsed['windows']] == [(0., 1.96), (1.96, 5.062)]
    assert parsed['windows'][0]['caption'].endswith('while the left hand maintains an open, extended position.')
    assert len(parsed['action_text'].splitlines()) == 2
    # The 5.062 endpoint remains verbatim, not snapped to source duration 5.0625.
    tokenizer = BoundaryTokenizer()
    ids = fmt.encode_target(tokenizer, parsed, [1])
    assert tokenizer.calls == [parsed['header_text'], parsed['action_text'], '<|im_end|>']
    assert ''.join(chr(token - 2) for token in ids[:-1]) == REAL_DRAFT
    assert fmt.parse_caption(REAL_DRAFT, ['contemporary'], strict_spec(), 5.0625,
                             target_windows=parsed['windows']) == parsed


def test_v2_retains_optional_final_lf_without_normalizing_caption():
    parsed = fmt.parse_caption(REAL_DRAFT + '\n', ['contemporary'], strict_spec(), 5.0625)
    assert parsed['header_text'] + parsed['action_text'] == REAL_DRAFT + '\n'
    assert parsed['action_text'].endswith('\n')


@pytest.mark.parametrize('actions', [
    '[0-2.500s] Raise both arms.', '[0.000-2.5s] Raise both arms.',
    '[0.0000-2.500s] Raise both arms.', '[0.000-2.5000s] Raise both arms.',
    '[0.000-1.250s] Raise both arms. [1.250-2.500s] Lower both arms.',
    '[0.000-2.500s] Raise both arms.\nKeep both palms open.',
    '\n[0.000-2.500s] Raise both arms.', ' [0.000-2.500s] Raise both arms.',
    '[0.000-1.250s] Raise both arms.\n\n[1.250-2.500s] Lower both arms.',
    '[0.000-2.500s] Raise both arms.\n\n',
    '[0.000-2.500s] Raise both arms.\u2028Keep both palms open.',
])
def test_v2_rejects_wrong_precision_physical_wrapping_or_multiple_windows_per_line(actions):
    with pytest.raises(ValueError, match='v2 profile'):
        fmt.parse_caption('Dance style: hip-hop\n' + actions, CLASSES, strict_spec(), 2.5)


@pytest.mark.parametrize('reference', ['by frame24', 'at frame #24', 'in frame 24',
    'at FRAME #24', 'at frame No. 24', 'at frame number 24', 'at frame-24', 'by frames 24-30'])
def test_v2_canonical_rejects_frame_draft_and_explicit_draft_profile_retains_it(reference):
    caption = 'Dance style: hip-hop\n[0.000-2.500s] Raise both arms ' + reference + '.'
    with pytest.raises(ValueError, match='Frame-number'):
        fmt.parse_caption(caption, CLASSES, strict_spec(), 2.5)
    draft = strict_spec(forbid_frame_references=False)
    assert draft != strict_spec()
    parsed = fmt.parse_caption(caption, CLASSES, draft, 2.5)
    assert reference in parsed['windows'][0]['caption']
    assert parsed['header_text'] + parsed['action_text'] == caption


@pytest.mark.parametrize('field', ['style_metadata_not_in_target: swing', 'frame_evidence: frame24',
    'label_source: auto', 'human_review_completed: false', 'confidence_score: .9',
    '"label_source": "auto"', "'label_source': 'auto'", 'semantic_approval: false',
    'review_provenance: automatic', 'audio_sha256: abc', 'source_group_id: source-1', 'label_version: v1'])
def test_v2_rejects_explicit_machine_metadata_keys_even_in_frame_permitting_draft(field):
    caption = 'Dance style: hip-hop\n[0.000-2.500s] Raise both arms. ' + field
    with pytest.raises(ValueError, match='Machine metadata'):
        fmt.parse_caption(caption, CLASSES, strict_spec(forbid_frame_references=False), 2.5)


def test_v2_natural_confidence_and_framing_prose_are_not_metadata_or_frame_numbers():
    actions = ('[0.000-2.500s] Move with confidence: raise both arms while remaining in frame. '
               'Continue for 24 frames with the same open pose.')
    parsed = fmt.parse_caption('Dance style: hip-hop\n' + actions, CLASSES, strict_spec(), 2.5)
    assert parsed['action_text'] == actions
    # The profile does not pretend to implement a reliable English detector.
    assert fmt.parse_caption('Dance style: 手势舞\n[0.000-2.500s] 双手展开。',
                             CLASSES, strict_spec(), 2.5)['genre'] == '手势舞'


@pytest.mark.parametrize('changes', [dict(timestamp_decimal_places=2), dict(timestamp_decimal_places=3.),
    dict(timestamp_decimal_places=True), dict(one_window_per_line=False), dict(one_window_per_line=1),
    dict(forbid_frame_references=None), dict(forbid_frame_references=1), dict(extra=True)])
def test_v2_profile_fields_are_explicit_and_strict(changes):
    with pytest.raises(ValueError):
        fmt.validate_format(strict_spec(**changes))


def test_v1_four_keys_and_legacy_prose_remain_unchanged():
    assert set(fmt.validate_format(spec())) == {'schema', 'genre_header_template',
                                               'time_tolerance_seconds', 'require_full_coverage'}
    actions = '[0-1.25000s] Raise both arms by frame24.\nKeep palms open. confidence_score: .9 [1.250-2.5s] Lower both arms.'
    parsed = fmt.parse_caption(text(actions), CLASSES, spec(), 2.5)
    assert parsed['action_text'] == actions
    with pytest.raises(ValueError, match='documented'):
        fmt.validate_format(dict(spec(), forbid_frame_references=False))
    incomplete = strict_spec()
    incomplete.pop('forbid_frame_references')
    with pytest.raises(ValueError, match='documented'):
        fmt.validate_format(incomplete)
