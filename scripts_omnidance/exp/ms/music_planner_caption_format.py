"""Exact text protocol: one allowed dance genre, then timestamped actions.

An action marker is ``[START-ENDs]``. START/END use ASCII digits, with an optional
decimal point followed by digits, e.g. [0-1.25s] or [1.2500-2.50000s]. Only the
ASCII hyphen and lowercase trailing s are accepted. Square brackets in the
action body are reserved for these markers. The historical v1 permits actions
to span lines and markers to share a line. The explicit v2 profile instead uses
exactly three decimal places and one window per physical LF-separated line.
Its timestamps describe window coverage, not an assertion of action onset.
No action, genre, timestamp, line wrapping or frame reference is rewritten.
"""
import math
import re
from decimal import Decimal


SCHEMA = 'music-planner-dance-caption-v1'
SCHEMA_V2 = 'music-planner-dance-caption-v2'
TOKEN_PROTOCOL = 'header-then-actions-then-native-im-end-v1'
_NUMBER = r'[0-9]+(?:\.[0-9]+)?'
_STAMP = re.compile(r'\[(?P<start>' + _NUMBER + r')-(?P<end>' + _NUMBER + r')s\]')
_KEYS = {'schema', 'genre_header_template', 'time_tolerance_seconds', 'require_full_coverage'}
_V2_KEYS = _KEYS | {'timestamp_decimal_places', 'one_window_per_line', 'forbid_frame_references'}
_FRAME_REFERENCE = re.compile(r'\bframes?(?:\s*(?:#|no\.?|number)\s*|\s*[-:]?\s*)[0-9]+\b', re.IGNORECASE)
# These explicit machine keys are rejected only when written as key: value.
# Ordinary prose about confidence or frame composition remains valid.
_METADATA_KEYS = ('style_metadata_not_in_target', 'frame_evidence', 'label_source',
                  'human_review_completed', 'confidence_score', 'semantic_approval',
                  'review_provenance', 'label_version', 'audio_sha256', 'source_group_id')
_METADATA_FIELD = re.compile(r'\b(?:' + '|'.join(_METADATA_KEYS) + r')[\"\']?\s*:', re.IGNORECASE)


def _plain(text):
    return (isinstance(text, str) and '<|' not in text and '<audio>' not in text.lower() and
            not any(ord(char) < 32 and char not in '\n\t' for char in text))


def validate_format(spec):
    """Validate a complete explicit spec; preserve all values verbatim."""
    if (not isinstance(spec, dict) or spec.get('schema') not in (SCHEMA, SCHEMA_V2) or
            set(spec) != (_V2_KEYS if spec.get('schema') == SCHEMA_V2 else _KEYS)):
        raise ValueError('Dance caption format requires exactly the documented v1 or v2 fields')
    if spec['schema'] == SCHEMA_V2:
        if type(spec['timestamp_decimal_places']) is not int or spec['timestamp_decimal_places'] != 3:
            raise ValueError('The v2 profile requires timestamp_decimal_places=3')
        if spec['one_window_per_line'] is not True:
            raise ValueError('The v2 profile requires one_window_per_line=true')
        if type(spec['forbid_frame_references']) is not bool:
            raise ValueError('forbid_frame_references must be an explicit boolean')
    template = spec['genre_header_template']
    if (not _plain(template) or template.count('{genre}') != 1 or not template.endswith('\n') or
            '{' in template.replace('{genre}', '') or '}' in template.replace('{genre}', '')):
        raise ValueError('genre_header_template needs exactly one {genre}, LF ending and no other fields')
    tolerance = _numeric(spec['time_tolerance_seconds'], 'time_tolerance_seconds')
    if tolerance < 0:
        raise ValueError('time_tolerance_seconds must be finite and nonnegative')
    if type(spec['require_full_coverage']) is not bool:
        raise ValueError('require_full_coverage must be a boolean')
    return dict(spec)


def _classes(classes):
    if (not isinstance(classes, (list, tuple)) or not classes or
            any(not _plain(name) or not name.strip() or name != name.strip() or '\n' in name or
                '\t' in name for name in classes) or len(set(classes)) != len(classes)):
        raise ValueError('A nonempty unique whitelist of plain dance genre names is required')
    return list(classes)


def format_header(genre, classes, spec):
    spec = validate_format(spec)
    classes = _classes(classes)
    if not isinstance(genre, str) or genre not in classes:
        raise ValueError('Dance genre is outside the configured whitelist')
    return spec['genre_header_template'].replace('{genre}', genre)


def _numeric(value, description):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(description + ' must be finite numeric seconds')
    return value


def _within(left, right, tolerance):
    # Compare declared decimal values without a binary float subtraction making
    # an exact tolerance boundary (e.g. 5.0635 - 5.0625 == .001) fail spuriously.
    return abs(Decimal(str(left)) - Decimal(str(right))) <= Decimal(str(tolerance))


def _check_windows(windows, duration, tolerance, full_coverage):
    previous = 0.
    for index, window in enumerate(windows):
        start = _numeric(window.get('start'), 'Window start')
        end = _numeric(window.get('end'), 'Window end')
        # A tolerance never permits a negative, empty or overlapping interval.
        if (not 0 <= start < end or (index and start < previous) or
                (end > duration and not _within(end, duration, tolerance))):
            raise ValueError('Windows must be positive, ordered, nonoverlapping and within source duration')
        if full_coverage and not _within(start, previous, tolerance):
            raise ValueError('Full coverage requires the first start at zero and no temporal gaps')
        previous = end
    if full_coverage and not _within(previous, duration, tolerance):
        raise ValueError('Full coverage requires the final end to match the source duration')


def _second_header(action_text, template):
    prefix, suffix = template.split('{genre}')
    # Bare ``{genre}\n`` has no distinguishing header marker. A style word in
    # action prose is not a second header; the required initial header is exact.
    if not (prefix.strip() or suffix.strip()):
        return False
    pattern = re.compile(r'(?m)^' + re.escape(prefix) + r'[^\r\n]+' +
                         re.escape(suffix[:-1]) + r'(?:\n|$)')
    return pattern.search(action_text) is not None


def _check_v2_body(action_text, spec):
    # A final LF is harmless and retained. Blank lines, indentation before a
    # timestamp and physically wrapped actions are not silently joined together.
    lines = action_text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    if not lines or any(not line or any(char in line for char in '\x85\u2028\u2029') for line in lines):
        raise ValueError('The v2 profile requires one window per physical LF-separated line')
    for line in lines:
        stamps = list(_STAMP.finditer(line))
        if len(stamps) != 1 or stamps[0].start() != 0:
            raise ValueError('The v2 profile requires one window per physical line without wrapping')
        for boundary in ('start', 'end'):
            if re.fullmatch(r'[0-9]+\.[0-9]{3}', stamps[0][boundary]) is None:
                raise ValueError('The v2 profile requires exactly three timestamp decimal places')
    if spec['forbid_frame_references'] and _FRAME_REFERENCE.search(action_text):
        raise ValueError('Frame-number references are forbidden by this caption profile')
    if _METADATA_FIELD.search(action_text):
        raise ValueError('Machine metadata key/value fields do not belong in action text')


def parse_caption(text, classes, spec, duration, target_windows=None):
    """Return genre, exact header/body and parsed action windows.

    Time tolerance applies to source endpoints, optional coverage, and metadata
    equality. The returned times are the original parsed numbers, never snapped.
    Metadata captions must exactly equal each action after removing delimiter
    whitespace; metadata never supplies or repairs missing text.
    """
    spec = validate_format(spec)
    classes = _classes(classes)
    duration = _numeric(duration, 'Source duration')
    if duration <= 0:
        raise ValueError('Source duration must be positive')
    if not _plain(text) or not text:
        raise ValueError('Caption must be nonempty plain text without reserved chat tokens')
    headers = [(name, format_header(name, classes, spec)) for name in classes]
    matches = [(name, header) for name, header in headers if text.startswith(header)]
    if len(matches) != 1:
        raise ValueError('Caption must start with exactly one whitelisted genre header')
    genre, header = matches[0]
    action_text = text[len(header):]
    if _second_header(action_text, spec['genre_header_template']):
        raise ValueError('Caption may contain only one genre header')
    if spec['schema'] == SCHEMA_V2:
        _check_v2_body(action_text, spec)
    stamps = list(_STAMP.finditer(action_text))
    if not stamps or action_text[:stamps[0].start()].strip():
        raise ValueError('Actions must begin with an explicit [START-ENDs] marker')
    windows = []
    for index, stamp in enumerate(stamps):
        following = stamps[index + 1].start() if index + 1 < len(stamps) else len(action_text)
        caption = action_text[stamp.end():following].strip()
        if not caption or '[' in caption or ']' in caption:
            raise ValueError('Every time marker requires action text; square brackets are reserved for valid timestamps')
        windows.append(dict(start=float(stamp['start']), end=float(stamp['end']), caption=caption))
    tolerance = spec['time_tolerance_seconds']
    _check_windows(windows, duration, tolerance, spec['require_full_coverage'])
    if target_windows is not None:
        if not isinstance(target_windows, list) or len(target_windows) != len(windows):
            raise ValueError('target_windows count differs from the caption')
        if any(not isinstance(window, dict) for window in target_windows):
            raise ValueError('target_windows must contain objects')
        _check_windows(target_windows, duration, tolerance, spec['require_full_coverage'])
        for parsed, target in zip(windows, target_windows):
            if (target.get('caption') != parsed['caption'] or
                    not _within(target['start'], parsed['start'], tolerance) or
                    not _within(target['end'], parsed['end'], tolerance)):
                raise ValueError('target_windows action text or boundary differs from the actual caption')
    return dict(genre=genre, action_text=action_text, windows=windows, header_text=header)


def _encode(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, (list, tuple)) or not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError('Tokenizer must encode nonempty text to a flat list of nonnegative token ids')
    return list(ids)


def encode_target(tokenizer, parsed, eos_ids):
    """Encode header + actions as SEPARATE segments, followed by native im_end.

    Generation must force the same independently encoded ``header_text`` before
    continuing actions. Encoding the concatenated text can merge across the
    boundary and is a different, unsupported token protocol.
    """
    if (not isinstance(parsed, dict) or not _plain(parsed.get('header_text')) or
            not parsed['header_text'].endswith('\n') or not _plain(parsed.get('action_text')) or
            not parsed['action_text'].strip()):
        raise ValueError('encode_target requires an already parsed header and nonempty action body')
    if type(eos_ids) is int:
        eos_ids = [eos_ids]
    if (not isinstance(eos_ids, (list, tuple)) or not eos_ids or
            any(type(value) is not int or value < 0 for value in eos_ids)):
        raise ValueError('Native generation EOS ids must be a nonempty integer list')
    header = _encode(tokenizer, parsed['header_text'])
    actions = _encode(tokenizer, parsed['action_text'])
    terminator = _encode(tokenizer, '<|im_end|>')
    if len(terminator) != 1 or terminator[0] not in eos_ids:
        raise ValueError('Native im_end must be one declared generation EOS token')
    if any(value in eos_ids for value in header + actions):
        raise ValueError('Header/action segments may not contain native generation EOS tokens')
    return header + actions + terminator
