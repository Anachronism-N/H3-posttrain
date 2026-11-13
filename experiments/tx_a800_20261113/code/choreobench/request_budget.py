"""Bound the complete harness request, including structured decoding schemas.

The unit is the number of Unicode characters in compact, sorted JSON with
ensure_ascii=False, not message-text length, UTF-8 bytes, tokenizer tokens, or
the provider's transformed HTTP body. This guard does not establish that a
request fits a model's context window or its grammar compiler's capacity.
"""
import json

from .baseline import ModelOutputError
from .io import finite_json


def guard_request(request, cfg):
    """Return the unchanged request or reject it before journal/provider dispatch.

    Call after adding *all* fields, especially response_schema, and before
    journal.call. No content is truncated, schema simplified, or model called.
    Counting stops as soon as the configured limit is exceeded, avoiding a
    second complete serialized copy of a large schema. The schema object must
    already exist; a schema-builder allocation limit is a separate concern.
    """
    if not isinstance(cfg, dict) or type(cfg.get("max_request_characters")) is not int or cfg["max_request_characters"] < 1:
        raise ValueError("max_request_characters must be an explicit positive integer")
    if not isinstance(request, dict):
        raise ModelOutputError("Complete harness request must be a finite JSON object; no request sent")
    try:
        finite_json(request)
    except (ValueError, RecursionError) as error:
        raise ModelOutputError("Complete harness request is not finite JSON; no request sent or content truncated") from error
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    count, limit = 0, cfg["max_request_characters"]
    try:
        for chunk in encoder.iterencode(request):
            count += len(chunk)
            if count > limit:
                raise ModelOutputError(
                    "Complete serialized harness request exceeds max_request_characters=%d "
                    "(at least %d characters, including response_schema and all other fields); "
                    "no request sent or content truncated. This character guard is not a tokenizer/context limit."
                    % (limit, count))
    except ModelOutputError:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise ModelOutputError("Complete harness request cannot be serialized as finite JSON; no request sent") from error
    return request
