#!/usr/bin/env python3
"""Absolute audio-input review with the same strict contract as pairwise review."""
try:
    from .judge_pairs_audio import (CRITERIA, ABSOLUTE_SYSTEM as SYSTEM, compact,
                                   audio_part, call, first_object, main as _main)
except ImportError:
    from judge_pairs_audio import (CRITERIA, ABSOLUTE_SYSTEM as SYSTEM, compact,
                                  audio_part, call, first_object, main as _main)


def main(argv=None):
    return _main(argv, absolute=True)


if __name__ == "__main__":
    main()
