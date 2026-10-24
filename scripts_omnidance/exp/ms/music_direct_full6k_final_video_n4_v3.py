"""Final2250 resume in the existing complete H3 environment; frozen v1/v2."""
from pathlib import Path
import music_direct_full6k_final_video_n4_v2 as resume

V2_SOURCE = Path(resume.__file__)
V2_SHA = '93a5fa170b0fae2b7663db021625ec90ad2c794db70c2717fc2ef02a558fbc59'


def main():
    if resume.sha(V2_SOURCE) != V2_SHA or resume.sha(resume.ORIGINAL_SOURCE) != resume.ORIGINAL_SOURCE_SHA:
        raise RuntimeError('Frozen manager sources differ')
    original = resume.original
    original.OUT = original.MS / 'music_direct_full6k_final2250_video_n4_v3'
    original.QUEUE = original.MS / 'music_direct_full6k_final2250_video_n4_queue_v3'
    original.full_cpu_checkpoint = resume.reuse_full_cpu_checkpoint
    original.acquire = resume.acquire_coordinated_gpu3
    original.__file__ = __file__
    original.main()


if __name__ == '__main__':
    main()
