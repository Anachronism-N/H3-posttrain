#!/usr/bin/env python3
"""Test NZC415/m2d quota with a tiny aux-file upload for stage4."""
from huggingface_hub import HfApi

api = HfApi(token='<HF_TOKEN_REDACTED>')
try:
    api.upload_file(
        path_or_fileobj='models/train/OSS_fullSFT_stage4_omni257k_b32_2ep/args.json',
        path_in_repo='OmniDance-fullSFT/OSS_fullSFT_stage4_omni257k_b32_2ep/args.json',
        repo_id='NZC415/m2d', repo_type='model',
        commit_message='upload aux stage4 args.json (quota test)')
    print('UPLOAD_OK: quota available')
except Exception as e:
    print('UPLOAD_FAIL:', str(e)[:300])
