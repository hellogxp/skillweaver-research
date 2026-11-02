"""Explicit local configuration shared by research scripts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent.parent


def argument_parser(description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', default=os.environ.get('SKILLWEAVER_MODEL', 'Qwen/Qwen2.5-7B-Instruct'))
    parser.add_argument('--model-14b', default=os.environ.get('SKILLWEAVER_MODEL_14B', 'Qwen/Qwen2.5-14B-Instruct'))
    parser.add_argument('--encoder', default=os.environ.get('SKILLWEAVER_ENCODER', 'sentence-transformers/all-MiniLM-L6-v2'))
    parser.add_argument('--alternative-encoder', help='Explicit optional encoder for the step-count control comparison.')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs')
    parser.add_argument('--query-file', type=Path, help='Override the query file, including for a separately recovered dataset.')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def get_options(parse_cli=False):
    return argument_parser().parse_args(None if parse_cli else [])


def seed_everything(seed, include_torch=False):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    if not include_torch:
        return
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

