"""Pin model artifacts in a per-model task-owned cache."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--allow-pattern', action='append')
    args = parser.parse_args()
    assert args.key and all(c.isalnum() or c in '-_' for c in args.key)
    root = Path('/scratch/bbuf-diffusion-20261011')
    cache = root / 'models' / args.key
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_TOKEN_PATH', str(Path.home()/'.cache/huggingface/token'))
    os.environ['HF_HOME'] = str(cache)
    os.environ['HF_XET_CACHE'] = str(cache/'xet')
    from huggingface_hub import snapshot_download
    manifest = dict(model=args.model, revision=args.revision, task_owned_cache=str(cache),
                    allow_patterns=args.allow_pattern)
    output = root/'results'/f'{args.key}-checkpoint.json'
    output.write_text(json.dumps(manifest, indent=2))
    path = snapshot_download(args.model, revision=args.revision, cache_dir=cache/'hub',
                             max_workers=8, allow_patterns=args.allow_pattern)
    manifest.update(snapshot_path=path, download_complete=True)
    output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
