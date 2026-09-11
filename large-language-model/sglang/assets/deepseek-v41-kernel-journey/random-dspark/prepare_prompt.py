"""Create a reproducible random 4096-token input from the model's ordinary tokens."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument('--model-path', type=Path, required=True)
p.add_argument('--out', type=Path, default=Path('prompt.json'))
p.add_argument('--seed', type=int, default=42)
a = p.parse_args()
tokenizer_path = a.model_path / 'tokenizer.json'
raw = tokenizer_path.read_bytes()
config = json.loads(raw)
tok = Tokenizer.from_file(str(tokenizer_path))
special = {v['id'] for v in config.get('added_tokens', []) if v.get('special')}
ordinary = sorted(set(tok.get_vocab().values()) - special)
assert ordinary
rng = random.Random(a.seed)
ids = [ordinary[rng.randrange(len(ordinary))] for _ in range(4096)]
prompt = {
    'dataset': 'synthetic-uniform-token-ids', 'seed': a.seed,
    'generation': '4096 independent uniform draws, with replacement, from sorted tokenizer vocabulary IDs excluding declared special tokens; Python random.Random(seed).randrange.',
    'tokenizer_sha256': hashlib.sha256(raw).hexdigest(),
    'ordinary_vocabulary_size': len(ordinary),
    'excluded_special_token_ids': sorted(special),
    'input_ids': ids,
    'input_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
    'input_tokens': len(ids), 'max_new_tokens': 1024,
    'ignore_eos': True, 'temperature': 0,
    'simulated_acceptance_target': 5.5,
    'simulated_acceptance_method': 'match-expected',
}
a.out.write_text(json.dumps(prompt, indent=2) + '\n')
print('WROTE', a.out.name, 'input_tokens', len(ids), 'input_sha256', prompt['input_ids_sha256'])
