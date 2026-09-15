"""Run the real VectorEngine relevance benchmark for Cycle 027.

This deliberately has no fallback model. Missing optional dependencies or a
failed model/index load produce a blocked report and a non-zero exit code.
The labels are a small synthetic acceptance corpus, not a population holdout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_local_embeddings import CASES, RECORDS


def recall_at_k(retrieved: list[str], expected: str, k: int) -> float:
    return float(expected in retrieved[:k])


def ndcg_at_k(retrieved: list[str], expected: str, k: int) -> float:
    try:
        rank = retrieved[:k].index(expected)
    except ValueError:
        return 0.0
    import math
    return 1.0 / math.log2(rank + 2)


def evaluate(model_name: str) -> dict:
    report = {
        'schema_version': 1,
        'cycle': '027',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model': model_name,
        'execution': 'real_sentence_transformers_and_faiss',
        'status': 'blocked',
        'engine_sha256': hashlib.sha256((ROOT / 'src/vector_engine.py').read_bytes()).hexdigest(),
        'metrics': {},
        'checks': [],
        'dependencies': {},
    }
    for package in ('faiss-cpu', 'sentence-transformers', 'torch', 'numpy'):
        try:
            report['dependencies'][package] = version(package)
        except PackageNotFoundError:
            report['dependencies'][package] = 'missing'
    try:
        from src.vector_engine import VectorEngine
        engine = VectorEngine(model_name=model_name)
        engine.build_index(
            [record['text'] for record in RECORDS],
            [{**record, 'source': {'snapshot': 'cycle-027-synthetic'}} for record in RECORDS],
            provenance={
                'workspace_id': 'cycle-027',
                'dataset_id': 'synthetic-vector-027',
                'generation': 'cycle-027',
                'current_fingerprint': 'synthetic-corpus-not-production-data',
            },
        )
        per_case = []
        for case in CASES:
            result = engine.search(case['query'], top_k=3)
            ids = [row.get('id') for row in result]
            per_case.append({
                'id': case['id'],
                'scope': case['scope'],
                'expected': case['expected'],
                'retrieved_ids': ids,
                'recall_at_1': recall_at_k(ids, case['expected'], 1),
                'recall_at_3': recall_at_k(ids, case['expected'], 3),
                'ndcg_at_1': ndcg_at_k(ids, case['expected'], 1),
                'ndcg_at_3': ndcg_at_k(ids, case['expected'], 3),
            })
        report['checks'] = per_case
        for k in (1, 3):
            report['metrics'][f'recall_at_{k}'] = sum(item[f'recall_at_{k}'] for item in per_case) / len(per_case)
            report['metrics'][f'ndcg_at_{k}'] = sum(item[f'ndcg_at_{k}'] for item in per_case) / len(per_case)
        report['status'] = 'passed' if all(item['recall_at_1'] == 1.0 for item in per_case) else 'failed'
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    report['passed'] = sum(1 for item in report['checks'] if item.get('recall_at_1') == 1.0)
    report['total'] = len(report['checks'])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = evaluate(args.model)
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
