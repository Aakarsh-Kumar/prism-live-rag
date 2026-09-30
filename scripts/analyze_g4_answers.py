#!/usr/bin/env python3
"""
Analyze G4 evaluation answers to distinguish between substantive answers and abstentions.

This script identifies which answers are actual attempts to answer vs abstentions like
"The supplied facts do not..." to properly measure grounding quality.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

# Patterns that indicate abstention
ABSTENTION_PATTERNS = [
    r"supplied facts? (?:do not|does not|don't|doesn't)",
    r"cannot (?:be )?determined",
    r"(?:are|is) insufficient",
    r"I (?:don't|do not) have",
    r"not (?:contain|provide|address|explain)",
]

def is_abstention(response: str) -> bool:
    """Check if a response is an abstention rather than substantive answer."""
    response_lower = response.lower()
    return any(re.search(pattern, response_lower) for pattern in ABSTENTION_PATTERNS)

def analyze_dataset(dataset_path: Path) -> Dict:
    """Analyze the eval dataset to categorize answers."""

    results = {
        'total': 0,
        'substantive': 0,
        'abstention': 0,
        'substantive_queries': [],
        'abstention_queries': [],
    }

    with open(dataset_path) as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            results['total'] += 1

            response = data.get('response', '')

            if is_abstention(response):
                results['abstention'] += 1
                results['abstention_queries'].append({
                    'query_id': data.get('query_id', ''),
                    'query': data.get('query', ''),
                    'response': response[:200],  # First 200 chars
                })
            else:
                results['substantive'] += 1
                results['substantive_queries'].append({
                    'query_id': data.get('query_id', ''),
                    'query': data.get('query', ''),
                    'response': response[:200],  # First 200 chars
                })

    return results

def main():
    dataset_path = Path('data/eval_dataset.jsonl')

    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found")
        return 1

    print("Analyzing G4 evaluation answers...")
    print("=" * 80)

    results = analyze_dataset(dataset_path)

    # Print summary
    print(f"\nSUMMARY:")
    print(f"Total queries: {results['total']}")
    print(f"Substantive answers: {results['substantive']} ({results['substantive']/results['total']*100:.1f}%)")
    print(f"Abstentions: {results['abstention']} ({results['abstention']/results['total']*100:.1f}%)")

    # Print examples of each
    print(f"\n{'='*80}")
    print("SAMPLE SUBSTANTIVE ANSWERS (first 5):")
    print("=" * 80)
    for i, query in enumerate(results['substantive_queries'][:5], 1):
        print(f"\n{i}. Query: {query['query']}")
        print(f"   Response: {query['response']}...")

    print(f"\n{'='*80}")
    print("SAMPLE ABSTENTIONS (first 5):")
    print("=" * 80)
    for i, query in enumerate(results['abstention_queries'][:5], 1):
        print(f"\n{i}. Query: {query['query']}")
        print(f"   Response: {query['response']}...")

    # Save detailed results
    output_path = Path('data/g4_answer_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Detailed analysis saved to: {output_path}")
    print("=" * 80)

    print(f"\nCRITICAL FINDING:")
    print(f"The current G4 'success' metric is misleading because it includes abstentions.")
    print(f"Real grounding quality should be measured only on the {results['substantive']} substantive answers.")
    print(f"Abstention rate of {results['abstention']/results['total']*100:.1f}% indicates retrieval is finding")
    print(f"irrelevant context for many queries (domain mismatch problem).")

    return 0

if __name__ == '__main__':
    exit(main())
