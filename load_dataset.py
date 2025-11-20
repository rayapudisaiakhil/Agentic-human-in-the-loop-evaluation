#!/usr/bin/env python
"""
Load the sample dataset into the system for testing.
"""

import json
import requests

BASE_URL = "http://localhost:8000"

def load_dataset():
    """Load samples from dataset/samples.json."""
    # Read the dataset
    with open("dataset/samples.json", "r") as f:
        samples = json.load(f)

    print(f"Loading {len(samples)} samples...")

    # Send batch inference request
    texts = [sample["text"] for sample in samples]

    response = requests.post(
        f"{BASE_URL}/batch_infer",
        json={"texts": texts, "batch_id": "initial_load"}
    )

    if response.status_code == 200:
        result = response.json()
        print(f"\nSuccessfully processed {result['total_processed']} samples!")
        print(f"Auto-accepted: {result['summary']['auto_accepted']}")
        print(f"Pending review: {result['summary']['pending']}")

        # Show some examples
        print("\n--- Sample Results ---")
        for i, res in enumerate(result["results"][:3]):
            print(f"\nSample {i+1}:")
            print(f"  Text: {res['text'][:50]}...")
            print(f"  Prediction: {res['model_label']}")
            print(f"  Confidence: {res['confidence']:.2%}")
            print(f"  Status: {res['status']}")

    else:
        print(f"Error: {response.status_code}")
        print(response.json())

    # Check final stats
    print("\n--- System Stats ---")
    stats_response = requests.get(f"{BASE_URL}/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Status breakdown:")
        for status, count in stats['status_breakdown'].items():
            print(f"  {status}: {count}")

if __name__ == "__main__":
    load_dataset()