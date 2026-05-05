"""
Evaluation Metrics for Music Generation (Task 1-4)
Based on project specifications
"""

import numpy as np
from pathlib import Path
import json
from collections import Counter
import sys

sys.path.append(str(Path(__file__).parent.parent))

def pitch_histogram_similarity(original_tokens, generated_tokens, pitch_range=(21, 109)):
    """
    Calculate pitch histogram similarity between original and generated music
    Formula: H(p,q) = Σ|p_i - q_i|
    
    Lower score = more similar (better genre preservation)
    """
    # Extract pitch tokens (assuming pitch tokens are in certain range)
    # For REMI tokenizer, pitch tokens are typically 0-127
    original_pitches = [t for t in original_tokens if 21 <= t <= 109]
    generated_pitches = [t for t in generated_tokens if 21 <= t <= 109]
    
    # Create histograms
    orig_hist = np.zeros(128)
    gen_hist = np.zeros(128)
    
    for p in original_pitches:
        orig_hist[p] += 1
    for p in generated_pitches:
        gen_hist[p] += 1
    
    # Normalize
    if len(original_pitches) > 0:
        orig_hist = orig_hist / len(original_pitches)
    if len(generated_pitches) > 0:
        gen_hist = gen_hist / len(generated_pitches)
    
    # L1 distance
    similarity = np.sum(np.abs(orig_hist - gen_hist))
    
    return similarity

def rhythm_diversity_score(tokens, time_resolution=16):
    """
    Calculate rhythm diversity
    Formula: D = unique_durations / total_notes
    
    Higher score = more rhythmic variety
    """
    # Extract duration information from tokens
    # For REMI, duration tokens are typically > 1000 (customize based on your tokenizer)
    
    # Simplified: Use note-on events and measure inter-onset intervals
    note_onsets = [i for i, t in enumerate(tokens) if is_note_on_token(t)]
    
    if len(note_onsets) < 2:
        return 0.0
    
    # Calculate durations between note onsets
    durations = []
    for i in range(1, len(note_onsets)):
        duration = note_onsets[i] - note_onsets[i-1]
        durations.append(duration)
    
    if len(durations) == 0:
        return 0.0
    
    unique_durations = len(set(durations))
    total_durations = len(durations)
    
    diversity = unique_durations / total_durations
    
    return diversity

def is_note_on_token(token):
    """Check if token represents a note-on event"""
    # REMI tokenizer typically uses 0-127 for pitches
    # Adjust based on your actual token range
    return 0 <= token <= 127

def repetition_ratio(tokens, pattern_length=8):
    """
    Calculate repetition ratio
    Formula: R = repeated_patterns / total_patterns
    
    Lower score = less repetitive (more creative)
    """
    if len(tokens) < pattern_length * 2:
        return 1.0
    
    # Split into patterns
    patterns = []
    for i in range(0, len(tokens) - pattern_length, pattern_length):
        pattern = tuple(tokens[i:i+pattern_length])
        patterns.append(pattern)
    
    if len(patterns) == 0:
        return 0.0
    
    # Count unique patterns
    pattern_counts = Counter(patterns)
    
    # Count repeated patterns (appear more than once)
    repeated = sum(1 for count in pattern_counts.values() if count > 1)
    total = len(pattern_counts)
    
    if total == 0:
        return 0.0
    
    repetition = repeated / total
    
    return repetition

def calculate_all_metrics(original_tokens, generated_tokens):
    """
    Calculate all three quantitative metrics at once
    """
    metrics = {
        "pitch_similarity": pitch_histogram_similarity(original_tokens, generated_tokens),
        "rhythm_diversity": rhythm_diversity_score(generated_tokens),
        "repetition_ratio": repetition_ratio(generated_tokens),
    }
    
    return metrics

def evaluate_generated_samples(data_path, output_path):
    """
    Evaluate all generated samples against validation set
    """
    # Load validation sequences (real music for comparison)
    val_seqs = np.load(data_path / "val_seqs_fixed.npy")
    
    # Load generated samples
    gen_samples = []
    for i in range(1, 6):
        sample_path = output_path / f"generated_sample_{i}.npy"
        if sample_path.exists():
            gen_samples.append(np.load(sample_path))
    
    if len(gen_samples) == 0:
        print("No generated samples found!")
        return None
    
    # Sample random validation sequences for comparison
    np.random.seed(42)
    val_samples = val_seqs[np.random.choice(len(val_seqs), min(100, len(val_seqs)), replace=False)]
    
    print("="*60)
    print("EVALUATION METRICS - Task 1")
    print("="*60)
    
    all_results = []
    
    for i, gen in enumerate(gen_samples):
        # Compare with average of validation samples
        avg_pitch_sim = 0
        avg_rhythm = 0
        avg_repetition = 0
        
        for val in val_samples[:10]:  # Compare with 10 validation samples
            metrics = calculate_all_metrics(val, gen)
            avg_pitch_sim += metrics["pitch_similarity"]
            avg_rhythm += metrics["rhythm_diversity"]
            avg_repetition += metrics["repetition_ratio"]
        
        avg_pitch_sim /= 10
        avg_rhythm /= 10
        avg_repetition /= 10
        
        results = {
            "sample_id": i,
            "pitch_similarity": avg_pitch_sim,
            "rhythm_diversity": avg_rhythm,
            "repetition_ratio": avg_repetition,
        }
        all_results.append(results)
    
    # Print results table
    print("\n| Sample | Pitch Similarity↓ | Rhythm Diversity↑ | Repetition Ratio↓ |")
    print("|--------|-------------------|-------------------|-------------------|")
    for r in all_results:
        print(f"| {r['sample_id']:6d} | {r['pitch_similarity']:17.4f} | {r['rhythm_diversity']:17.4f} | {r['repetition_ratio']:17.4f} |")
    
    # Averages
    avg_pitch = np.mean([r["pitch_similarity"] for r in all_results])
    avg_rhythm = np.mean([r["rhythm_diversity"] for r in all_results])
    avg_repetition = np.mean([r["repetition_ratio"] for r in all_results])
    
    print("\n" + "-"*60)
    print(f"| AVERAGE | {avg_pitch:17.4f} | {avg_rhythm:17.4f} | {avg_repetition:17.4f} |")
    print("-"*60)
    
    # Interpretation
    print("\n📊 INTERPRETATION:")
    print(f"  • Pitch Similarity (lower = better genre match): {avg_pitch:.4f}")
    print(f"  • Rhythm Diversity (higher = more varied): {avg_rhythm:.4f}")
    print(f"  • Repetition Ratio (lower = less repetitive): {avg_repetition:.4f}")
    
    # Compare with random baseline (expected values)
    print("\n📈 Comparison with Random Baseline:")
    print(f"  • Random pitch similarity: ~1.5-2.0 (yours: {avg_pitch:.4f})")
    print(f"  • Random rhythm diversity: ~0.3-0.5 (yours: {avg_rhythm:.4f})")
    print(f"  • Random repetition ratio: ~0.6-0.8 (yours: {avg_repetition:.4f})")
    
    return all_results

def generate_human_survey_template(output_path):
    """
    Generate a template for human listening survey (Task 4)
    """
    survey_template = {
        "survey_name": "Music Generation Quality Assessment",
        "instructions": "Listen to each 30-second clip and rate on scale 1-5",
        "scale": {
            "1": "Very poor - sounds random/unmusical",
            "2": "Poor - some musical elements but disjointed",
            "3": "Average - recognizably musical but repetitive",
            "4": "Good - coherent and enjoyable",
            "5": "Excellent - indistinguishable from human composition"
        },
        "samples": [
            {
                "id": i,
                "file": f"generated_sample_{i}.mid",
                "genre_guess": "_____",
                "rating": "_____",
                "comments": "_____"
            }
            for i in range(1, 6)
        ]
    }
    
    with open(output_path / "human_survey_template.json", "w") as f:
        json.dump(survey_template, f, indent=2)
    
    print(f"✓ Human survey template saved to {output_path}/human_survey_template.json")

# Run evaluation
if __name__ == "__main__":
    DATA_PATH = Path("data/processed/task1")
    OUTPUT_PATH = Path("outputs/task1_samples")
    
    if DATA_PATH.exists() and OUTPUT_PATH.exists():
        results = evaluate_generated_samples(DATA_PATH, OUTPUT_PATH)
        
        # Generate survey template for Task 4
        generate_human_survey_template(OUTPUT_PATH)
        
        # Save metrics to JSON
        if results:
            with open(OUTPUT_PATH / "evaluation_metrics.json", "w") as f:
                json.dump(results, f, indent=2)
            print("\n✓ Metrics saved to evaluation_metrics.json")
    else:
        print("Data or output path not found!")