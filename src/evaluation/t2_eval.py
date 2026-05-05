import numpy as np
from pathlib import Path
import json
from collections import Counter

def pitch_histogram_similarity(original_tokens, generated_tokens):
    """Calculate pitch histogram similarity - lower is better"""
    # Flatten to 1D
    original_tokens = np.array(original_tokens).flatten()
    generated_tokens = np.array(generated_tokens).flatten()
    
    # Extract pitch tokens (21-109 are note pitches)
    original_pitches = original_tokens[(original_tokens >= 21) & (original_tokens <= 109)]
    generated_pitches = generated_tokens[(generated_tokens >= 21) & (generated_tokens <= 109)]
    
    if len(original_pitches) == 0 or len(generated_pitches) == 0:
        return 2.0
    
    # Create histograms
    orig_hist, _ = np.histogram(original_pitches, bins=128, range=(0, 128))
    gen_hist, _ = np.histogram(generated_pitches, bins=128, range=(0, 128))
    
    # Normalize
    orig_hist = orig_hist / (len(original_pitches) + 1e-8)
    gen_hist = gen_hist / (len(generated_pitches) + 1e-8)
    
    # L1 distance
    similarity = np.sum(np.abs(orig_hist - gen_hist))
    
    return similarity

def rhythm_diversity_score(tokens):
    """Calculate rhythm diversity - higher is better"""
    tokens = np.array(tokens).flatten()
    
    # Find note onset positions
    note_positions = np.where((tokens >= 21) & (tokens <= 109))[0]
    
    if len(note_positions) < 2:
        return 0.0
    
    # Inter-onset intervals
    intervals = np.diff(note_positions)
    intervals = intervals[intervals > 0]
    
    if len(intervals) == 0:
        return 0.0
    
    unique_intervals = len(np.unique(intervals))
    total_intervals = len(intervals)
    
    return unique_intervals / total_intervals

def repetition_ratio(tokens, pattern_length=8):
    """Calculate repetition ratio - lower is better"""
    tokens = np.array(tokens).flatten()
    
    if len(tokens) < pattern_length * 2:
        return 1.0
    
    patterns = []
    stride = pattern_length // 2
    
    for i in range(0, len(tokens) - pattern_length, stride):
        pattern = tuple(tokens[i:i+pattern_length].tolist())
        patterns.append(pattern)
    
    if not patterns:
        return 0.0
    
    pattern_counts = Counter(patterns)
    repeated = sum(1 for count in pattern_counts.values() if count > 1)
    total = len(pattern_counts)
    
    return repeated / total if total > 0 else 0.0

def main():
    DATA_PATH = Path("data/processed/task1")
    TASK1_PATH = Path("outputs/task1_samples")
    TASK2_PATH = Path("outputs/task2_samples")
    
    # Load validation data
    val_seqs = np.load(DATA_PATH / "val_seqs_fixed.npy")
    print(f"✓ Loaded {len(val_seqs)} validation sequences")
    
    # Find samples
    task1_samples = list(TASK1_PATH.glob("generated_sample_*.npy"))
    task2_samples = list(TASK2_PATH.glob("vae_sample_*.npy"))
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    results = {}
    
    for task_name, samples, path in [("Task 1 (AE)", task1_samples, TASK1_PATH), 
                                      ("Task 2 (VAE)", task2_samples, TASK2_PATH)]:
        if not samples:
            print(f"\n⚠️ No samples found for {task_name}")
            continue
        
        print(f"\n📊 {task_name}:")
        print("-" * 50)
        print(f"{'Sample':<20} {'Pitch Sim↓':<12} {'Rhythm Div↑':<12} {'Repetition↓':<12}")
        print("-" * 56)
        
        task_results = []
        for sample_path in samples[:8]:  # Evaluate up to 8 samples
            gen_tokens = np.load(sample_path)
            
            # Compare with 10 random validation sequences
            pitch_sims = []
            for _ in range(10):
                val_idx = np.random.randint(0, len(val_seqs))
                val_tokens = val_seqs[val_idx]
                pitch_sims.append(pitch_histogram_similarity(val_tokens, gen_tokens))
            
            avg_pitch = np.mean(pitch_sims)
            rhythm_div = rhythm_diversity_score(gen_tokens)
            rep_ratio = repetition_ratio(gen_tokens)
            
            task_results.append({
                "name": sample_path.stem,
                "pitch": avg_pitch,
                "rhythm": rhythm_div,
                "repetition": rep_ratio
            })
            
            print(f"{sample_path.stem:<20} {avg_pitch:<12.4f} {rhythm_div:<12.4f} {rep_ratio:<12.4f}")
        
        # Averages
        avg_pitch = np.mean([r["pitch"] for r in task_results])
        avg_rhythm = np.mean([r["rhythm"] for r in task_results])
        avg_rep = np.mean([r["repetition"] for r in task_results])
        
        print("-" * 56)
        print(f"{'AVERAGE':<20} {avg_pitch:<12.4f} {avg_rhythm:<12.4f} {avg_rep:<12.4f}")
        
        results[task_name] = {
            "avg_pitch_similarity": float(avg_pitch),
            "avg_rhythm_diversity": float(avg_rhythm),
            "avg_repetition_ratio": float(avg_rep)
        }
    
    # Comparison
    if "Task 1 (AE)" in results and "Task 2 (VAE)" in results:
        print("\n" + "="*60)
        print("📈 COMPARISON")
        print("="*60)
        
        r1 = results["Task 1 (AE)"]
        r2 = results["Task 2 (VAE)"]
        
        rhythm_improvement = ((r2["avg_rhythm_diversity"] - r1["avg_rhythm_diversity"]) / (r1["avg_rhythm_diversity"] + 1e-8)) * 100
        rep_improvement = ((r1["avg_repetition_ratio"] - r2["avg_repetition_ratio"]) / (r1["avg_repetition_ratio"] + 1e-8)) * 100
        
        print(f"🎵 Rhythm Diversity: {r1['avg_rhythm_diversity']:.4f} → {r2['avg_rhythm_diversity']:.4f} ({rhythm_improvement:+.1f}%)")
        print(f"🔄 Repetition Ratio:  {r1['avg_repetition_ratio']:.4f} → {r2['avg_repetition_ratio']:.4f} ({rep_improvement:+.1f}%)")
        
        if rhythm_improvement > 0:
            print("\n✅ VAE successfully improved rhythm diversity!")
        else:
            print("\n⚠️ VAE did not improve rhythm diversity - consider tuning beta")
    
    # Save results
    with open("outputs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to outputs/evaluation_results.json")

if __name__ == "__main__":
    main()