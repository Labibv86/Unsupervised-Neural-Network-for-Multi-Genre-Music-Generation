"""
Generate synthetic human ratings for RLHF demonstration
Simulates realistic human preferences based on musical features
"""

import numpy as np
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

def calculate_music_quality(tokens):
    """Calculate synthetic quality score based on musical features"""
    
    tokens = np.array(tokens).flatten()
    
    # Features that correlate with "good" music
    has_notes = np.any((tokens >= 21) & (tokens <= 109))
    
    # Note density (more notes = more interesting, but not too many)
    note_positions = np.where((tokens >= 21) & (tokens <= 109))[0]
    note_density = len(note_positions) / len(tokens) if len(tokens) > 0 else 0
    
    # Repetition (less repetition = more creative)
    from collections import Counter
    patterns = []
    pattern_len = 8
    for i in range(0, len(tokens) - pattern_len, pattern_len):
        pattern = tuple(tokens[i:i+pattern_len].tolist())
        patterns.append(pattern)
    
    if patterns:
        pattern_counts = Counter(patterns)
        repetition_bonus = 1 - (sum(1 for c in pattern_counts.values() if c > 1) / len(pattern_counts))
    else:
        repetition_bonus = 0.5
    
    # Rhythmic variety
    if len(note_positions) > 1:
        intervals = np.diff(note_positions)
        rhythm_variety = min(1.0, len(np.unique(intervals)) / 10)
    else:
        rhythm_variety = 0
    
    # Calculate score (1-5 scale)
    score = 2.0  # Base score
    
    if has_notes:
        score += 0.5
        score += note_density * 2
        score = min(score, 4.5)
    
    score += repetition_bonus * 1.0
    score += rhythm_variety * 1.0
    
    # Add noise for realism
    score += np.random.normal(0, 0.3)
    
    # Clamp to 1-5 range
    score = np.clip(score, 1.0, 5.0)
    
    return score

def generate_mock_ratings():
    """Generate synthetic ratings for all survey samples"""
    
    SURVEY_PATH = Path("outputs/rlhf_survey")
    
    # Load survey samples
    samples_file = SURVEY_PATH / "survey_samples.json"
    if not samples_file.exists():
        print("✗ survey_samples.json not found!")
        print("Please run human_survey.py first to generate samples")
        return None
    
    with open(samples_file, 'r') as f:
        samples = json.load(f)
    
    print("🎵 Generating synthetic human ratings...")
    print("-" * 40)
    
    ratings = {}
    
    for sample in samples:
        sample_id = f"sample_{sample['id']}"
        sample_path = Path(sample['file'])
        
        if sample_path.exists():
            tokens = np.load(sample_path)
            quality_score = calculate_music_quality(tokens)
            
            ratings[sample_id] = quality_score  # Direct float, not dict
            
            print(f"  {sample_id}: {quality_score:.2f}/5 | Notes: {sample['has_notes']} | Temp: {sample['temperature']:.2f}")
        else:
            print(f"  ⚠ {sample_id}: File not found")
            ratings[sample_id] = 3.0  # Default rating
    
    # Save average ratings
    with open(SURVEY_PATH / "average_ratings.json", 'w') as f:
        json.dump(ratings, f, indent=2)
    
    print(f"\n✓ Generated average ratings for {len(ratings)} samples")
    print(f"✓ Average rating: {np.mean(list(ratings.values())):.2f}/5")
    print(f"✓ Rating range: {min(ratings.values()):.1f} - {max(ratings.values()):.1f}")
    
    return ratings

if __name__ == "__main__":
    print("="*60)
    print("Generating Mock Human Ratings (RLHF Demonstration)")
    print("="*60)
    
    ratings = generate_mock_ratings()
    
    if ratings:
        print("\n✅ Mock ratings ready for reward model training!")
        print("\nNext step: python src/rlhf/reward_model.py")