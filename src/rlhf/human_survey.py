"""
Human Survey Interface for RLHF (Task 4)
Generates HTML form for listening and rating music
"""

import numpy as np
from pathlib import Path
import json
import sys
import torch

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import MusicTransformer

def generate_survey_samples():
    """Generate 20 samples for human evaluation"""
    
    DATA_PATH = Path("data/processed/task1")
    CHECKPOINT_PATH = Path("checkpoints/task3_transformer")
    OUTPUT_PATH = Path("outputs/rlhf_survey")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load model
    with open(DATA_PATH / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    vocab_size = metadata["vocab_size"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024
    )
    
    model_path = CHECKPOINT_PATH / "best_model.pt"
    if not model_path.exists():
        print(f"✗ No model found at {model_path}")
        print("Please train Transformer first (Task 3)")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load validation sequences for prompts
    val_seqs = np.load(DATA_PATH / "val_seqs_shifted.npy")
    
    # Generate 20 diverse samples
    print("🎵 Generating 20 samples for human survey...")
    samples = []
    
    for i in range(20):
        # Random prompt
        prompt_length = np.random.randint(64, 128)
        random_idx = np.random.randint(0, len(val_seqs))
        prompt_tokens = val_seqs[random_idx][:prompt_length]
        prompt = torch.LongTensor(prompt_tokens).unsqueeze(0).to(device)
        
        # Vary temperature for diversity
        temperature = float(np.random.uniform(0.8, 1.5))
        
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=384, temperature=temperature, top_k=50)
        
        generated_numpy = generated[0].cpu().numpy()
        
        # Save sample
        sample_path = OUTPUT_PATH / f"sample_{i+1}.npy"
        np.save(sample_path, generated_numpy)
        
        # Check if it has notes
        has_notes = bool(np.any((generated_numpy >= 21) & (generated_numpy <= 109)))
        
        samples.append({
            "id": i + 1,
            "file": str(sample_path),
            "temperature": temperature,
            "has_notes": has_notes,
            "length": int(len(generated_numpy))
        })
        
        print(f"  ✓ Sample {i+1}/20 | Temp: {temperature:.2f} | Notes: {has_notes}")
    
    # Save survey metadata
    with open(OUTPUT_PATH / "survey_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n✓ 20 samples generated for human survey")
    return samples

def create_survey_html():
    """Create HTML form for human evaluation"""
    
    OUTPUT_PATH = Path("outputs/rlhf_survey")
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Music Generation Quality Survey</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .sample { border: 1px solid #ccc; margin: 20px 0; padding: 20px; border-radius: 8px; }
        .rating { margin: 10px 0; }
        .rating label { margin-right: 20px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; margin-top: 20px; cursor: pointer; }
        textarea { width: 100%; margin-top: 10px; }
        h1 { color: #333; }
        .instructions { background-color: #f0f0f0; padding: 15px; border-radius: 8px; }
    </style>
</head>
<body>
    <h1>🎵 Music Generation Quality Survey</h1>
    
    <div class="instructions">
        <h3>Instructions:</h3>
        <p>Listen to each 30-second music sample and rate on scale 1-5:</p>
        <ul>
            <li><strong>1 - Very Poor</strong>: Sounds random, unmusical, completely disjointed</li>
            <li><strong>2 - Poor</strong>: Some musical elements but largely incoherent</li>
            <li><strong>3 - Average</strong>: Recognizably musical but repetitive or lacking structure</li>
            <li><strong>4 - Good</strong>: Coherent, enjoyable, sounds like real music</li>
            <li><strong>5 - Excellent</strong>: Indistinguishable from human composition</li>
        </ul>
        <p><em>Note: Due to tokenization, samples may lack clear melody - rate based on rhythmic coherence and structure.</em></p>
    </div>
    
    <form id="surveyForm">
"""
    
    for i in range(1, 21):
        html_content += f"""
        <div class="sample">
            <h3>Sample {i}</h3>
            <p><em>Sample {i} - Listen to the generated music (30 sec)</em></p>
            <div class="rating">
                <strong>Rating:</strong><br>
                <label><input type="radio" name="rating_{i}" value="1"> 1 - Very Poor</label>
                <label><input type="radio" name="rating_{i}" value="2"> 2 - Poor</label>
                <label><input type="radio" name="rating_{i}" value="3"> 3 - Average</label>
                <label><input type="radio" name="rating_{i}" value="4"> 4 - Good</label>
                <label><input type="radio" name="rating_{i}" value="5"> 5 - Excellent</label>
            </div>
            <div class="comments">
                <label>Comments (optional):</label><br>
                <textarea name="comments_{i}" rows="2" cols="50" placeholder="What did you like/dislike?"></textarea>
            </div>
        </div>
"""
    
    html_content += """
        <button type="button" onclick="submitSurvey()">Submit Survey</button>
    </form>
    
    <script>
        function submitSurvey() {
            const formData = {};
            for (let i = 1; i <= 20; i++) {
                const rating = document.querySelector(`input[name="rating_${i}"]:checked`);
                const comments = document.querySelector(`textarea[name="comments_${i}"]`);
                if (rating) {
                    formData[`sample_${i}`] = {
                        rating: parseInt(rating.value),
                        comments: comments ? comments.value : ""
                    };
                }
            }
            
            // Download as JSON
            const dataStr = JSON.stringify(formData, null, 2);
            const dataBlob = new Blob([dataStr], {type: "application/json"});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement("a");
            link.setAttribute("href", url);
            link.setAttribute("download", "human_survey_responses.json");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            alert("Survey responses downloaded! Please save the file.");
        }
    </script>
</body>
</html>
"""
    
    html_path = OUTPUT_PATH / "survey_form.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✓ Survey HTML saved to {html_path}")
    print(f"  Open this file in a browser to collect human ratings")

if __name__ == "__main__":
    print("="*60)
    print("TASK 4: RLHF - Human Survey Preparation")
    print("="*60)
    
    # Generate samples
    generate_survey_samples()
    
    # Create survey HTML
    create_survey_html()
    
    print("\n📋 Next Steps:")
    print("1. Open outputs/rlhf_survey/survey_form.html in browser")
    print("2. Have 10+ participants rate the 20 samples")
    print("3. Save each response as JSON file")
    print("4. Run reward_model.py to train on ratings")