import numpy as np
from pathlib import Path
from miditoolkit import MidiFile, Instrument, Note

NPY_FOLDER = Path("outputs/rlhf_comparison")
OUTPUT_FOLDER = Path("outputs/task4_midi")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

for npy_path in NPY_FOLDER.glob("*.npy"):
    print(f"Processing: {npy_path.name}")

    token_ids = np.load(npy_path).flatten()

    # Extract note tokens (21-109 are note pitches)
    note_tokens = token_ids[(token_ids >= 21) & (token_ids <= 109)]

    print(f"  Found {len(note_tokens)} note tokens out of {len(token_ids)} total")

    if len(note_tokens) == 0:
        print(f"  No note tokens found - skipping")
        continue

    # Create MIDI file
    midi = MidiFile()
    instrument = Instrument(0, is_drum=False, name="Generated Music")

    # Add notes with simple timing (120 ticks per note = quarter note)
    tick = 0
    notes_added = 0
    for pitch in note_tokens[:500]:  # Limit to 500 notes
        note = Note(
            velocity=80,
            pitch=int(pitch),
            start=tick,
            end=tick + 120
        )
        instrument.notes.append(note)
        tick += 120
        notes_added += 1

    midi.instruments.append(instrument)

    # Save
    midi_path = OUTPUT_FOLDER / npy_path.with_suffix('.mid').name
    midi.dump(midi_path)
    print(f"  ✓ Saved with {notes_added} notes to {midi_path}\n")

print(f"All MIDI files saved to {OUTPUT_FOLDER}")