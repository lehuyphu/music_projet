# IdentifiantMusical.py
import numpy as np
from scipy.io import wavfile
from mido import Message, MidiFile, MidiTrack, MetaMessage

# ----------------------------
# CONFIGURATION / MAPPINGS
# ----------------------------
RYTHMES = {
    "homme": [1.0, 0.5, 0.5, 1.0],
    "femme": [0.5, 0.5, 1.0, 1.0],
    "non_binaire": [0.75, 0.75, 0.75, 0.75]
}

TEMPOS = {
    "Informatique": 110,
    "Informatique et Science de l'Ingénieur": 120,
    "Maths": 90,
    "Maths Info": 100,
    "Physique Chimie": 130,
    "Physique": 140
}

INSTRUMENTS = {"L1": 0, "L2": 24, "L3": 40, "M1": 73, "M2": 80}

NOTE_MAP = {'0': 60, '1': 62, '2': 64, '3': 65, '4': 67,
            '5': 69, '6': 71, '7': 72, '8': 74, '9': 76}

WAVEFORMS = {
    "L1": "sine",
    "L2": "square",
    "L3": "sawtooth",
    "M1": "triangle",
    "M2": "harmonic"
}

SAMPLE_RATE = 44100
PREAMBLE = [60, 64]
PREAMBLE_DUR = 0.4  # seconds

# ----------------------------
# UTILITAIRES SYNTHÈSE WAV
# ----------------------------
def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

def waveform_sample(freq, dur, waveform="sine", sr=SAMPLE_RATE):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    if waveform == "sine":
        sig = np.sin(2 * np.pi * freq * t)
    elif waveform == "square":
        sig = np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == "sawtooth":
        sig = np.zeros_like(t)
        for h in range(1, 20):
            sig += (1.0 / h) * np.sin(2 * np.pi * freq * h * t)
        sig = sig * (2 / np.pi)
    elif waveform == "triangle":
        sig = np.zeros_like(t)
        for h in range(1, 20, 2):
            sig += ((-1)**((h-1)//2) * (1.0 / (h*h))) * np.sin(2 * np.pi * freq * h * t)
        sig = sig * (8 / (np.pi**2))
    elif waveform == "harmonic":
        sig = np.sin(2 * np.pi * freq * t) + 0.5*np.sin(2*np.pi*freq*2*t) + 0.2*np.sin(2*np.pi*freq*3*t)
    else:
        sig = np.sin(2 * np.pi * freq * t)
    N = len(sig)
    fade = np.ones(N)
    ramp = int(0.01 * sr)
    if ramp*2 < N:
        fade[:ramp] = np.linspace(0, 1, ramp)
        fade[-ramp:] = np.linspace(1, 0, ramp)
    return (sig * fade).astype(np.float32)

def synthesize_sequence(notes, durations, waveform, tempo_bpm, filename_wav):
    seconds_per_beat = 60.0 / tempo_bpm
    out = np.array([], dtype=np.float32)
    for midi_note, dur_beats in zip(notes, durations):
        dur = dur_beats * seconds_per_beat
        freq = midi_to_freq(midi_note)
        sig = waveform_sample(freq, dur, waveform)
        silence = np.zeros(int(0.02 * SAMPLE_RATE), dtype=np.float32)
        out = np.concatenate([out, sig, silence])
    if np.max(np.abs(out)) > 0:
        out = out / np.max(np.abs(out) ) * 0.9
    wavfile.write(filename_wav, SAMPLE_RATE, (out * 32767).astype(np.int16))

# ----------------------------
# GÉNÉRATION MIDI + WAV
# ----------------------------
def generer_melodie_complet(genre, filiere, annee, numero):
    rythme = RYTHMES[genre]
    bpm = TEMPOS[filiere]
    notes = []
    durations = []

    # preamble
    for p in PREAMBLE:
        notes.append(p)
        durations.append(PREAMBLE_DUR * bpm / 60.0)
    
    # numéro -> notes
    for i, ch in enumerate(numero):
        midi_note = NOTE_MAP[ch]
        notes.append(midi_note)
        dur = rythme[i % len(rythme)]
        durations.append(dur)
    
    # MIDI
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("track_name", name="Identite musicale"))
    track.append(Message('program_change', program=INSTRUMENTS[annee], time=0))
    for midi_note, dur in zip(notes, durations):
        track.append(Message('note_on', note=int(midi_note), velocity=64, time=0))
        track.append(Message('note_off', note=int(midi_note), velocity=64, time=int(dur*480)))
    mid.save("identite_musicale.mid")

    # WAV
    waveform = WAVEFORMS[annee]
    synthesize_sequence(notes, durations, waveform, bpm, "identite_musicale.wav")

    return notes, durations, bpm
