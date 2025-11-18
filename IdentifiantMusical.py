# identite_musicale_complete.py
import os
import shutil
import math
import numpy as np
from scipy.io import wavfile
from mido import Message, MidiFile, MidiTrack, MetaMessage
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import pygame
import librosa
import sounddevice as sd

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

# instrument MIDI mapping (pour MIDI only)
INSTRUMENTS = {"L1": 0, "L2": 24, "L3": 40, "M1": 73, "M2": 80}

# map chiffre -> note MIDI (C4..)
NOTE_MAP = {'0': 60, '1': 62, '2': 64, '3': 65, '4': 67,
            '5': 69, '6': 71, '7': 72, '8': 74, '9': 76}

# waveform per year (for WAV synth)
WAVEFORMS = {
    "L1": "sine",
    "L2": "square",
    "L3": "sawtooth",
    "M1": "triangle",
    "M2": "harmonic"  # richer harmonic content
}

SAMPLE_RATE = 44100
PREAMBLE = [60, 64]  # two fixed notes for sync
PREAMBLE_DUR = 0.4   # seconds each

# ----------------------------
# UTILITAIRES SYNTHÈSE WAV
# ----------------------------
def midi_to_freq(midi_note):
    """Convertit note MIDI en fréquence (Hz)."""
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

def waveform_sample(freq, dur, waveform="sine", sr=SAMPLE_RATE):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    if waveform == "sine":
        sig = np.sin(2 * np.pi * freq * t)
    elif waveform == "square":
        sig = np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == "sawtooth":
        # simple saw via additive synthesis of harmonics
        sig = np.zeros_like(t)
        max_h = 20
        for h in range(1, max_h):
            sig += (1.0 / h) * np.sin(2 * np.pi * freq * h * t)
        sig = sig * (2 / np.pi)
    elif waveform == "triangle":
        sig = np.zeros_like(t)
        max_h = 20
        for h in range(1, max_h, 2):
            sig += ((-1)**((h-1)//2) * (1.0 / (h*h))) * np.sin(2 * np.pi * freq * h * t)
        sig = sig * (8 / (np.pi**2))
    elif waveform == "harmonic":
        # sine + octave + small detuned partials
        sig = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * freq * 2 * t) \
              + 0.2 * np.sin(2 * np.pi * freq * 3 * t)
    else:
        sig = np.sin(2 * np.pi * freq * t)
    # windowed ADSR-like: short fade-in/out
    N = len(sig)
    fade = np.ones(N)
    ramp = int(0.01 * sr)
    if ramp*2 < N:
        fade[:ramp] = np.linspace(0, 1, ramp)
        fade[-ramp:] = np.linspace(1, 0, ramp)
    return (sig * fade).astype(np.float32)

def synthesize_sequence(notes, durations, waveform, tempo_bpm, filename_wav):
    """
    Génère un WAV simple à partir d'une séquence de notes (MIDI numbers) et durées (en beats).
    tempo_bpm donne le tempo; 1 beat = 60/bpm secondes.
    """
    seconds_per_beat = 60.0 / tempo_bpm
    out = np.array([], dtype=np.float32)
    for midi_note, dur_beats in zip(notes, durations):
        dur = dur_beats * seconds_per_beat
        freq = midi_to_freq(midi_note)
        sig = waveform_sample(freq, dur, waveform)
        # court silence entre notes
        silence = np.zeros(int(0.02 * SAMPLE_RATE), dtype=np.float32)
        out = np.concatenate([out, sig, silence])
    # normaliser
    if np.max(np.abs(out)) > 0:
        out = out / np.max(np.abs(out)) * 0.9
    wavfile.write(filename_wav, SAMPLE_RATE, (out * 32767).astype(np.int16))

# ----------------------------
# ENCODAGE : Générer MIDI + WAV
# ----------------------------
def generer_melodie_complet(genre, filiere, annee, numero):
    """Génère le MIDI 'identite_musicale.mid' et un WAV 'identite_musicale.wav' synthétique."""
    # VALIDATIONS simples déjà faites avant appel

    # Construire séquence notes/durée (en beats)
    rythme = RYTHMES[genre]
    bpm = TEMPOS[filiere]
    # notes: preamble + mapped digits
    notes = []
    durations = []

    # preamble
    for p in PREAMBLE:
        notes.append(p)
        durations.append(PREAMBLE_DUR * bpm / 60.0)  # convertir secondes->beats approx

    # numéro -> notes
    for i, ch in enumerate(numero):
        midi_note = NOTE_MAP[ch]
        notes.append(midi_note)
        # utiliser pattern rythmique : cycler dans rythme
        dur = rythme[i % len(rythme)]
        durations.append(dur)

    # Générer MIDI simple
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("track_name", name="Identite musicale"))
    track.append(Message('program_change', program=INSTRUMENTS[annee], time=0))
    # approximating timings: put note_on and note_off with delta times proportional to durations*480
    for midi_note, dur in zip(notes, durations):
        track.append(Message('note_on', note=int(midi_note), velocity=64, time=0))
        track.append(Message('note_off', note=int(midi_note), velocity=64, time=int(dur * 480)))
    mid.save("identite_musicale.mid")

    # Synth WAV using waveform mapped for year
    waveform = WAVEFORMS[annee]
    # durations are in beats -> synthesize with tempo bpm
    synth_filename = "identite_musicale.wav"
    synthesize_sequence(notes, durations, waveform, bpm, synth_filename)

    return notes, durations, bpm

# ----------------------------
# DÉCODAGE : from audio (wav/mp3) -> data
# ----------------------------
def freq_to_midi_approx(freq):
    """Convert freq en note MIDI approx (int)"""
    if freq <= 0 or np.isnan(freq):
        return None
    return int(round(69 + 12 * math.log2(freq / 440.0)))

def decode_audio_file(path_audio):
    """
    Retourne un dict estimé : { 'numero':..., 'filiere':..., 'genre':..., 'annee':... }
    Méthode heuristique basée sur:
      - onset/tempo detection pour tempo -> filière
      - durations pattern pour genre
      - spectral centroid / harmonicity pour 'année'
      - pitch detection pour numéro (après preamble)
    """
    y, sr = librosa.load(path_audio, sr=SAMPLE_RATE, mono=True)
    # 1) Estimer tempo (bpm)
    # use librosa.beat
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=100)
    except Exception:
        tempo = None

    # map tempo -> filiere (closest)
    filiere_est = None
    if tempo is not None:
        # find nearest key in TEMPOS
        closest = min(TEMPOS.items(), key=lambda kv: abs(kv[1] - tempo))
        filiere_est = closest[0]

    # 2) pitch tracking via piptrack
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr)
    detected_notes = []
    last_note = None
    # for each frame choose strongest freq
    for i in range(pitches.shape[1]):
        idx = mags[:, i].argmax()
        freq = pitches[idx, i]
        if freq > 0.0 and mags[idx, i] > 1e-6:
            midi_n = freq_to_midi_approx(freq)
            # collapse repeated frames into single note by change detection
            if last_note is None or abs(midi_n - last_note) > 0.5:
                detected_notes.append((times[i], midi_n))
                last_note = midi_n

    # 3) find preamble by searching for first two known notes (60,64) in detected_notes
    times_only = [t for (t, n) in detected_notes]
    mids_only = [n for (t, n) in detected_notes]
    numero_est = None
    if len(mids_only) >= 3:
        # search for preamble occurrence (60 then 64 within small time)
        pre_idx = None
        for i in range(len(mids_only)-1):
            if abs(mids_only[i] - 60) <= 1 and abs(mids_only[i+1] - 64) <= 1:
                pre_idx = i
                break
        if pre_idx is not None:
            # next notes after preamble correspond to digits
            digits = []
            for n in mids_only[pre_idx+2: pre_idx+10]:  # take up to 8 digits
                # map back using NOTE_MAP inverse tolerant match
                # compute nearest NOTE_MAP key
                inv = {v: k for k, v in NOTE_MAP.items()}
                # find nearest key
                nearest = min(inv.keys(), key=lambda kv: abs(kv - n))
                if abs(nearest - n) <= 2:
                    digits.append(inv[nearest])
                else:
                    digits.append('?')
            numero_est = ''.join(digits[:8])
        else:
            # if no preamble found, try to interpret first 8 detected notes as digits
            digits = []
            inv = {v: k for k, v in NOTE_MAP.items()}
            for n in mids_only[:8]:
                nearest = min(inv.keys(), key=lambda kv: abs(kv - n))
                digits.append(inv[nearest] if abs(nearest - n) <= 2 else '?')
            numero_est = ''.join(digits)
    else:
        numero_est = "?" * 8

    # 4) Rhythm classification -> genre
    # compute inter-onset durations (in seconds) between the first few notes after preamble (or first notes)
    durations_sec = []
    ref_times = None
    if len(detected_notes) >= 4:
        # choose region after preamble if exists
        if 'pre_idx' in locals() and pre_idx is not None:
            region = detected_notes[pre_idx+2: pre_idx+10]
        else:
            region = detected_notes[:10]
        ref_times = [t for (t, n) in region]
        durations_sec = np.diff(ref_times) if len(ref_times) >= 2 else []

    genre_est = None
    if len(durations_sec) >= 3:
        # normalize durations to beats using estimated tempo (if available)
        if tempo and tempo > 0:
            seconds_per_beat = 60.0 / tempo
            dur_beats = durations_sec / seconds_per_beat
        else:
            dur_beats = durations_sec / np.mean(durations_sec)  # relative
        # compute simple distance to prototypes
        proto = {k: np.array(v[:len(dur_beats)]) for k, v in RYTHMES.items() if len(v) >= len(dur_beats)}
        best = None
        best_score = 1e9
        for k, p in proto.items():
            # normalize both to unit scale
            p_norm = p / np.linalg.norm(p)
            d_norm = dur_beats / np.linalg.norm(dur_beats)
            score = np.linalg.norm(p_norm - d_norm)
            if score < best_score:
                best_score = score
                best = k
        genre_est = best
    else:
        genre_est = "inconnu"

    # 5) Timbre -> année estimation via spectral centroid/harmonicity
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    # rough heuristic thresholds (empirical)
    # compute centroids of reference simple waveforms to compare
    # create small probes (sine/square/saw/triangle/harmonic) at freq=440 for short time
    def centroid_of_waveform(wf):
        sample = waveform_sample(440, 0.3, waveform=wf, sr=sr)
        return librosa.feature.spectral_centroid(y=sample.astype(float), sr=sr).mean()
    refs = {yr: centroid_of_waveform(wf) for yr, wf in WAVEFORMS.items()}
    # pick nearest ref
    nearest_year = min(refs.items(), key=lambda kv: abs(kv[1] - centroid))[0]
    # nearest_year is like 'L1'/'L2' etc because WAVEFORMS keys are years
    # but WAVEFORMS uses year keys, so nearest_year is one of L1..M2
    # map key (string) to same
    annee_est = nearest_year

    result = {
        'numero': numero_est,
        'filiere': filiere_est if filiere_est else 'inconnu',
        'genre': genre_est if genre_est else 'inconnu',
        'annee': annee_est
    }
    return result

# ----------------------------
# LIVE RECORD (micro) -> temp WAV -> decode
# ----------------------------
def record_and_decode(duration_sec=6):
    messagebox.showinfo("Enregistrement", f"Enregistrement micro : parler/jouer la mélodie maintenant ({duration_sec}s).")
    try:
        rec = sd.rec(int(duration_sec * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        wavfile_temp = "temp_record.wav"
        wavfile.write(wavfile_temp, SAMPLE_RATE, rec)
        res = decode_audio_file(wavfile_temp)
        os.remove(wavfile_temp)
        return res
    except Exception as e:
        messagebox.showerror("Erreur micro", str(e))
        return None

# ----------------------------
# INTERFACE TKINTER
# ----------------------------
def valider_entree(genre, filiere, annee, numero):
    if not numero.isdigit() or len(numero) != 8:
        messagebox.showerror("Erreur", "Le numéro d'étudiant doit contenir exactement 8 chiffres.")
        return False
    if not (genre and filiere and annee):
        messagebox.showerror("Erreur", "Veuillez remplir tous les champs.")
        return False
    return True

# GUI actions
derniere_melodie = None
nom_derniere_melodie = None

def action_generer():
    genre = combo_genre.get()
    filiere = combo_filiere.get()
    annee = combo_annee.get()
    numero = entry_numero.get().strip()
    if not valider_entree(genre, filiere, annee, numero):
        return
    notes, durs, bpm = generer_melodie_complet(genre, filiere, annee, numero)
    global derniere_melodie
    derniere_melodie = {'notes': notes, 'durs': durs, 'bpm': bpm}
    messagebox.showinfo("Généré", "MIDI et WAV générés : identite_musicale.mid / identite_musicale.wav")

def action_play():
    if not os.path.exists("identite_musicale.wav"):
        messagebox.showwarning("Aucun WAV", "Génère d'abord la mélodie (WAV).")
        return
    pygame.mixer.init()
    try:
        pygame.mixer.music.load("identite_musicale.wav")
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showerror("Erreur lecture", str(e))

def action_save():
    if not os.path.exists("identite_musicale.mid"):
        messagebox.showwarning("Aucun fichier", "Génère d'abord une mélodie.")
        return
    nom = simpledialog.askstring("Nom de votre mélodie", "Entrez le nom de votre mélodie :")
    if not nom:
        return
    nom_fichier = "".join(c for c in nom if c.isalnum() or c in ('_', '-')) + ".mid"
    chemin = filedialog.asksaveasfilename(defaultextension=".mid", initialfile=nom_fichier,
                                          filetypes=[("Fichiers MIDI", "*.mid")])
    if not chemin:
        return
    # insérer meta track_name et sauvegarder
    mid = MidiFile("identite_musicale.mid")
    if len(mid.tracks) > 0:
        mid.tracks[0].insert(0, MetaMessage("track_name", name=nom))
    mid.save(chemin)
    messagebox.showinfo("Sauvegardé", f"Fichier MIDI sauvegardé sous {chemin}")

def action_decode_file():
    fichier = filedialog.askopenfilename(title="Choisir un fichier audio à décoder", filetypes=[("Audio files", "*.wav *.mp3 *.flac")])
    if not fichier:
        return
    res = decode_audio_file(fichier)
    if res:
        messagebox.showinfo("Décodage", f"Résultat :\nNuméro: {res['numero']}\nFilière: {res['filiere']}\nGenre: {res['genre']}\nAnnée: {res['annee']}")

def action_decode_live():
    # record ~6 sec
    try:
        duration = simpledialog.askinteger("Durée", "Durée de l'enregistrement (s)", initialvalue=6, minvalue=2, maxvalue=12)
    except Exception:
        duration = 6
    if duration is None:
        return
    messagebox.showinfo("Enregistrement", "Le micro va enregistrer pendant quelques secondes. Parle/joue la mélodie.")
    try:
        rec = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        wavfile_temp = "temp_live.wav"
        wavfile.write(wavfile_temp, SAMPLE_RATE, rec)
        res = decode_audio_file(wavfile_temp)
        os.remove(wavfile_temp)
        if res:
            messagebox.showinfo("Décodage live", f"Résultat :\nNuméro: {res['numero']}\nFilière: {res['filiere']}\nGenre: {res['genre']}\nAnnée: {res['annee']}")
    except Exception as e:
        messagebox.showerror("Erreur", str(e))

# Build GUI
root = tk.Tk()
root.title("Identité musicale - Encodage & Décodage complet")
root.geometry("520x640")
root.configure(bg="#f4f6f8")
style = ttk.Style()
style.configure("TLabel", background="#f4f6f8", font=("Arial", 10))
style.configure("TButton", font=("Arial", 11, "bold"), padding=6)

ttk.Label(root, text="Genre :").pack(pady=5)
combo_genre = ttk.Combobox(root, values=list(RYTHMES.keys()), state="readonly")
combo_genre.pack()

ttk.Label(root, text="Filière :").pack(pady=5)
combo_filiere = ttk.Combobox(root, values=list(TEMPOS.keys()), state="readonly")
combo_filiere.pack()

ttk.Label(root, text="Année :").pack(pady=5)
combo_annee = ttk.Combobox(root, values=list(INSTRUMENTS.keys()), state="readonly")
combo_annee.pack()

ttk.Label(root, text="Numéro étudiant (8 chiffres) :").pack(pady=5)
entry_numero = ttk.Entry(root)
entry_numero.pack()

ttk.Button(root, text="🎶 Générer (MIDI + WAV)", command=action_generer).pack(pady=10)
ttk.Button(root, text="▶️ Jouer WAV", command=action_play).pack(pady=6)
ttk.Button(root, text="💾 Enregistrer (MIDI avec titre)", command=action_save).pack(pady=6)

ttk.Separator(root, orient="horizontal").pack(fill="x", pady=12)
ttk.Label(root, text="--- Décodage ---").pack(pady=6)
ttk.Button(root, text="📂 Décoder depuis fichier audio", command=action_decode_file).pack(pady=6)
ttk.Button(root, text="🎤 Décoder depuis micro (live)", command=action_decode_live).pack(pady=6)

ttk.Label(root, text="(Le décodage utilise heuristiques. Pour fiabilité, jouer la WAV synthétisée générée par le programme.)", wraplength=480).pack(pady=12)

root.mainloop()
