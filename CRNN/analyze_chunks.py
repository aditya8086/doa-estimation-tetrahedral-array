import soundfile as sf

filepath = r"C:\Users\csio\doa_project\chunks\dp01_base1_chunk0010.wav"
audio, sr = sf.read(filepath)
print("Shape:", audio.shape)
print("Sample rate:", sr)
print("Duration (s):", len(audio) / sr)
