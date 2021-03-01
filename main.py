from asur import AudioFile
from matplotlib import pyplot as plt


file_path = "D:\\soundofai\\nsynth-guitar-subset\\test\\audio\\guitar_acoustic_010-023-127.wav"

audio_file = AudioFile(
    file_path=file_path,
    hop_size=64,
    frame_size=256,
    n_mels=256
)

audio_file.plot_spectrogram()
plt.show()


plt.plot(audio_file.mean_envelope(), label='Mean Envelope')
plt.show()
