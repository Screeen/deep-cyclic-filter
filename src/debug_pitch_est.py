import numpy as np
import penn
import torchaudio

# Load audio
audio, sample_rate = torchaudio.load('../test.wav')

# Here we'll use a 10 millisecond hopsize
hopsize = .01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Choose a gpu index to use for inference. Set to None to use cpu.
gpu = None

# If you are using a gpu, pick a batch size that doesn't cause memory errors
# on your gpu
batch_size = 2048

# Select a checkpoint to use for inference. Selecting None will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = None

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
center = 'half-hop'

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065

# (Optional) Select a decoding method. One of ['argmax', 'pyin', 'viterbi'].
decoder = 'viterbi'

# Infer pitch and periodicity
pitch, periodicity = penn.from_audio(
    audio,
    sample_rate,
    hopsize=hopsize,
    fmin=fmin,
    fmax=fmax,
    checkpoint=checkpoint,
    batch_size=batch_size,
    center=center,
    interp_unvoiced_at=interp_unvoiced_at,
    gpu=gpu)

# Using same hopsize, compute spectrogram
spec = torchaudio.transforms.Spectrogram(
    n_fft=2048,
    hop_length=int(hopsize * sample_rate),
    power=None)(audio)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, squeeze=True, figsize=(6, 8))
ax[0].plot(pitch.numpy()[0])
ax[0].set_title('Pitch')
ax[1].plot(periodicity.numpy()[0])
ax[1].set_title('Periodicity')

# in ax[2], plot the spectrogram
ax[2].imshow(np.log10(np.abs(spec[0, :250])), aspect='auto', origin='lower')

# overlay pitch on spectrogram (convert from hz to spectral bin)
pitch_print = pitch.numpy()[0]
pitch_print[periodicity.numpy()[0] < .2] = 0
print(pitch_print)
pitch_bin = pitch_print * 2048 / sample_rate
ax[2].plot(pitch_bin, color='r', linewidth=0.5)

fig.tight_layout()
plt.show()




