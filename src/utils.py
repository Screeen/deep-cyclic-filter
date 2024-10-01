import copy
from itertools import zip_longest
from pathlib import Path

# import librosa
import numpy as np
import scipy
import sys
import warnings
import sounddevice
import os

from matplotlib import pyplot as plt, ticker
from scipy.ndimage import uniform_filter1d


# generate_random_seed = False
# if generate_random_seed:
#     rnd_seed = int(np.random.rand() * (2 ** 32 - 1))
#     print(f"Random seed: {rnd_seed}")
# else:
#     # rnd_seed = 866369434
#     # rnd_seed = 1220931063
#     # rnd_seed = 122093106313
#     rnd_seed = 5924627591
#     print(f"Random seed: {rnd_seed}")
rng = np.random.default_rng(12345)
eps = np.finfo(float).eps
cmap = 'plasma'


def savefig(figure, file_name: Path, dpi=300, transparent=True):
    # If filename already exists, append a number to the end of the filename (but before extension), starting from -1
    # and increasing until a filename is found that does not exist.
    idx = 0
    file_name_stem = file_name.stem
    file_name = file_name.with_name(file_name_stem + f"_{idx}" + file_name.suffix)
    while file_name.exists():
        idx += 1
        file_name = file_name.with_name(file_name_stem + f"_{idx}" + file_name.suffix)

    figure.savefig(file_name, dpi=dpi, transparent=transparent,
                   bbox_inches='tight',
                   facecolor=figure.get_facecolor(), edgecolor=figure.get_edgecolor())

    print(f"Figure saved as {file_name}")

    return file_name


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


# @jit(nopython=True, cache=True)
def compute_correction_term(stft_shape, overlap, complex_stft=False):
    """
    Compute the correction term to account for delay in STFT. See for example Equation 3 in
    "Fast computation of the spectral correlation" by Antoni, 2017.

    :param complex_stft: bool, if True, then the input stft to correct is complex, otherwise it is real
    :param stft_shape: shape of the stft matrix, e.g. (num_mics, num, num_frames)
    :param overlap: a number between 0 and win_len-1, where win_len = (num_freqs-1)*2
    :return:
    """

    (num_freqs, num_frames) = stft_shape

    # If only positive ("real") frequencies are available, the window length is (num_freqs-1)*2
    win_len = num_freqs if complex_stft else (num_freqs - 1) * 2
    shift_samples = win_len - overlap

    # Range is [0, 0.5) (real part). With negative frequencies, the range would be [-0.5, 0.5).
    normalized_frequencies = np.arange(-num_freqs // 2, num_freqs // 2) if complex_stft else np.arange(0, num_freqs)
    normalized_frequencies = normalized_frequencies[:, np.newaxis] / win_len

    # Compute correction term using array broadcasting. Accounts for delay in STFT.
    time_frames = np.arange(0, shift_samples * num_frames, shift_samples)[np.newaxis, :]
    correction_term = np.exp(-2j * np.pi * normalized_frequencies * time_frames)

    return correction_term


# def plot(x, ax=None, title=''):
#     """For one or multiple 1-D plots, i.e. for time-domain plots."""
#
#     if 1:
#         is_subplot = ax is not None
#         if is_subplot:
#             fig = plt.gcf()
#         else:
#             fig, ax = plt.subplots(1, 1)
#
#     if x.ndim == 2 and x.shape[1] > x.shape[0]:
#         ax.plot(x.T)
#     else:
#         ax.plot(x)
#     ax.grid(True)
#
#     if title != '':
#         ax.set_title(title)
#
#     plt.show()
#     return fig
# def pad(array, pad_width, mode='constant', **kwargs):
def pad_last_dim(x, N_, prepad=False):
    assert x.ndim <= 2, "Only 1d and 2d arrays are supported."
    # Should work both for 1d and 2d arrays
    if N_ > x.shape[-1]:
        if not prepad:
            return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, N_ - x.shape[-1])])
        else:
            # Same as above, but the zeros should be before the signal
            return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(N_ - x.shape[-1], 0)])
    else:
        return x


def plot(x, ax=None, titles='', title='', fs=16000, time_axis=True, plot_config=None, subplot_height=0.8,
         show_fig=True, transform=None):
    """For one or multiple 1-D plots, i.e. for time-domain plots."""

    font_size = 'small'
    title_font_size = 'medium'

    """
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
        if isinstance(titles, list):
            titles = titles[0]
    """
    if titles == '' and title != '':
        titles = [title]

    if isinstance(x, list):
        num_plots = len(x)

        if transform is not None:
            x = [transform(item) for item in x]

        # sharex flag true if all arrays in x have the same length
        sharex = all(len(x[0]) == len(item) for item in x)

        max_len = max(item.shape[-1] for item in x)
        x_shape_before = [item.shape for item in x]
        x = [pad_last_dim(item, max_len) for item in x]
        x_shape_after = [item.shape for item in x]
        if not all(x_shape_before[i] == x_shape_after[i] for i in range(num_plots)):
            warnings.warn(f"Padding was applied to the input arrays to make them have the same length. "
                          " {x_shape_before = } -> {x_shape_after = }")

        fig_opt = dict(figsize=(6, 0.5 + num_plots * subplot_height), layout='compressed', squeeze=False)
        fig, axes = plt.subplots(num_plots, 1, sharey='all', sharex=sharex, **fig_opt)

        for ax, audio_sample, title in zip_longest(axes.flat, x, titles):
            plot(audio_sample, ax, fs=fs, time_axis=time_axis, plot_config=plot_config)
            ax.set_ylabel("Amplitude", fontsize=font_size)

            if time_axis:
                ax.set_xlabel("Time [s]", fontsize=font_size)
                x_locations, _ = ax.get_xticks(), ax.get_xticklabels()
                labels_str = [f"{x / fs:.2f}" for x in x_locations]
                ax.set_xticks(x_locations, labels_str)
                ax.set_xlim(0, audio_sample.shape[-1])

                num_x_ticks = 10
                x_locator = ticker.MaxNLocator(num_x_ticks)  # , integer=True
                x_minor_locator = ticker.AutoMinorLocator(4)  # 4
                y_minor_locator = ticker.AutoMinorLocator(2)  # 2
                ax.xaxis.set_major_locator(x_locator)
                ax.tick_params(axis='both', labelsize=font_size)
                ax.grid(which='both')

                # Change minor ticks to show every 5. (20/4 = 5)
                if x_minor_locator is not None:
                    ax.xaxis.set_minor_locator(x_minor_locator)
                ax.yaxis.set_minor_locator(y_minor_locator)
                ax.grid(which='major', color='#CCCCCC')
                ax.xaxis.grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=0.3)
                ax.yaxis.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=0.3)

            ax.set_title(title, fontsize=title_font_size)

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axes.flat:
            if sharex:
                ax.label_outer()
            ax.grid(True)

        if show_fig:
            fig.show()
        return fig

    else:
        is_subplot = ax is not None
        if is_subplot:
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(1, 1)
            ax.set_title(titles, fontsize=title_font_size)

    if plot_config is None:
        plot_config = dict()

    if transform is not None:
        x = transform(x)

    if x.ndim == 2 and x.shape[1] > x.shape[0]:
        ax.plot(x.T, **plot_config)
    else:
        ax.plot(x, **plot_config)

    ax.grid(True)

    if not is_subplot:
        plt.show()
        plt.pause(0.05)

    return fig


def plot_matrix(X_, title='', xy_label=('', ''), xy_ticks=None, log=True, show_figures=True,
                amp_range=(None, None), figsize=None, normalized=False):

    if np.allclose(X_, 0) or X_.size == 0:
        warnings.warn(f"X with {title = } and {X_.shape = } is zero. Skipping plot.")
        return None

    X = copy.deepcopy(X_)
    X = np.atleast_2d(np.squeeze(X))
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize)
    options = dict(cmap=cmap, antialiased=True)

    if amp_range[0] is not None:
        options['vmin'] = amp_range[0]
    if amp_range[1] is not None:
        options['vmax'] = amp_range[1]
    if 'vmin' in options and 'vmax' in options and options['vmin'] > options['vmax']:
        raise ValueError(f"Invalid amp_range: {amp_range}, vmin > vmax.")
    font_size = 11
    title_font_size = 14

    if normalized:
        X = X / (eps + np.max(X))

    if log:
        if X.size == 0:
            warnings.warn("X is empty. Cannot compute log.")
            return fig
        try:
            small_const = np.nanmin(np.abs(X)[np.abs(X) > 0]) / 100
        except ValueError:
            small_const = 1e-12
        X = 10 * np.log(small_const + np.abs(X) ** 2)

    if xy_ticks is not None:
        # X, Y = np.meshgrid(*xy_ticks)
        pcm_mag = ax.pcolormesh(*xy_ticks, X, **options)
    else:
        pcm_mag = ax.pcolormesh(X, **options)

    if title == '':
        title = f"Spectral correlation"
    ax.set_title(title, fontsize=title_font_size)

    cl = fig.colorbar(pcm_mag, ax=ax)
    label = 'Magnitude (dB)' if log else 'Magnitude'
    cl.set_label(label, size=font_size)
    cl.ax.tick_params(labelsize=font_size)

    # ax.invert_yaxis()
    ax.xaxis.set_ticks_position('both')

    # Tick label size
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    if xy_label != ('', ''):  # if not empty
        ax.set_xlabel(xy_label[0], fontsize=font_size)
        ax.set_ylabel(xy_label[1], fontsize=font_size)

    if show_figures:
        fig.show()
        plt.pause(0.05)

    return fig


def fig_to_subplot(existing_fig, title, ax, xy_ticks=(np.array([]), np.array([])), xlabel='', ylabel='',
                   font_size='x-large', title_font_size='xx-large'):

    if existing_fig is None or not existing_fig:
        return None

    # Retrieve the image data from the existing figure
    img = existing_fig.axes[0].collections[0].get_array().data

    # Retrieve vmin and vmax from the existing figure
    vmin, vmax = existing_fig.axes[0].collections[0].get_clim()

    # Retrieve the x ticks, y ticks, color map, and labels from the existing figure
    cmap = existing_fig.axes[0].collections[0].get_cmap()

    # x ticks and x ticks labels
    # x_ticks = existing_fig.axes[0].get_xticks()
    # x_ticks_labels = existing_fig.axes[0].get_xticklabels()
    # x_ticks_labels = np.array([item.get_text() for item in x_ticks_labels])

    # y ticks and y ticks labels
    # y_ticks = existing_fig.axes[0].get_yticks()
    # y_ticks_labels = existing_fig.axes[0].get_yticklabels()
    # y_ticks_labels = np.array([item.get_text() for item in y_ticks_labels])

    # xy_ticks = (x_ticks, y_ticks) if xy_ticks[0].size == 0 else xy_ticks

    # Display the image data in the new subplot
    if xy_ticks[0].size > 0:
        im = ax.pcolormesh(img, antialiased=True, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(xy_ticks[0])
        ax.set_yticks(xy_ticks[1])
    else:
        im = ax.pcolormesh(img, antialiased=True, vmin=vmin, vmax=vmax, cmap=cmap)

    if xlabel == '':
        xlabel = 'Cyclic freq.~$\\alpha_p$ [kHz]'

    if ylabel == '':
        ylabel = 'Freq.~$\\omega_k$ [kHz]'

    # Set the title of the subplot
    ax.set_title(title, fontsize=title_font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)

    return im


def stft(y, fs_=16000, Nw_=512, noverlap_samples_=0, complex_stft=False, window='hann', padding=False):
    # Had to disable boundary and padded to get the same results as in paper "A faster algorithm for the calculation
    # of the fast spectral correlation" by Borghesani, 2018.

    if padding:
        padded = True
        boundary = 'zeros'
    else:
        padded = False
        boundary = None

    _, _, y_stft = scipy.signal.stft(y, fs=fs_, window=window, nperseg=Nw_, noverlap=noverlap_samples_, detrend=False,
                                     return_onesided=not complex_stft,
                                     boundary=boundary, padded=padded, axis=-1)
    return y_stft


def istft(y_stft, fs_=16000, Nw_=512, noverlap_samples_=0, complex_stft=False):
    _, y = scipy.signal.istft(y_stft, fs=fs_, window='hann', nperseg=Nw_, noverlap=noverlap_samples_,
                              input_onesided=not complex_stft)
    return y


def set_printoptions_numpy():
    """ Set numpy print options to make it easier to read. Also set pprint as default for dict() """
    desired_width = 180  # 220
    np.set_printoptions(precision=2, linewidth=desired_width, suppress=True)

    # make warnings more readable
    def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
        return '%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)

    warnings.formatwarning = warning_on_one_line


def play(sound, fs=16000, max_length_seconds=5, normalize_flag=True, volume=0.75):
    sound_normalized = volume * normalize_volume(sound) if normalize_flag else sound
    max_length_samples = int(max_length_seconds * fs)
    blocking = True

    if sound_normalized.shape[-1] < int(1. * fs):
        sound_normalized = pad_last_dim(sound_normalized, int(1. * fs), prepad=True)

    if 2 <= sound_normalized.shape[0] < 10:  # multichannel input was given! Play first and last channel
        sounddevice.play(sound_normalized[(0, -1), :max_length_samples].T, fs, blocking=blocking)
    else:
        sounddevice.play(np.squeeze(sound_normalized)[:max_length_samples], fs, blocking=blocking)


def normalize_volume(x_samples, max_value=0.9):
    if np.max(np.abs(x_samples)) < 1e-6:
        warnings.warn(f"Skipping normalization as it would amplify numerical noise.")
        return x_samples
    else:
        return max_value * x_samples / np.max(np.abs(x_samples))


def check_create_folder(folder_name, parent_folder=None):
    if parent_folder is None:
        parent_folder = os.getcwd()
    else:
        if not parent_folder.exists():
            parent_folder.mkdir(parents=True)
    child_folder = os.path.join(parent_folder, folder_name)
    if not os.path.exists(child_folder):
        os.mkdir(child_folder)
    return child_folder


def set_plot_options(use_tex=False):
    plt.style.use('seaborn-v0_8-paper')

    if not use_tex:
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
    else:
        """
        This might be interesting at some point
        https://github.com/Python4AstronomersAndParticlePhysicists/PythonWorkshop-ICE/tree/master/examples/use_system_latex
        """
        plt.rcParams['text.usetex'] = True
        plt.rcParams["axes.formatter.use_mathtext"] = True
        font = {'family': 'serif',
                'size': 10,
                'serif': 'cmr10'
                }
        plt.rc('font', **font)
        plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}'  # for \text command


# markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
# linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
#           'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def plot_surface(z, x, y, title=None, xy_label=('', ''), xlim=None, ylim=None, show_figures=True):
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, np.abs(z), cmap=cmap, antialiased=True, rstride=1, cstride=1)

    if xy_label != ('', ''):
        ax.set_xlabel(xy_label[0])
        ax.set_ylabel(xy_label[1])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    num_ticks_desired = 5
    spacing_options = np.array([100, 200, 250, 300, 500, 600, 700, 750, 800, 900, 1000, 1500, 2000])

    # Use spacing option that leads closest to num_ticks_desired ticks
    best_option_idx_x = np.argmin(np.abs(spacing_options * num_ticks_desired - xlim[1] + xlim[0]))
    spacing_x = spacing_options[best_option_idx_x]
    x_ticks = np.arange(xlim[0], xlim[1], spacing_x).astype(int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks)

    best_option_idx_y = np.argmin(np.abs(spacing_options * num_ticks_desired - ylim[1] + ylim[0]))
    spacing_y = spacing_options[best_option_idx_y]
    y_ticks = np.arange(ylim[0], ylim[1], spacing_y).astype(int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)

    # Make tick labels smaller and closer to axis
    ax.tick_params(axis='x', which='major', pad=-2, labelsize=8)
    ax.tick_params(axis='y', which='major', pad=-2, labelsize=8)

    ax.set_zlabel('Magnitude (normalized)')
    ax.xaxis.labelpad = -3  # Position z-label closer to z-axis
    ax.yaxis.labelpad = -3  # Position z-label closer to z-axis
    ax.zaxis.labelpad = -10  # Position z-label closer to z-axis

    if title is not None:
        # Set title and position it close to the plot
        ax.set_title(title, pad=-20, fontsize=12, y=1)

    # Make panes transparent
    ax.xaxis.pane.fill = False  # Left pane
    ax.yaxis.pane.fill = False  # Right pane
    ax.zaxis.pane.fill = False  # Right pane

    # Remove grid lines
    ax.grid(False)

    ax.set_zticks([])
    ax.set_zticklabels([])

    # Transparent spines (axes lines). If we remove this, axes labels are not visible.
    # ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.view_init(elev=22, roll=0, azim=330)
    fig.subplots_adjust(wspace=0, hspace=0)

    if show_figures:
        fig.show()
        plt.pause(0.05)

    return fig


def is_debug_mode():
    if sys.gettrace() is None:
        return False
    return True


def reload(module_name):
    import importlib
    importlib.reload(module_name)


# Clips real and imag part of complex number independently
def clip_cpx(x, a_min, a_max):
    if np.iscomplexobj(x):
        x = np.clip(x.real, a_min, a_max) + 1j * np.clip(x.imag, a_min, a_max)
    else:
        x = np.clip(x, a_min, a_max)
    return x


def repeat_along_time_axis(x, num_reps=3):
    return np.concatenate([x] * num_reps, axis=-1)


def load_audio_file(audio_file_path, fs_, N_num_samples=-1, offset_seconds=0, smoothing_window=True,
                    repeat_signal_to_length=-1):
    # Load audio file and return signal and sampling frequency

    if not os.path.exists(audio_file_path):
        raise ValueError(f"File {audio_file_path} does not exist.")

    if repeat_signal_to_length != -1 and N_num_samples != -1 and N_num_samples < repeat_signal_to_length:
        raise ValueError(f"Cannot repeat signal to length {repeat_signal_to_length} when {N_num_samples = }.")

    # Load audio file
    s, fs = librosa.load(audio_file_path, sr=fs_)

    # Offset
    if offset_seconds * fs < s.shape[-1]:
        s = s[int(offset_seconds * fs):]

    if N_num_samples == -1:
        N_num_samples = s.shape[-1]

    if s.shape[-1] >= N_num_samples:  # If signal is too long, cut it
        s = s[:N_num_samples]
    elif s.shape[-1] < N_num_samples:
        warnings.warn(f"Signal is too short: {s.shape[-1]} samples, but {N_num_samples} samples are needed.")
        s = np.pad(s, (0, N_num_samples - s.shape[-1]))

    # Smoothen corners before repeating signal
    if smoothing_window:
        s = smoothen_corners(s, alpha=0.5)

    if repeat_signal_to_length > 0 and repeat_signal_to_length > N_num_samples:
        num_reps = int(np.ceil(repeat_signal_to_length / N_num_samples))
        s = repeat_along_time_axis(s, num_reps)
        if s.shape[-1] > repeat_signal_to_length:
            s = s[:repeat_signal_to_length]
        elif s.shape[-1] < repeat_signal_to_length:
            s = np.pad(s, (0, repeat_signal_to_length - s.shape[-1]))

    if np.max(np.abs(s)) < 1e-4:
        raise ValueError(f"Signal is too small for file {audio_file_path}: {np.max(np.abs(s)) = :.2f}. "
                         f"Select a different offset than {offset_seconds = } or a file other than {audio_file_path = }.")

    return s, fs


def smoothen_corners(s, alpha=0.5):
    # Apply Tukey window to smoothen corners. Higher alpha in Tukey window leads to more smoothing.
    # 0 is rectangular window, 1 is Hann window.

    y = copy.deepcopy(s)
    win_len = min(100, s.shape[-1] // 4)
    win = scipy.signal.windows.tukey(win_len, alpha=alpha)
    y[:win_len // 2] = s[:win_len // 2] * win[:win_len // 2]
    y[-win_len // 2:] = s[-win_len // 2:] * win[-win_len // 2:]
    return y


def generate_uniform_filtered_process(N_samples_, low=0, high=1, ma_order=10):
    x = rng.uniform(low, high, N_samples_)
    x = uniform_filter1d(x, size=ma_order)
    return x


def generate_gaussian_filtered_process(N_samples_, mean=0., variance=1., ma_order=10):
    x = rng.normal(mean, variance, N_samples_)
    x = uniform_filter1d(x, size=ma_order)
    return x


def circular_gaussian(shape):
    # Generate a circular Gaussian random matrix.
    # shape: tuple of integers, shape of the matrix.
    # Returns: complex-valued matrix with i.i.d. circular Gaussian entries.
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def generate_harmonic_process(freqs_hz, N_samples, fs, amplitudes_over_time=None, phases=None,
                              smooth_edges_window=False, amp_harmonic=0.5, var_harmonic=10,
                              correlated_amplitudes=False):
    """
    Generate a sinusoid with the given frequencies and number of samples.
    The phase is a uniform random variable.
    """

    def amp_generator_wss(mean_, variance_, ma_order_=1600):
        return generate_gaussian_filtered_process(N_samples, mean=mean_, variance=variance_, ma_order=ma_order_)

    def amp_generator_wss_lpf(mean_, variance_, max_freq_hz):
        x = rng.normal(mean_, variance_, N_samples)
        # Filter with the simplest LPF and cutoff at max_freq_hz
        x = scipy.signal.lfilter(*scipy.signal.butter(4, max_freq_hz / fs, 'low'), x)
        return x

    num_harmonics_ = freqs_hz.shape[0]
    discrete_times = np.arange(N_samples) / fs

    if phases is None:
        phases = rng.uniform(-np.pi, np.pi, num_harmonics_)

    if amplitudes_over_time is None:
        if not correlated_amplitudes:
            amplitudes_over_time = [amp_generator_wss(amp_harmonic, var_harmonic) for _ in range(num_harmonics_)]
        else:
            # amplitude_over_time = amp_generator_wss(amp_harmonic, var_harmonic, ma_order_=100000)
            amplitude_over_time = amp_generator_wss_lpf(amp_harmonic, var_harmonic, max_freq_hz=5)
            amplitudes_over_time = [rng.uniform(0.1, 2) * amplitude_over_time for _ in range(num_harmonics_)]

        amplitudes_over_time = np.array(amplitudes_over_time)

    if freqs_hz.ndim == 1:  # frequency is fixed over time
        z = np.sum(amplitudes_over_time * np.cos(2 * np.pi * freqs_hz[:, None] * discrete_times[None, :] + phases[:, None]),
                   axis=0)
    else:  # frequency varies over time
        z_individual = np.cos((2 * np.pi * np.cumsum(freqs_hz, axis=1)) / fs + phases[:, None]) * amplitudes_over_time
        z = np.sum(amplitudes_over_time * z_individual, axis=0)

    if smooth_edges_window:
        win = scipy.signal.windows.tukey(N_samples, alpha=0.1)
        z = z * win

    return z


def normalize_and_pad(x, pad_to_len_):
    x = normalize_volume(x)
    x = pad_last_dim(x, pad_to_len_, prepad=False)
    return x


def get_modulation_frequencies(f0_over_time, max_modulation_frequency_hz):
    alpha_vec_hz_ = np.array([0])

    fundamental_freq_hz_ = np.nan
    # std_f0 = np.nan
    if np.any(np.isfinite(f0_over_time)):
        fundamental_freq_hz_ = np.nanmean(f0_over_time)
        # std_f0 = np.nanstd(f0_over_time)
        alpha_vec_hz_ = np.array([n * fundamental_freq_hz_ for n in range(100)])
        alpha_vec_hz_ = alpha_vec_hz_[alpha_vec_hz_ < max_modulation_frequency_hz]

        # It is better to shift the signal to higher frequencies than lower ones
        # alpha_vec_hz_ = np.insert(alpha_vec_hz_, 0, -alpha_vec_hz_)
        alpha_vec_hz_ = -alpha_vec_hz_

        alpha_vec_hz_ = np.unique(alpha_vec_hz_)
        if alpha_vec_hz_.size == 0:
            alpha_vec_hz_ = np.array([0])

        # DEBUG
        # if std_f0 > threshold_hz_f0_std:
        #     alpha_vec_hz_ = np.array([0])

    # std_f0_list.append(std_f0)

    # Make sure that 0 is the first element of alpha_vec_hz_ (move it there if needed)
    # if alpha_vec_hz_[0] != 0:
    #     idx_0 = np.where(alpha_vec_hz_ == 0)[0]
    #     idx_other = np.where(alpha_vec_hz_ != 0)[0]
    #     alpha_vec_hz_ = np.concatenate([alpha_vec_hz_[idx_0], alpha_vec_hz_[idx_other]])

    # Sort alpha_vec_hz_ using the absolute values in ascending order
    alpha_vec_hz_ = np.r_[sorted(alpha_vec_hz_, key=lambda x: abs(x))]

    return alpha_vec_hz_, fundamental_freq_hz_


def extract_block_diag(X, block_size):
    num_blocks = X.shape[0] // block_size
    temp1 = np.array([X[block_size * i:block_size * (i + 1), block_size * i:block_size * (i + 1)] for i in range(num_blocks)])
    return scipy.linalg.block_diag(*temp1)


def corresponds_to_real_time_domain_signal(stft_matrix, tol=1e-10):
    """
    Checks if an STFT domain signal corresponds to a real time-domain signal.

    Parameters:
    stft_matrix (numpy.ndarray): The STFT matrix with dimensions (frequency_bins, time_frames).
    tol (float): Tolerance for floating-point comparison.

    Returns:
    bool: True if the STFT matrix corresponds to a real time-domain signal, False otherwise.
    """
    if stft_matrix.ndim != 3:
        raise ValueError("The input STFT matrix must have three dimensions: (num_mics, num_bins, num_frames) but "
                            f"found {stft_matrix.ndim} dimensions with {stft_matrix.shape = }.")

    # Get the number of frequency bins and time frames
    num_mics, num_bins, num_frames = stft_matrix.shape

    # The code above can be vectorized as follows:
    if not np.allclose(stft_matrix[:, 1:num_bins // 2, :], np.conj(stft_matrix[:, -1:num_bins // 2:-1, :]), atol=tol):
        print("Hermitian symmetry is not satisfied.")
        return False

    # Special case for the Nyquist frequency if it exists (only when num_bins is even)
    if num_bins % 2 == 0 and not np.all(np.isreal(stft_matrix[:, num_bins // 2, :])):
        print("Nyquist frequency is not real.")
        return False

    return True
