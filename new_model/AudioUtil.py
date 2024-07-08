import random
import numpy as np

import librosa
from spafe.features import gfcc, msrcc, ngcc
import torch
import torchaudio
from spafe.utils.preprocessing import SlidingWindow
from torchaudio import transforms

from PyEMD import EMD


class AudioUtil:
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # 加载音频文件。将信号作为张量和采样率返回
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        # print("原始声音：", "sig: ", sig, " sig.shape: ", sig.shape, " sr: ", sr)
        return sig, sr

    # ----------------------------
    # Convert the given audio to the desired number of channels
    # 将给定的音频转换为所需数量的通道
    # ----------------------------

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if sig.shape[0] == new_channel:
            # Nothing to do
            return aud

        if new_channel == 1:
            # Convert from stereo to mono by selecting only the first channel
            # 只选择第一个声道，即可从立体声转换为单声道
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            # 通过复制第一个声道将单声道转换为立体声
            resig = torch.cat([sig, sig])
        # print("改变声道后：", "sig: ", resig, " sig.shape: ", resig.shape, " sr: ", sr)
        return resig, sr

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # 由于“重采样”应用于单个通道，因此我们一次重采样一个通道
    # ----------------------------

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if sr == newsr:
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :]) # 将一个信号重采样到另一个频率
        if num_channels > 1:
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        # print("改变采样率后：", "sig: ", resig, " sig.shape: ", resig.shape,  " sr: ", newsr)

        return resig, newsr

    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # 将信号填充（或截断）为固定长度的“max_ms”（以毫秒为单位）
    # ----------------------------

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms # 以毫秒为单位，sr以秒为单位

        # print("sig_len: ", sig_len, " max_len: ", max_len)

        if sig_len > max_len:
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len) # 生成0到max_len-sig_len之间的随机整数，前后随机填充，不是1：1
            pad_end_len = max_len - sig_len - pad_begin_len # 长度

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        # print("改变声音长度后：", "resig: ", sig, " resig.shape: ", sig.shape, " newsr: ", sr)
        return sig, sr

    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # 将信号向左或向右移动一定百分比。末尾的值被“环绕”到转换信号的开头。
    # ----------------------------

    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return sig.roll(shift_amt), sr

    # ----------------------------
    # Generate a Spectrogram
    # 生成音频的梅尔频谱图
    # ----------------------------

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # 通过在频率维度（即水平条）和时间维度（垂直条）上屏蔽光谱图的某些部分来增强光谱图，以防止过度拟合，并帮助模型更好地概括。遮罩部分将替换为平均值。
    # ----------------------------

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

    @staticmethod
    def frame(x, lframe, mframe):  # 定义分帧函数
        x = torch.squeeze(x)
        x = np.append(x[0], x[1:] - 0.97 * x[:-1])
        signal_length = len(x)  # 获取语音信号的长度

        fn = (signal_length - lframe) / mframe  # 分成fn帧
        fn1 = np.ceil(fn)  # 将帧数向上取整，如果是浮点型则加一
        fn1 = int(fn1)  # 将帧数化为整数

        # 求出添加的0的个数
        numfillzero = (fn1 * mframe + lframe) - signal_length
        # 生成填充序列
        fillzeros = torch.zeros(numfillzero)
        # 填充以后的信号记作fillsignal
        fillsignal = np.concatenate((x, fillzeros))  # concatenate连接两个维度相同的矩阵
        # 对所有帧的时间点进行抽取，得到fn1*

        d = np.tile(np.arange(0, lframe), (fn1, 1)) + np.tile(np.arange(0, fn1 * mframe, mframe), (lframe, 1)).T
        # 将d转换为矩阵形式（数据类型为int类型）
        d = np.array(d, dtype=np.int32)
        signal = fillsignal[d]

        return (signal, fn1, numfillzero)

    @staticmethod
    def make_stft(aud, n_fft=2048, win_length=2048, hop_length=512):
        sig, sr = aud

        # 预加重
        for i in range(1, len(sig)):
            sig[i] = sig[i] - 0.98 * sig[i - 1]
        sig = sig.numpy()

        """
            y.shape = (1,160000)
            默认:n_fft = 2048 win_length = 2048 hop_length = 512
                stft.shape = (1025, 313) (n_fft//2+1, np.ceil(160000/hp_length))
        
        """

        stft = librosa.stft(y=sig, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

        '''
        util.valid_audio(y, mono=False) 检查音频是否为np.ndarray、是否为有限值、是否有至少一个维度
        fft_window = get_window(window, win_length, fftbins=True) 加窗的窗函数
        fft_window = util.expand_to(fft_window, ndim=1 + y.ndim, axes=-2) 扩展维度，为什么扩展为1+y.ndim不是很清楚
        开始计算之前要对音频进行填充，左右各填充n_fft/2的值
        '''

        stft = np.real(stft)

        # print(stft.shape)

        return stft

    @staticmethod
    def make_mfcc(aud, n_fft=2048, win_length=2048, hop_length=512, n_mfcc=128):
        sig, sr = aud
        sig = sig.numpy()

        # 预加重
        for i in range(1, len(sig)):
            sig[i] = sig[i] - 0.98 * sig[i-1]

        # print(f"n_fft: {n_fft}  win_length: {win_length}  hop_length: {hop_length}  n_mfcc: {n_mfcc}  len(y):{len(sig[0])} sr: {sr}")

        S = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False, n_mels=128)

        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc)

        # print(mfcc.shape)

        return mfcc

    @staticmethod
    def make_lstm_mfcc_conv2d(aud, n_fft=2048, win_length=2048, hop_length=512, n_mfcc=128):
        # 此处音频有20s，因此对1s生成一个频谱图
        sig, sr = aud
        # print(sig.size(), sr)
        sig = sig.numpy()
        lstm_mfcc = []
        for i in range(20):
            # print(sig[0][i*sr: i*sr+sr])
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=sig[0][i*sr: i*sr+sr], sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length), ref=np.max)
            lstm_mfcc.append(librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc))
        lstm_mfcc = torch.tensor(np.array(lstm_mfcc))
        lstm_mfcc = torch.unsqueeze(lstm_mfcc, dim=1)
        return lstm_mfcc

    @staticmethod
    def make_lstm_mfcc_conv1d(aud, n_fft=2048, win_length=2048, hop_length=512, n_mfcc=128):
        # 此处音频有20s，因此对1s生成一个频谱图
        sig, sr = aud
        # print(sig.size(), sr)
        sig = sig.numpy()
        lstm_mfcc = []
        for i in range(20):
            # print(sig[0][i*sr: i*sr+sr])
            S = librosa.power_to_db(
                librosa.feature.melspectrogram(y=sig[0][i * sr: i * sr + sr], sr=sr, n_fft=n_fft, win_length=win_length,
                                               hop_length=hop_length), ref=np.max)
            lstm_mfcc.append(librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc))
        lstm_mfcc = torch.tensor(np.array(lstm_mfcc))
        return lstm_mfcc

    @staticmethod
    def make_lfcc(aud, n_fft=2048, win_length=2048, hop_length=512, n_lfcc=128):
        sig, sr = aud
        transform = transforms.LFCC(sample_rate=sr, n_lfcc=n_lfcc, speckwargs={"n_fft": n_fft,  "win_length": win_length, "hop_length": hop_length, "center": False})
        return transform(sig)

    @staticmethod
    def make_gfcc(aud, n_fft=2048, win_length=2048, hop_length=512, n_gfcc=128):
        sig, sr = aud
        return gfcc.gfcc(sig=sig, fs=sr, num_ceps=n_gfcc, nfilts=n_gfcc, nfft=n_fft, window=SlidingWindow(win_length/sr, hop_length/sr, "hamming")).T

    @staticmethod
    def make_ngcc(aud, n_fft=2048, win_length=2048, hop_length=512, n_gfcc=128):
        sig, sr = aud
        return ngcc.ngcc(sig=sig, fs=sr, num_ceps=n_gfcc, nfilts=n_gfcc, nfft=n_fft,
                         window=SlidingWindow(win_length / sr, hop_length / sr, "hamming")).T

    @staticmethod
    def make_msrcc(aud, n_fft=2048, win_length=2048, hop_length=512, n_msrcc=128):
        sig, sr = aud
        return msrcc.msrcc(sig=sig, fs=sr, num_ceps=n_msrcc, nfilts=n_msrcc, nfft=n_fft, window=SlidingWindow(win_length/sr, hop_length/sr, "hamming"), gamma=1).T

    @staticmethod
    def iceemd(data, num_imfs):
        emd = EMD()

        std = 0.2 * np.std(data)

        decompostition = emd.emd(data)

        print(decompostition.shape)

        imfs = decompostition[:num_imfs]

        print(imfs.shape)

        return imfs