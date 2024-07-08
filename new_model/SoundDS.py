import torch
from torch.utils.data import Dataset
import AudioUtil
from AudioUtil import *
# ----------------------------
# Sound Dataset
# ----------------------------


class SoundDS(Dataset):
    def __init__(self, df, data_path, audio_type, n_fft=2048, win_length=2048, hop_length=512, n_xxcc=128):
        self.df = df  # 路径集
        self.data_path = str(data_path)  # 数据路径（df的前缀），用来将df的路径变成绝对路径
        self.duration = 20000  # 持续时间
        self.sr = 8000  # 采样率
        self.channel = 1  # 声音的声道
        self.shift_pct = 0.4  # 偏移值，这个的作用是？
        self.audio_type = audio_type
        self.n_xxcc = n_xxcc
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    # 这个函数的作用可以说，使用下标时会调用此函数
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + self.df.loc[idx, 'path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'ID']
        # print("audio_file: ", audio_file)
        aud = AudioUtil.open(audio_file) # 返回幅度和采样率

        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.

        reaud = AudioUtil.resample(aud, self.sr)  # 单纯改变采样率，同时也改变了前一个参数
        rechan = AudioUtil.rechannel(reaud, self.channel)  # 改变声道
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)  # 填充信号为固定长度

        if self.audio_type == 'stft':
            mfcc = AudioUtil.make_stft(dur_aud, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            return mfcc, class_id

        elif self.audio_type == 'mfcc':
            mfcc = AudioUtil.make_mfcc(dur_aud, n_mfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            return mfcc, class_id

        elif self.audio_type == 'lfcc':
            lfcc = AudioUtil.make_lfcc(dur_aud, n_lfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            return lfcc, class_id

        elif self.audio_type == 'gfcc':
            gfcc = AudioUtil.make_gfcc(dur_aud, n_gfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            gfcc = np.expand_dims(gfcc, axis=0).astype(np.float32)
            return gfcc, class_id

        elif self.audio_type == 'ngcc':
            ngcc = AudioUtil.make_gfcc(dur_aud, n_gfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length)
            ngcc = np.expand_dims(ngcc, axis=0).astype(np.float32)
            return ngcc, class_id

        elif self.audio_type == 'msrcc':
            msrcc = AudioUtil.make_msrcc(dur_aud, n_msrcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length)
            msrcc = np.expand_dims(msrcc, axis=0).astype(np.float32)
            return msrcc, class_id

        elif self.audio_type == 'convLSTM_conv2d':
            mfcc = AudioUtil.make_lstm_mfcc_conv2d(dur_aud, n_mfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            # print(mfcc.size())
            return mfcc, class_id

        elif self.audio_type == "convLSTM_conv1d":
            mfcc = AudioUtil.make_lstm_mfcc_conv1d(dur_aud, n_mfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            # print(mfcc.size())
            return mfcc, class_id

        elif self.audio_type == "mf-lf-gf":
            mfcc = AudioUtil.make_mfcc(dur_aud, n_mfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            mfcc = torch.tensor(mfcc)

            lfcc = AudioUtil.make_lfcc(dur_aud, n_lfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)

            gfcc = AudioUtil.make_gfcc(dur_aud, n_gfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            gfcc = np.expand_dims(gfcc, axis=0).astype(np.float32)
            gfcc = torch.tensor(gfcc)

            mlg = torch.cat((mfcc, lfcc, gfcc), dim=0)

            return mlg, class_id

        elif self.audio_type == 'mf-lf-ng':
            mfcc = AudioUtil.make_mfcc(dur_aud, n_mfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length)
            mfcc = torch.tensor(mfcc)

            lfcc = AudioUtil.make_lfcc(dur_aud, n_lfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length)

            ngcc = AudioUtil.make_ngcc(dur_aud, n_gfcc=self.n_xxcc, n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length)
            ngcc = np.expand_dims(ngcc, axis=0).astype(np.float32)
            ngcc = torch.tensor(ngcc)

            mlg = torch.cat((mfcc, lfcc, ngcc), dim=0)

            return mlg, class_id