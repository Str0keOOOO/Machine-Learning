import numpy as np
from scipy.fftpack import fft
from scipy.signal import savgol_filter, welch, find_peaks_cwt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


"""滤波"""


def moving_average(data, window_size):
    # 常规方法进行平滑滤波
    smoothed_data = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        window = data[start:end]
        smoothed_data.append(sum(window) / len(window))
    return smoothed_data


def moving_average_conv(data, window_size):
    # 卷积平滑滤波
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode="same")
    return smoothed_data


def moving_average_sg(data, window_size, order):
    # SG滤波
    sg_data = savgol_filter(data, window_size, order)
    return sg_data


def denoised_with_svd(data, nlevel=8):
    """
    :param data: 需要降噪的原始数据,1D-array
    :param nlevel: 阶次
    :return:重构后的信号
    """
    N = len(data)
    A = np.lib.stride_tricks.sliding_window_view(data, (N // 2,))
    U, S, Vh = np.linalg.svd(A)
    # 重构信号
    X = np.zeros_like(A)
    for i in range(nlevel):
        X += Vh[i, :] * S[i] * U[:, i : i + 1]
    X = X.T
    data_recon = np.zeros(N)
    for i in range(N):
        a = 0
        m = 0
        for j1 in range(N // 2):
            for j2 in range(N // 2 + 1):
                if i == j1 + j2:
                    a = a + X[j1, j2]  # 把矩阵重构回一维信号
                    m = m + 1
        if m != 0:
            data_recon[i] = a / m
    return data_recon


"""
函数定义:fft,psd,autocorr取多个峰值
"""


# FFT
def get_fft_values(y_values_old, f_s):
    y_values = [x for x in y_values_old if np.isnan(x) == False]
    N = len(y_values)
    f_values = np.linspace(0.0, f_s / 2.0, N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0 : N // 2])
    return f_values, fft_values


def get_fft_n_peaks_values(data_values, f_s, window_size, order, n_peaks):
    fft_peaks_feature = np.ndarray((data_values.shape[1], n_peaks * 2))
    for i in range(data_values.shape[1]):
        f_values, fft_values = get_fft_values(data_values[:, i], f_s)
        fft_values_average = moving_average_sg(fft_values, window_size, order)
        peaks_index = find_peaks_cwt(fft_values_average, np.arange(1, 30))
        one_feature = np.append(
            f_values[peaks_index[0:n_peaks]], fft_values_average[peaks_index[0:n_peaks]]
        )
        fft_peaks_feature[i, :] = one_feature
    return fft_peaks_feature


# PSD
def get_psd_values(y_values_old, f_s):
    y_values = [x for x in y_values_old if np.isnan(x) == False]
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values


def get_psd_n_peaks_values(data_values, f_s, n_peaks):
    psd_peaks_feature = np.ndarray((data_values.shape[1], n_peaks * 2))
    for i in range(data_values.shape[1]):
        f_values, psd_values = get_psd_values(data_values[:, i], f_s)
        peaks_index = find_peaks_cwt(psd_values, np.arange(1, 10))
        one_feature = np.append(
            f_values[peaks_index[0:n_peaks]], psd_values[peaks_index[0:n_peaks]]
        )
        psd_peaks_feature[i, :] = one_feature
    return psd_peaks_feature


# Autocorrelation
def autocorr(x):
    result = np.correlate(x, x, mode="full")
    return result[len(result) // 2 :]


def get_autocorr_max_values(data_values, f_s):
    autocorr_max_feature = np.ndarray((data_values.shape[1], 2))
    for i in range(data_values.shape[1]):
        y_values_old = data_values[:, i]
        y_values = [x for x in y_values_old if np.isnan(x) == False]
        N = len(y_values)
        autocorr_values = autocorr(y_values)
        x_values = np.array([1.0 * jj / f_s for jj in range(0, N)])
        max_index = np.where(autocorr_values == np.max(autocorr_values))
        autocorr_max_feature[i, :] = np.append(
            x_values[max_index], autocorr_values[max_index]
        )
    return autocorr_max_feature


def get_time_values(data_values):
    # 最大值,最小值,峰峰值,峰值,均值,均方值,标准差,方差,RMS均方根,方根幅值
    data_values = np.nan_to_num(data_values)
    max_feature = np.max(data_values, axis=0)
    min_feature = np.min(data_values, axis=0)
    max_minus_min = max_feature - min_feature
    mean_feature = np.mean(data_values, axis=0)
    mean_square_feature = np.mean(data_values**2, axis=0)
    std_feature = np.std(data_values, axis=0)
    var_feature = np.var(data_values, axis=0)
    mean_sqrt_feature = np.sqrt(np.mean(data_values**2, axis=0))
    RMS_feature = (np.mean(np.abs(data_values), axis=0)) ** 2
    time_feature = np.array(
        [
            max_feature,
            min_feature,
            max_minus_min,
            mean_feature,
            mean_square_feature,
            std_feature,
            var_feature,
            mean_sqrt_feature,
            RMS_feature,
        ]
    )
    return time_feature.T


"""
提取特征

fft,psd,autocorr
最大值,最小值,峰峰值,峰值,均值,均方值,标准差,方差,RMS均方根,方根幅值
"""


def get_all_features(data_values, f_s, window_size, order, n_peaks=3):
    data_features_FFT = get_fft_n_peaks_values(
        data_values, f_s, window_size, order, n_peaks
    )
    data_features_PSD = get_psd_n_peaks_values(data_values, f_s, n_peaks)
    data_features_autocorr = get_autocorr_max_values(data_values, f_s)
    time_feature = get_time_values(data_values)  # 没有使用时序特征因为没用
    data_features = np.concatenate(
        (data_features_FFT, data_features_PSD, data_features_autocorr), axis=1
    )
    # 标准化或归一化,指的是每一列标准化或归一化,但不再能使用贝叶斯分类器
    scaler = MinMaxScaler()
    data_features = scaler.fit_transform(data_features)
    return data_features