from motion_features import get_jerk
from motion_features import get_pca
from motion_features import get_movement_features
from music_features import get_accentuation
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np


def main():
    sample_rate = 250
    directory_motion = "motion_phone"
    directory_music = "annotations"
    motion = get_movement_features(directory_motion, sample_rate)
    pre_processed_data = pre_process(motion)
    music = get_accentuation(directory_music, sample_rate)

    results = compute_TLCC(pre_processed_data, music)

    #TODO classify and use new music annotations

    #### compute DTW
    #### TODO compute FFT of accelerometer data and accentuation signal, common frequencies?


    #cross_correlation_false = signal.correlate(music_sample1, motion_sample_false, mode='same')
    #fig, (ax_music_sample1, ax_motion_sample_false, ax_cross_correlation_false) = pyplot.subplots(3, 1, sharex=True)
    #ax_music_sample1.plot(music_sample1)
    #ax_motion_sample_false.plot(motion_sample_false[:, 0])
    #ax_cross_correlation_false.plot(cross_correlation_false)

    #pyplot.show()

    #cross_correlation_false = signal.correlate(music_sample1, motion_sample_false[:, 0], mode='same') / 128


def pre_process(data):
    pca_motion = get_pca(data, 1)
    return pca_motion


def cross_correlation(music_data, motion_data):
    music_centered = zero_centering(music_data)
    motion_centered = zero_centering(motion_data)
    cross_corr = np.correlate(music_centered, motion_centered, mode='full')
    lag = cross_corr.argmax() #- (len(music_centered) - 1) TODO check this
    max_corr = cross_corr[lag]
    print("lag: " + str(lag))
    print("max corr: " + str(max_corr))
    return max_corr
    #plt.plot(cross_corr)
    #plt.show()
    #plt.plot(music_centered, 'r')
    #plt.plot(motion_centered, 'b')
    #plt.show()


def zero_centering(data):
    return data-np.mean(data)


def compute_TLCC(motion_data, music):
    results = {}
    for index in motion_data.keys():
        corrs_per_sample = {}
        for music_key in music.keys():
            corrs_per_sample[music_key] = cross_correlation(music[music_key], np.squeeze(motion_data[index]))
        results[index] = corrs_per_sample
    return results


if __name__ == "__main__":
    main()