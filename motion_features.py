from numpy import genfromtxt
from glob import glob
from numpy import gradient
from scipy import signal
from sklearn.decomposition import PCA


def get_movement_features(directory, rate):
    motion = get_motion_data(directory)
    #jerk = get_jerk(motion)
    adjusted_motion_features = get_adjusted_motion(motion, rate)
    return adjusted_motion_features


def get_adjusted_motion(data, rate):
    adjusted_jerk_data = {}
    for filename, jerk_data in data.items():
        adjusted_jerk_data[filename] = signal.resample(jerk_data, num=rate)
    return adjusted_jerk_data


def get_motion_data(directory):
    motion_data = {}
    file_names = glob(directory+'/*acc.csv')
    for file in file_names:
        data = genfromtxt(file, delimiter=',')
        motion_data[file] = data
    return motion_data


def get_pca(data, n_components):
    pca_data = {}
    for filename, raw_data in data.items():
        pca = PCA(n_components=n_components, whiten=True)
        pca_data[filename] = pca.fit_transform(raw_data)
    return pca_data


def get_jerk(motion_data):
    jerk_data = {}
    for filename, raw_data in motion_data.items():
        jerk_data[filename] = gradient(raw_data, axis=0)
    return jerk_data
