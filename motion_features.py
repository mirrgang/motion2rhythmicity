from numpy import genfromtxt
from glob import glob
from numpy import gradient
from scipy import signal


def get_movement_features(directory, rate):
    motion = get_motion_data(directory)
    jerk = get_jerk(motion)
    adjusted_jerk = get_adjusted_motion(jerk, rate)
    return adjusted_jerk

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


def get_jerk(motion_data):
    jerk_data = {}
    for filename, raw_data in motion_data.items():
        jerk_data[filename] = gradient(raw_data, axis=0)
    return jerk_data
