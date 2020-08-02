import random
import pandas
from motion_features import get_pca
from motion_features import get_movement_features
from music_features import get_accentuation
import numpy as np
from sklearn.metrics import classification_report
from music_features import get_song_titles
from music_features import visualize_annotations
from matplotlib import pyplot as plt


def main():

#    visualize_annotations("Monument")

    df = 30

    directory_music = "annotations_all_levels"
    filter_all_metrical_levels = '[2-8]'
    music = get_accentuation(directory_music, filter_all_metrical_levels, df)

    directory_motion = "motion_phone"
    motion = get_movement_features(directory_motion, df)
    cut_off_beginning = 5
    pre_processed_data = pre_process(motion, cut_off_beginning, df)

    results = compute_TLCC(pre_processed_data, music)
    target_names = list(get_song_titles(directory_music))
    ### remove when complete
    target_names.append("YouDontKnowMyName")
    #####
    y_true = get_true_labels(motion, target_names)
    y_predicted = get_labels(classify_by_TLCC(results).values(), target_names)
    y_random = classify_randomly(len(motion), 0, 13)
    random_report = classification_report(y_true, y_random, target_names=target_names, output_dict=True, digits=4)
    print(random_report)

    print(classification_report(y_true, y_predicted, target_names=target_names, output_dict=True, digits=4))

    random_report_df = pandas.DataFrame.from_dict(random_report).transpose()
    random_report_df.to_csv('classification_results/classification_report_random.csv')
    report_tlcc = classification_report(y_true, y_predicted, target_names=target_names, output_dict=True, digits=4)
    tlcc_report_df = pandas.DataFrame(report_tlcc).transpose()
    tlcc_report_df.to_csv('classification_results/classification_report_tlcc.csv')


    plt.plot(music["GirlOnFire"]['accentuation'])
    plt.plot(pre_processed_data["motion_phone/GirlOnFire_#12_acc.csv"])
    plt.plot(pre_processed_data["motion_phone/GirlOnFire_#13_acc.csv"])
    plt.plot(pre_processed_data["motion_phone/GirlOnFire_#14_acc.csv"])
    plt.plot(pre_processed_data["motion_phone/GirlOnFire_#15_acc.csv"])
    plt.show()

    #print(report_tlcc)
    print("done")
    #TODO classify and use new music annotations

    #### compute DTW
    #### TODO compute FFT of accelerometer data and accentuation signal, common frequencies?


def pre_process(data, cut_off_in_s, df):
    #cut off first 5 seconds before
    truncated_data = cut_off(data, cut_off_in_s, df)
    pca_motion = get_pca(truncated_data, 1)
    return pca_motion


def cut_off(data, cut_off_in_s, df):
    truncated = {}
    cut_off_index = cut_off_in_s*df
    for filename, raw_data in data.items():
        truncated[filename] = raw_data[cut_off_index:]
    return truncated




def cross_correlation(music_data, motion_data):
    music = music_data['accentuation'].values
    #music_centered = zero_centering(music_d)
    motion_centered = motion_data#zero_centering(motion_data)
    cross_corr = np.correlate(music, motion_centered, mode='full')
    lag = cross_corr.argmax() #- (len(music_centered) - 1) TODO check this
    max_corr = cross_corr[lag]
    print("lag: " + str(lag))
    print("max corr: " + str(max_corr))
    return max_corr


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


def classify_by_TLCC(TLCC_results):
    classification = {}
    for key in TLCC_results.keys():
        corrs_per_motion_sample = TLCC_results[key]
        max_corr = max(corrs_per_motion_sample, key=corrs_per_motion_sample.get)
        classification[key] = max_corr
    return classification


def classify_randomly(n, start_id, end_id):
    y = []
    for i in range(0, n):
        y.append(random.randint(start_id, end_id))
    return y


def get_true_labels(motion, target_classes):
    true_labels = []
    for index in motion:
        title = index.split('_')[1].split('/')[1]
        true_labels.append(target_classes.index(title))
    return true_labels

def get_labels(results, target_classes):
    true_labels = []
    for result in results:
        true_labels.append(target_classes.index(result))
    return true_labels


def plot_accentuation_against_motion(sample_accentuation, motion):
    plt.plot(sample_accentuation['accentuation'])
    plt.plot(motion)
    plt.show()

if __name__ == "__main__":
    main()
