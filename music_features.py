import pandas as pd
from matplotlib import pyplot
import numpy as np
import collections

from matplotlib.lines import Line2D
from scipy import signal
from glob import glob



# 1. merged version for half-note metrical level: accentuation on beats
# merge half-note-non syncopation and half-note syncopation
#-> version in annotations/half_note_level

# 2. merged version for beat level (syncopation on off-beats)
# merge half-note-non-syncopation till beat-level syncopation
#-> version in annotations/beat_level

# 3. merged version all accents over all levels:
# merge half-note-level to eighth note level into one
#-> version in annotations/eighth_note_level

# post-processing add zero accentuation in between every time interval
#-> version pre_processed_annotations/*_level

# how to relate these levels to acceleration?
# compare to norm of jerk instead of jerk?


def get_accentuation(directory, rate):
    annotation = read_annotation(directory)
    adjusted_annotation = adjust_sampling_rate(annotation, rate)
    return adjusted_annotation


def read_annotation(directory, filter, columns):
    annotations = {}
    file_names = glob(directory + filter)
    for filename in file_names:
        annotations[filename] = pd.DataFrame(pd.read_csv(filename, sep=',\s*',  usecols=columns, header=0))
    return annotations


def adjust_sampling_rate(data, rate):
    adjusted_data = {}
    for filename, df in data.items():
        timestamps = pd.to_datetime(df['tEnd'], unit='s')
        series = pd.Series(df['accentuation'].values, index=timestamps)
        newFreq = series.resample('10ms').asfreq()
        newSeries = series.reindex(newFreq.index, method='nearest', tolerance=pd.Timedelta('10ms')).interpolate(method='polynomial', order=2)
        newValues = signal.resample(newSeries, num=rate)
        data = np.column_stack((np.arange(0, 25, 25 / rate).transpose(), newValues))
        adjusted_data[filename] = pd.DataFrame(data, columns=['tEnd', 'accentuation'])
    return adjusted_data


def merge_levels(levels):
    first_key = list(levels.keys())[0]
    levels_merged = levels.pop(first_key)#remark just doing this whole thing because something had to be there to be merged in the first place
    for i, key in enumerate(levels, start=0):
        levels_merged = pd.concat([levels_merged, levels[key]]).sort_values(['tEnd'])
    levels_merged.reset_index(drop=True, inplace=True)
    key = first_key.split("_", 3)[2].split("/", 1)[1]
    dict = {}
    dict[key] = levels_merged
    return dict


def add_zeros_between_annotations(accentuation):
    pre_processed_accentuation = {}
    for filename in accentuation:
        current_annotation_series = accentuation[filename]
        processed_annotation_series = []

        for index, row in current_annotation_series.iterrows():
            current_time_slot = row['tEnd']
            accent = row['accentuation']
            # take the beginning of the annotation as accent
            processed_annotation_series.append([current_time_slot, accent])
            # add non accentuated part in the middle
            # between this and the next accentuated time slot
            if index < len(current_annotation_series) - 1:
                next_time_slot = current_annotation_series.get_value(index+1, 'tEnd')
                processed_annotation_series.append([(current_time_slot + next_time_slot) * 0.5, 0])
        df = pd.DataFrame(processed_annotation_series, columns=['tEnd', 'accentuation'])
        pre_processed_accentuation[filename] = df
    return pre_processed_accentuation


def visualize_annotations(song_title):
    columns = ['tEnd', 'accentuation']  # time points only have end time in praat
    ######## HALF NOTE LEVEL ######################################
    filter_half_note = '[2]'
    levels = read_annotation('annotations_all_levels', '/' + song_title + '*' + filter_half_note + '_.csv', columns)
    merged_levels = merge_levels(levels.copy())
    pre_processed_annotation = add_zeros_between_annotations(merged_levels)
    resampled = adjust_sampling_rate(pre_processed_annotation, 250)
    key_non_syncopated_2 = list(levels.keys())[0]
    key_syncopated_2 = list(levels.keys())[1]

    cols = ['half note level', 'beat level', 'eighth note level']
    fig, axs = pyplot.subplots(ncols=3, nrows=5, sharex=True)
    #for ax, col in zip(axs[0], cols):
    #    ax.set_title(col)

    points_non_syncopated = axs[0, 0].scatter(levels[key_non_syncopated_2]['tEnd'], levels[key_non_syncopated_2]['accentuation'])
    # axs[0, 0].set_title('non-syncopated')

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Scatter',
                              markerfacecolor='b', markersize=15)]
    axs[0, 0].legend(handles=legend_elements, loc='right')

    points_syncopated = axs[1, 0].scatter(levels[key_syncopated_2]['tEnd'], levels[key_syncopated_2]['accentuation'])
    # axs[1, 0].set_title('syncopated')
    axs[1, 0].legend(points_syncopated, ['syncopated'])


    list(merged_levels.values())[0].plot(x='tEnd', y='accentuation', ax=axs[2, 0])
    axs[2, 0].set_title('merged')
    list(pre_processed_annotation.values())[0].plot(x='tEnd', y='accentuation', ax=axs[3, 0])
    axs[3, 0].set_title('added zeros')
    resampled[list(resampled.keys())[0]].plot(x='tEnd', y='accentuation', ax=axs[4, 0])
    axs[4, 0].set_title('resampled')
    ############# BEAT LEVEL ##########################################
    filter_beat = '[2-4]'
    levels_4 = read_annotation('annotations_all_levels', '/' + song_title + '*' + filter_beat + '_.csv', columns)
    merged_levels_4 = merge_levels(levels_4.copy())
    pre_processed_annotation_4 = add_zeros_between_annotations(merged_levels_4)
    resampled_4 = adjust_sampling_rate(pre_processed_annotation_4, 250)
    key_syncopated_4 = list(levels_4.keys())[1]


    axs[0, 1].scatter(levels[key_non_syncopated_2]['tEnd'], levels[key_non_syncopated_2]['accentuation'])
    axs[0, 1].scatter(levels[key_syncopated_2]['tEnd'], levels[key_syncopated_2]['accentuation'])
    axs[0, 1].set_title('non-syncopated')

    axs[1, 1].scatter(levels_4[key_syncopated_4]['tEnd'], levels_4[key_syncopated_4]['accentuation'])
    axs[1, 1].set_title('syncopated')

    list(merged_levels_4.values())[0].plot(x='tEnd', y='accentuation', ax=axs[2, 1])
    axs[2, 1].set_title('merged')
    list(pre_processed_annotation_4.values())[0].plot(x='tEnd', y='accentuation', ax=axs[3, 1])
    axs[3, 1].set_title('added zeros')
    resampled_4[list(resampled_4.keys())[0]].plot(x='tEnd', y='accentuation', ax=axs[4, 1])
    axs[4, 1].set_title('resampled')

    ############# EIGHTH NOTE LEVEL ###################################
    filter_eighth_note = '[2-8]'


    pyplot.ylim((0, 1))
    pyplot.xlim((0, 25))
    pyplot.show()

song_title = 'Chandelier'
visualize_annotations(song_title)
x = 1