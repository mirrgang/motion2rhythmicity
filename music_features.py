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


def get_accentuation(directory, filter, rate):
    columns = ['tEnd', 'accentuation']  # time points only have end time in praat
    levels = read_annotation(directory, '/*' + filter + '_.csv', columns)
    merged_levels = merge_levels(levels.copy())
    pre_processed_annotation = add_zeros_between_annotations(merged_levels)
    resampled = adjust_sampling_rate(pre_processed_annotation, rate)
    return resampled


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

    fig, axs = pyplot.subplots(ncols=3, nrows=5, sharex=True)

    axs[0, 0].set_title('half note level')
    axs[0, 0].scatter(levels[key_non_syncopated_2]['tEnd'], levels[key_non_syncopated_2]['accentuation'], c='b', marker='.')
    #---------- legend non-syncopated------------
    legend_elements = [Line2D([0], [0], marker='.', color='w', label='non-syncopated',
                              markerfacecolor='b', markersize=15)]
    axs[0, 0].legend(handles=legend_elements, loc='right')
    #---------------------------------------------
    axs[1, 0].scatter(levels[key_syncopated_2]['tEnd'], levels[key_syncopated_2]['accentuation'], c='b', marker='.')
    # ---------- legend syncopated------------
    legend_elements_syncopated = [Line2D([0], [0], marker='.', color='w', label='syncopated',
                              markerfacecolor='b', markersize=15)]
    axs[1, 0].legend(handles=legend_elements_syncopated, loc='right')
    # ---------------------------------------------
    list(merged_levels.values())[0].plot(x='tEnd', y='accentuation', ax=axs[2, 0], c='b')
    # ---------- legend merged------------
    legend_elements = [Line2D([0], [0], color='b', label='merged')]
    axs[2, 0].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------
    list(pre_processed_annotation.values())[0].plot(x='tEnd', y='accentuation', ax=axs[3, 0], c='b')
    # ---------- legend added zeros------------
    legend_elements = [Line2D([0], [0], color='b', label='added zeros')]
    axs[3, 0].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------
    resampled[list(resampled.keys())[0]].plot(x='tEnd', y='accentuation', ax=axs[4, 0], c='b')
    # ---------- legend resampled------------
    legend_elements = [Line2D([0], [0], color='b', label='resampled')]
    axs[4, 0].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------

    ############# BEAT LEVEL ##########################################
    filter_beat = '[2-4]'
    levels_4 = read_annotation('annotations_all_levels', '/' + song_title + '*' + filter_beat + '_.csv', columns)
    merged_levels_4 = merge_levels(levels_4.copy())
    pre_processed_annotation_4 = add_zeros_between_annotations(merged_levels_4)
    resampled_4 = adjust_sampling_rate(pre_processed_annotation_4, 250)
    search_key = 'syncopation_4'
    key_syncopated_4 = [key for key in levels_4.keys() if search_key in key][0]

    axs[0, 1].set_title('beat level')
    axs[0, 1].scatter(levels[key_non_syncopated_2]['tEnd'], levels[key_non_syncopated_2]['accentuation'], c='b', marker='.')
    axs[0, 1].scatter(levels[key_syncopated_2]['tEnd'], levels[key_syncopated_2]['accentuation'], c='b', marker='.')
    # ---------- legend non-syncopated------------
    legend_elements = [Line2D([0], [0], marker='.', color='w', label='non-syncopated',
                              markerfacecolor='b', markersize=15)]
    axs[0, 1].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------

    axs[1, 1].scatter(levels_4[key_syncopated_4]['tEnd'], levels_4[key_syncopated_4]['accentuation'], c='b', marker='.')
    # ---------- legend non-syncopated------------
    legend_elements = [Line2D([0], [0], marker='.', color='w', label='syncopated',
                              markerfacecolor='b', markersize=15)]
    axs[1, 1].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------
    list(merged_levels_4.values())[0].plot(x='tEnd', y='accentuation', ax=axs[2, 1], c='b')
    # ---------- legend merged------------
    legend_elements = [Line2D([0], [0], color='b', label='merged')]
    axs[2, 1].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------
    list(pre_processed_annotation_4.values())[0].plot(x='tEnd', y='accentuation', ax=axs[3, 1], c='b')
    # ---------- legend added zeros ------------
    legend_elements = [Line2D([0], [0], color='b', label='added zeros')]
    axs[3, 1].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------

    resampled_4[list(resampled_4.keys())[0]].plot(x='tEnd', y='accentuation', ax=axs[4, 1], c='b')
    # ---------- legend resampled ------------
    legend_elements = [Line2D([0], [0], color='b', label='resampled')]
    axs[4, 1].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------

    ############# EIGHTH NOTE LEVEL ###################################
    filter_eighth_note = '[2-8]'
    levels_8 = read_annotation('annotations_all_levels', '/' + song_title + '*' + filter_eighth_note + '_.csv', columns)
    merged_levels_8 = merge_levels(levels_8.copy())
    pre_processed_annotation_8 = add_zeros_between_annotations(merged_levels_8)
    resampled_8 = adjust_sampling_rate(pre_processed_annotation_8, 250)
    search_key = 'syncopation_8'
    key_syncopated_8 = [key for key in levels_8.keys() if search_key in key][0]

    axs[0, 2].set_title('eighth note level')
    axs[0, 2].scatter(levels[key_non_syncopated_2]['tEnd'], levels[key_non_syncopated_2]['accentuation'], c='b',
                      marker='.')
    axs[0, 2].scatter(levels[key_syncopated_2]['tEnd'], levels[key_syncopated_2]['accentuation'], c='b', marker='.')
    axs[0, 2].scatter(levels_4[key_syncopated_4]['tEnd'], levels_4[key_syncopated_4]['accentuation'], c='b', marker='.')
    # ---------- legend non-syncopated------------
    legend_elements = [Line2D([0], [0], marker='.', color='w', label='non-syncopated',
                              markerfacecolor='b', markersize=15)]
    axs[0, 2].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------

    axs[1, 2].scatter(levels_8[key_syncopated_8]['tEnd'], levels_8[key_syncopated_8]['accentuation'], c='b', marker='.')
    # ---------- legend non-syncopated------------
    legend_elements = [Line2D([0], [0], marker='.', color='w', label='syncopated',
                              markerfacecolor='b', markersize=15)]
    axs[1, 2].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------
    list(merged_levels_8.values())[0].plot(x='tEnd', y='accentuation', ax=axs[2, 2], c='b')
    # ---------- legend merged------------
    legend_elements = [Line2D([0], [0], color='b', label='merged')]
    axs[2, 2].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------
    list(pre_processed_annotation_8.values())[0].plot(x='tEnd', y='accentuation', ax=axs[3, 2], c='b')
    # ---------- legend added zeros ------------
    legend_elements = [Line2D([0], [0], color='b', label='added zeros')]
    axs[3, 2].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------

    resampled_4[list(resampled_8.keys())[0]].plot(x='tEnd', y='accentuation', ax=axs[4, 2], c='b')
    # ---------- legend resampled ------------
    legend_elements = [Line2D([0], [0], color='b', label='resampled')]
    axs[4, 2].legend(handles=legend_elements, loc='right')
    # ---------------------------------------------
    for x1 in range(5):
        for x2 in range(3):
            axs[x1, x2].set_ylim(0, 1)
            axs[x1, x2].set_xlim(0, 25)
    pyplot.show()

song_title = 'Chandelier'
visualize_annotations(song_title)
#x = 1
