import pandas as pd
from matplotlib import pyplot
from scipy import signal
from glob import glob


def get_accentuation(directory, rate):
    annotation = read_annotation(directory)
    adjusted_annotation = adjust_sampling_rate(annotation, rate)
    return adjusted_annotation


def read_annotation(directory):
    annotations = {}
    file_names = glob(directory + '/*_.csv')
    for filename in file_names:
        annotations[filename] = pd.DataFrame(pd.read_csv(filename, sep=',\s*', header=0))
    return annotations


def adjust_sampling_rate(data, rate):
    adjusted_data = {}
    for filename, df in data.items():
        timestamps = pd.to_datetime(df['tStart'], unit='s')
        series = pd.Series(df['accentuation'].values, index=timestamps)
        newFreq = series.resample('10ms').asfreq()
        newSeries = series.reindex(newFreq.index, method='nearest', tolerance=pd.Timedelta('10ms')).interpolate(method='polynomial', order=2)
        adjusted_data[filename] = signal.resample(newSeries, num=rate)
        #newSeries.plot()
        #series.plot()
        #pyplot.plot(final)
        pyplot.show()
    return adjusted_data
