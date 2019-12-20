from music_features import get_accentuation
from motion_features import get_movement_features
from scipy import signal
from matplotlib import pyplot


def main():
    sample_rate = 250
    directory_motion = "motion_phone"
    directory_music = "annotations"
    motion = get_movement_features(directory_motion, sample_rate)
    music = get_accentuation(directory_music, sample_rate)

    music_sample1 = music["annotations/AladdinSane_.csv"]
    motion_sample_true = motion["motion_phone/AladdinSane_#20_acc.csv"]
    motion_sample_false = motion["motion_phone/Bad_#20_acc.csv"]
    cross_correlation_true = signal.correlate(music_sample1, motion_sample_true[:, 0], mode='same')
    fig, (ax_music_sample1, ax_motion_sample_true, ax_cross_correlation_true) = pyplot.subplots(3, 1, sharex=True)
    ax_music_sample1.plot(music_sample1)
    ax_motion_sample_true.plot(motion_sample_true[:, 0])
    ax_cross_correlation_true.plot(cross_correlation_true)

    pyplot.show()


    cross_correlation_false = signal.correlate(music_sample1, motion_sample_false[:, 0], mode='same')
    fig, (ax_music_sample1, ax_motion_sample_false, ax_cross_correlation_false) = pyplot.subplots(3, 1, sharex=True)
    ax_music_sample1.plot(music_sample1)
    ax_motion_sample_false.plot(motion_sample_false[:, 0])
    ax_cross_correlation_false.plot(cross_correlation_false)

    pyplot.show()

    cross_correlation_false = signal.correlate(music_sample1, motion_sample_false[:, 0], mode='same') / 128

if __name__ == "__main__":
    main()
