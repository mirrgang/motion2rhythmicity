import os
from praat_python import praatTextGrid, generalUtility
from glob import glob


def get_sample_name(file):
    name = file[5:-20]
    return name


directory = '../annotations_all_levels/'

#file_names = glob(directory+'/*_accentuation.TextGrid')
#for file in file_names:

# look for all TextGrids in the directory
for fName in os.listdir(directory):
    if fName.split('.')[-1] == 'TextGrid':
       # instantiate a new TextGrid object
       textGrid = praatTextGrid.PraatTextGrid(0, 0)
       # initialize the TextGrid object from the TextGrid file
       # arrTiers is an array of objects (either PraatIntervalTier or
       # PraatPointTier)
       arrTiers = textGrid.readFromFile(directory + fName)

       for current_tier in arrTiers:
           # open the output file

           csv_file_name = get_sample_name(fName) + '_' + current_tier.getName() + '_.csv'
           csvFile = open(directory + csv_file_name, 'w+')
           # writer csv header
           csvFile.write("tStart, tEnd, accentuation\n")
           fileNameOnly = generalUtility.getFileNameOnly(fName)

           # now loop over all the defined intervals in the tier.
           for i in range(current_tier.getSize()):
               # only consider those intervals that are actually labelled.
               if current_tier.getLabel(i) != '':
                interval = current_tier.get(i)
                print("\t", interval)
                # write to CSV file
                if len(interval) < 3:
                   csvFile.write("%f, %f, %s\n" % (0, interval[0], interval[1]))#if first entry = ""
                else:
                   csvFile.write("%f, %f, %s\n" % (interval[0], interval[1], interval[2]))
           csvFile.close()

