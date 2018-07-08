from collections import defaultdict
import pandas as pd
import numpy as np


class DatasetTransformer:

    def __init__(self):
        pass

    ''' redefineTrainSet is used to aggregate the new sample and its label
            from Uncertain_sample or Random_sample to the existing
            training features and training label respectively '''

    def redefineTrainSet(self, train_X_tfidf, train_Y, uncertain_sample, label):
        train_X_tfidf = np.vstack((train_X_tfidf, uncertain_sample))

        trainY_len = len(train_Y)
        lastRowIndex = train_Y.index[trainY_len - 1]
        firstRowIndex = train_Y.index[0]
        train_Y = train_Y.append(pd.Series(np.array([label])), ignore_index=False)
        train_Y.index = range(firstRowIndex, lastRowIndex + 2)

        return train_X_tfidf, train_Y

    def slice_array(self, array, index):
        array = np.delete(array, (index), axis=0)

        return array

    def remove_row_from_array(self, array, index):
        array = np.delete(array, (index), axis=0)

        return array

    ''' returns  the name of the label that is the most insignificant (less 1) '''

    def getMostInsignificantLabel(self, labelDF):
        shapeOfDF = labelDF.shape
        numberOfColumns = shapeOfDF[1]
        colNameSumList = defaultdict(list)

        columnNames = list(labelDF.columns.values)

        for i in range(0, numberOfColumns):
            colNameSumTuple = pd.Series(labelDF[columnNames[i]].sum(), index=[columnNames[i]])
            colNameSumList[colNameSumTuple[0]].append(columnNames[i])

        mostInsignificantLabel = min(colNameSumList.items())

        return mostInsignificantLabel[1][0]

    def getMostSignificantLabel(self, labelDF):
        shapeOfDF = labelDF.shape
        numberOfColumns = shapeOfDF[1]
        colNameSumList = defaultdict(list)

        columnNames = list(labelDF.columns.values)

        for i in range(0, numberOfColumns):
            colNameSumTuple = pd.Series(labelDF[columnNames[i]].sum(), index=[columnNames[i]])
            colNameSumList[colNameSumTuple[0]].append(columnNames[i])

        mostSignificantLabel = max(colNameSumList.items())

        return mostSignificantLabel[1][0]
