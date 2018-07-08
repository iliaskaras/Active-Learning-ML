import numpy as np
from dataset_operation import DatasetTransformer


class ActiveLearning:

    def __init__(self):
        self.datasetTransformer = DatasetTransformer()

        pass
    ''' prepares and returns our unlabeledPoolSet and their true "Unknown" labels '''

    def prepareUnlabeledDF(self, unlabeledPoolSet, mostInsignificantLabel, countVectorizer, tfidf_transformer):
        ''' poolSet_Y will be needed at the fucntion :get_the_most_uncertain_sample
            and will act as the person that will label our uncertain sample. (oracle) '''

        poolSet_Y = unlabeledPoolSet[mostInsignificantLabel]
        unlabeledPoolSet = countVectorizer.transform(unlabeledPoolSet['sentences'])

        unlabeledPoolSet = tfidf_transformer.transform(unlabeledPoolSet)

        unlabeledPoolSet = unlabeledPoolSet.toarray()
        poolSet_Y = poolSet_Y.values

        return unlabeledPoolSet, poolSet_Y

    ''' Will return the sample and its label that our classifier is more uncertain based on the probability it
            has to label it in either one of the classes. 
        For example if the probabilities for the two classes are close to .50 then this means that is most
            uncertain for this example on how to label it. '''

    def get_the_most_uncertain_sample(self, clf, unlabeledPoolSet, poolSet_Y):
        probabilities = clf.predict_proba(unlabeledPoolSet)
        scores = -np.max(probabilities, axis=1)
        ask_id = np.argmax(scores)

        changedUnlabeledPoolSet = self.datasetTransformer.remove_row_from_array(unlabeledPoolSet, ask_id)
        changedPoolSet_Y = self.datasetTransformer.slice_array(poolSet_Y, ask_id)

        return unlabeledPoolSet[ask_id], poolSet_Y[ask_id], changedUnlabeledPoolSet, changedPoolSet_Y

    ''' Splits the dataframe in half, used to get the unlabeledPoolSet (first 50% of the examples)
            and the testSet with which we will train our model (rest 50 % of our examples) '''

    def splitDataframeInHalf(self, df):
        half = int(len(df) / 2)
        unlabeledPoolSet, testSet = df.iloc[:half], df.iloc[half:]
        return unlabeledPoolSet, testSet

    ''' Get a random uncertain sample and its label '''

    def get_random_sample(self, clf, unlabeledPoolSet, poolSet_Y, i):
        import random
        unlabeledPoolSetSize = len(unlabeledPoolSet)
        random.seed(i)
        randomID = random.randint(0, unlabeledPoolSetSize)

        changedUnlabeledPoolSet = self.datasetTransformer.remove_row_from_array(unlabeledPoolSet, randomID)
        changedPoolSet_Y = self.datasetTransformer.slice_array(poolSet_Y, randomID)

        return unlabeledPoolSet[randomID], poolSet_Y[randomID], changedUnlabeledPoolSet, changedPoolSet_Y