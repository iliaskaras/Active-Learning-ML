import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from collections import Counter
from imblearn.under_sampling import NearMiss
from active_learning import ActiveLearning
from dao import DAO
from plot_creator import PlotCreator


def main():
    _activeLearning = ActiveLearning()
    _dao = DAO()
    _plotCreator = PlotCreator()

    _dao.data_to_file("test-data")
    test_label_df, test_data_df = _dao.read_csv("test-label", "new_test-data.csv")

    ''' mostInsignificantLabel : the name of the label that is the most insignificant '''
    mostInsignificantLabel = _activeLearning.datasetTransformer.getMostInsignificantLabel(test_label_df)
    ''' test_label_df : the test_label dataframe with only the most insignificant label as a single column '''
    test_label_df = test_label_df[mostInsignificantLabel]

    ''' testDF consists of sentences in a form of bag of words, and their labels '''
    testDF = test_data_df.join(test_label_df)

    ''' unlabeledPoolSet contains the first half of our test data without their labels,
        testSet contains the rest half of our test data with their corresponding labels. '''
    unlabeledPoolSet, testSet = _activeLearning.splitDataframeInHalf(testDF)

    ''' create train_X and test_Y, 80% for train_X and 20% for test_Y.
        This two sets will be used to train our model, and in each iterations we will add 1 labeled example 
            to this trained model, by adding that example to train_X and retrain the model. '''
    trainSetSize = int(len(testSet) * 0.8)
    train_X, test_X = testSet.iloc[:trainSetSize], testSet.iloc[trainSetSize:]


    ''' get the most Insignificant Label for train and test '''
    train_Y = train_X[mostInsignificantLabel]
    test_Y = test_X[mostInsignificantLabel]

    ''' preprocess of our features, run CountVectorizer and TF-IDF '''
    countVectorizer = CountVectorizer(min_df=0., max_df=1.0)
    train_X_vectorized = countVectorizer.fit_transform(train_X['sentences'])
    test_X_vectorized = countVectorizer.transform(test_X['sentences'])
    tfidf_transformer = TfidfTransformer()
    train_X_tfidf = tfidf_transformer.fit_transform(train_X_vectorized)
    test_X_tfidf = tfidf_transformer.transform(test_X_vectorized)

    train_X_tfidf = train_X_tfidf.toarray()
    test_X_tfidf = test_X_tfidf.toarray()

    print('Original Imbalance dataset shape trainY{}'.format(Counter(train_Y)))
    print('Original Imbalance dataset shape testY{}'.format(Counter(test_Y)))

    nm = NearMiss(random_state=42)
    train_X_tfidf, train_Y = nm.fit_sample(train_X_tfidf, train_Y)
    train_Y = pd.Series(train_Y)
    print('Balanced dataset shape trainY{}'.format(Counter(train_Y)))
    print('Balanced dataset shape testY{}'.format(Counter(test_Y)))

    ''' lists that we append the accuracy results for plot creation '''
    un_sample_accuracy_list = []
    rnd_sample_accuracy_list = []

    ''' model initial creation to evaluate the accuracy with 0 samples learned '''
    clf = LogisticRegression().fit(train_X_tfidf, train_Y)
    predictions = clf.predict(test_X_tfidf)
    print("With Zero Samples accuracy: " + str(
            metrics.accuracy_score(test_Y, predictions)))
    un_sample_accuracy_list.append(metrics.accuracy_score(test_Y, predictions))
    rnd_sample_accuracy_list.append(metrics.accuracy_score(test_Y, predictions))

    ''' unlabeledPoolSet : Contains only the features of our Unlabeled Pool Set, from this
            set we will choose our samples to label with uncertain or random sampling.
        poolSet_Y : Contains the labels of our unlabeledPoolSet, from this set we will pull
            the true labels of the most uncertain example from unlabeledPoolSet, its our
            program's oracle '''
    unlabeledPoolSet, poolSet_Y = _activeLearning.prepareUnlabeledDF(unlabeledPoolSet, mostInsignificantLabel,
                                                                     countVectorizer, tfidf_transformer)

    ''' every variable with _random extension is used for Random Sampling method '''
    unlabeledPoolSet_random, poolSet_Y_random = unlabeledPoolSet, poolSet_Y
    train_X_tfidf_random, train_Y_random = train_X_tfidf, train_Y
    uncertainClf = clf
    randomClf = clf



    ''' The main loop that will run 10 times and each time will do the following :
            1) Get the most Uncertain Sample based on how uncertain is our classifier on classifying it,
               ask our oracle for this specific uncertain example's label,
               and finally retrain our model adding that example and print the accuracy
            2) Get a Random Sample in range 0 to length of our unlabeledPoolSet_random,
               ask our oracle for this specific random example's label,
               and finally retrain our model adding that example and print the accuracy '''
    for i in range(0, 10):

        ''' Active Learning - Uncertain_sample '''
        uncertain_sample, label, unlabeledPoolSet, poolSet_Y = \
            _activeLearning.get_the_most_uncertain_sample(clf=uncertainClf, unlabeledPoolSet=unlabeledPoolSet, poolSet_Y=poolSet_Y)
        train_X_tfidf, train_Y = _activeLearning.datasetTransformer\
            .redefineTrainSet(train_X_tfidf, train_Y, uncertain_sample, label)
        # print("uncertain_sample : " + str(uncertain_sample) + ", its label:" + str(label))
        ''' Active Learning - Uncertain_sample '''

        ''' Random_Sampling '''
        uncertain_sample, label, unlabeledPoolSet_random, poolSet_Y_random = \
            _activeLearning.get_random_sample(clf=randomClf, unlabeledPoolSet=unlabeledPoolSet_random, poolSet_Y=poolSet_Y_random, i=i)
        train_X_tfidf_random, train_Y_random = _activeLearning.datasetTransformer\
            .redefineTrainSet(train_X_tfidf_random, train_Y_random, uncertain_sample, label)
        # print("random_sample : "+str(uncertain_sample)+", its label:"+str(label))
        ''' Random_Sampling '''

        ''' Re create our LogisticRegression classifiers and fit the models with the +1 labeled example
                and get our new Accuracy '''
        uncertainClf = LogisticRegression().fit(train_X_tfidf, train_Y)
        predictions = uncertainClf.predict(test_X_tfidf)
        un_sample_accuracy_list .append(metrics.accuracy_score(test_Y, predictions))
        print("With Uncertain_Sample method : i= "+str(i)+", accuracy: "+str(metrics.accuracy_score(test_Y, predictions)))

        randomClf = LogisticRegression().fit(train_X_tfidf_random, train_Y_random)
        predictions = randomClf.predict(test_X_tfidf)
        rnd_sample_accuracy_list.append(metrics.accuracy_score(test_Y, predictions))
        print("With Random_Sample method : i= " + str(i) + ", accuracy: " + str(
            metrics.accuracy_score(test_Y, predictions)))

    _plotCreator.createPlot(un_sample_accuracy_list,rnd_sample_accuracy_list)


if __name__ == "__main__":
    main()

