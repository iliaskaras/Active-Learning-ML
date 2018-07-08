import re
import pandas as pd
import csv

class PlotCreator:

    def __init__(self):
        pass


    def createPlot(self, list1, list2):

        for item1 in list1 :
            print("un:"+str(item1))

        for item1 in list2:
            print("rnd:"+str(item1))

        import numpy as np
        import matplotlib.pyplot as plt

        list1 = np.array(list1)
        list2 = np.array(list2)

        plt.ylabel('accuracy')
        plt.xlabel('number of samples added')
        plt.title('Accuracy Results')
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        my_xticks = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        plt.xticks(x, my_xticks)
        plt.xlim([0, 10])
        plt.ylim([0, 1])
        plt.plot(list1, color='g', linestyle='-.',)
        plt.plot(list2, color='red', linestyle='--',)

        plt.rcParams["legend.fontsize"] = 11
        plt.rcParams["legend.framealpha"] = 0

        line1, = plt.plot([1, 2, 3], color='g',  linestyle='-.', label='Uncertainty Sampling')
        line2, = plt.plot([3, 2, 1], color='red',  linestyle='--', label='Random')
        plt.legend(handles=[line1, line2])


        plt.show()

        pass