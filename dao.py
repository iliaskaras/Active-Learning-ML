import re
import pandas as pd
import csv

class DAO:

    def __init__(self):
        pass

    def label_to_csv(self, filename, header):
        dat_content = [i.strip().split() for i in open(filename).readlines()]

        df = pd.DataFrame(dat_content)

        df.to_csv(filename.split(".")[0], header=header)

    def data_to_file(self, filename):
        dat_content = [re.sub(r'<\d+>', "", i.strip()).split() for i in open(filename).readlines()]

        with open("new_" + filename.split(".")[0], 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerows(dat_content)

    def data_to_file(self, filename):
        dat_content = [re.sub(r'<\d+>', "", i.strip()).split() for i in open(filename).readlines()]

        with open("new_" + filename.split(".")[0] + ".csv", 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerows(dat_content)

    def load_data(self, filename):
        try:
            lines = []
            with open("new_" + filename) as f:
                lines = [line.strip() for line in f]

        except Exception:
            print("problem reading " + filename)

        return lines

    def read_csv(self, label_filename, feature_filename):
        test_label_df = pd.read_csv(label_filename, header=None, sep=' ',
                                    names=['programming', 'style', 'reference', 'java', 'web', 'internet', 'culture',
                                           'design', 'education', 'language',
                                           'books', 'writing', 'computer', 'english', 'politics', 'history',
                                           'philosophy', 'science', 'religion', 'grammar'])
        test_data_df = pd.read_csv(feature_filename, header=None, sep='\n',
                                   names=['sentences'])

        return test_label_df,test_data_df
