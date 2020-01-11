import os
import urllib
import tarfile
import pandas as pd
import numpy as np

####  link to data file @
#### https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz

# def fetch_housing_data(housing_url,housing_path):
#     if not os.path.isdir(housing_path):
#         os.makedirs(housing_path)
#         tgz_path = os.path.join(housing_path, "housing.tgz")
#         urllib.request.urlretrieve(housing_url, tgz_path)
#         housing_tgz = tarfile.open(tgz_path)
#         housing_tgz.extractall(path=housing_path)
#         housing_tgz.close()



def load_housing_data(prawcsv):
    p = os.getcwd()
    df = pd.read_csv(prawcsv)
    # print(df.info())
    # print(df.head())
    # print(df.describe())
    return df


df = load_housing_data('housing.csv')

##### Histogram ######
def create_hist():
    import matplotlib.pyplot as plt
    df.hist(bins=50, figsize=(10,5))
    plt.show()
#create_hist()


####### Create Test Set of Data ########
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.premutation(len(data))
    print(shuffled_indices)

split_train_test()