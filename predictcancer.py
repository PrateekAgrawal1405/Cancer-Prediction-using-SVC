# DEPENDENCIES

import numpy as np
import pandas as pd
import itertools
import pylab as py
import scipy.optimize as opt
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report, jaccard_similarity_score


"%matplotlib inline"

df = pd.read_csv('cell_samples.csv')
# print(df.head())

# VISUALIZING CLASS BASED ON CLUMP THICKNESS AND UNIFORMITY OF CELL SIZE

""" ax = df[df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
df[df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()
 """

# print(df.dtypes)

""" DATA PREPROCESSING  """
# BareNuc is of type object but we can only evaluate on int values hence drop those rows

df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype('int')
# print(df.dtypes)

X = np.asanyarray(df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh',
                      'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
y = np.asanyarray(df['Class'])

# Splitting data into test and train

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4)
# print("Shape of train data : ",X_train.shape,y_train.shape)
# print("Shape of test data : ",X_test.shape,y_test.shape)

# Classifier

clf = svm.SVC(kernel='rbf')  # kernel is standard Radial Basis Function
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

""" print("The Jaccard Index of the data is :",
      jaccard_similarity_score(y_test, y_pred))

print("The f1 score is :", f1_score(y_test, y_pred, average='weighted')) """

# CONFUSION MATRIX

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, y_pred))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')