import pandas as pd
import numpy as np
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()

df1 = pd.read_csv('./student-por.csv', sep=';')
df1

# bool type
df1['G3'].plot(kind='hist', bins=20)
df1.columns.tolist()
booldict = {'yes': 1, 'no': 0}
originalname = [
'school',
 'sex',
 'age',
 'address',
 'famsize',
 'Pstatus',
 'Medu',
 'Fedu',
 'Mjob',
 'Fjob',
 'reason',
 'guardian',
 'traveltime',
 'studytime',
 'failures',
 'schoolsup',
 'famsup',
 'paid',
 'activities',
 'nursery',
 'higher',
 'internet',
 'romantic',
 'famrel',
 'freetime',
 'goout',
 'Dalc',
 'Walc',
 'health',
 'absences',
 'G1',
 'G2',
 'G3'
]
dtype = [
 'category',
 'category',
 np.uint32,
 'category',
 'category',
 
 'category',
 'category',
 'category',
 'category',
 'category',
 
 'category',
 'category',
 'category',
 'category',
 np.uint8,
 
 'category',
 'category',
 'category',
 'category',
 'category',
 
 'category',
 'category',
 'category',
 'category',
 'category',
 
 'category',
 'category',
 'category',
 'category',
 np.uint16,
 
 np.uint8,
 np.uint8,
 np.uint8,
]
bins = [-np.inf, 10, np.inf]

df1 = df1.astype(dict(zip(originalname, dtype)))
df1[df1.columns[15:23]] = df1[df1.columns[15:23]].apply(
 lambda x : x.map(booldict)
)

df1[df1.columns[15:23]] = df1[df1.columns[15:23]].astype('bool') 
# df1['failures'] = df1['failures'].astype('bool')
df1['G3_Cat'] = pd.cut(df1['G3'], bins=bins, labels=list(range(len(bins)-1)))
df1.info()
# if dtype of df col is category change it to code
df1_backup = df1.copy(deep=True)
for col in df1.columns:
    if df1[col].dtype == 'category':
        df1[col] = df1[col].cat.codes.astype('category')
df1.info()
df1['G3_Cat'].value_counts().plot(kind='bar')
df1.corr(numeric_only=False)['failures'].abs().sort_values(ascending=False)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from itertools import combinations
target = 'failures'
searchList = df1.corr(numeric_only=False)['G3'].abs().sort_values(ascending=False).index[5:20].tolist()
try:
    searchList.remove(target)
    searchList.remove('age')
except:
    pass
resultDict = {}
resultList = []
rowList = []
filterList = ['school', 'reason', 'higher',]
y = df1[target]

# for filter in filterList:
#     searchList.remove(filter)

for x in range(1,len(searchList)):
    for row in combinations(searchList, x):
        rowList.append(list(row))
len(rowList)
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import NuSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
def traindata(row, resultList:list, Final=False, model = LogisticRegression):
    X = df1[row]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    knn = clone(model(max_iter=10**10))
    knn.fit(X_train, y_train)
    result = knn.score(X_test, y_test)
    resultList.append([row, str(model).split('.')[-1], result])
    if Final:
        print(f'{row} : {result}')
        print(classification_report(y_test, knn.predict(X_test)))
        print(confusion_matrix(y_test, knn.predict(X_test)))
        # print(f'AUC: {roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1]):.3f}')
        # print(f'ROC: {roc_curve(y_test, knn.predict_proba(X_test)[:, 1])}')
        

model = [
    # SVC,
    # NuSVC,
    # RidgeClassifier,
    # LogisticRegression,
    KNeighborsClassifier,
]
    


# ]
resultList.clear()
for modelname in model:
    for row in rowList:
        traindata(row, resultList, False, modelname)

print('complte')