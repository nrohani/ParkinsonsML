# -*- coding: utf-8 -*-


# import all necessary libraries
import pandas
from sklearn import svm
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import GridSearchCV,LeaveOneOut,train_test_split,KFold,cross_validate,cross_val_score,StratifiedKFold
from sklearn.metrics import matthews_corrcoef,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss,accuracy_score,roc_curve,auc,f1_score,recall_score,precision_score,roc_auc_score,precision_recall_curve
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
# load the dataset (local path)
def performance_metric(y_true, y_predict):
    error = f1_score(y_true, y_predict, pos_label=1)
    return error
def fit_model(X, y):
  
    classifier = svm.SVC(probability=True)

    parameters = {'kernel':['rbf', 'linear'], 'degree':[1, 2, 3], 'C':[0.1, 1, 10]}


    f1_scorer = make_scorer(performance_metric,
                                   greater_is_better=True)

    clf = GridSearchCV(classifier,
                       param_grid=parameters,
                       scoring=f1_scorer)

    clf.fit(X, y)

    return clf

url = "parkinsonss.csv"
# feature names
features = ['Jitter (local)','Jitter (local, absolute)','Jitter (rap)','Jitter (ppq5)','Jitter (ddp)',' Shimmer (local)','Shimmer (local, dB)','Shimmer (apq3)','Shimmer (apq5)', 'Shimmer (apq11)','Shimmer (dda)',' AC','NTH','HTN',' Mp','Mean pitch','Sd'',Minimum pitch','Maximum pitch', ' NoPuls','Number of periods','Mean period','Standard deviation of period', ' Fraction of locally unvoiced frames','Number of voice breaks','Degree of voice breaks','UPDRS','status'   ]
dataset = pandas.read_csv(url, names = features)
target_col = dataset.columns[26]
# store the dataset as an array for easier processing
array = dataset.values
# X stores feature values
X = array[:,0:26]
# Y stores "answers", the flower species / class (every row, 4th column)
parkinsondata=np.loadtxt("parkinsonss.csv",dtype=float,delimiter=",")
Y=parkinsondata[:,27]
X = parkinsondata[:,0:26]

print('target',Y)
validation_size = 0.3
# randomize which part of the data is training and which part is validation
seed = 7
# split dataset into training set (80%) and validation set (20%)
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, random_state = seed)
num_folds = 5
num_instances = len(X_train)
seed = 7
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('QDA', QDA()))
#models.append(('NB', GaussianNB()))
#models.append(('LREg',RidgeClassifier(alpha=0)))
models.append(('knn',KNeighborsClassifier(n_neighbors=1)))
models.append(('tree',DecisionTreeClassifier(random_state=0)))
models.append(('svm',svm.SVC(probability=True)))
class_index=0
# evaluate each algorithm / model
results = []
names = []
kf = StratifiedKFold(n_splits=5)
kf.get_n_splits(X, Y)
StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
X=np.array(X)
Y=np.array(Y)
all_performance_lr = []
all_performance_LDA = []
all_performance_NB = []
all_performance_QDA = []
loo = LeaveOneOut()
loo.get_n_splits(X)
yprob=[]
all_prob={}
all_prob[1] = []
all_prob[2] = []
all_prob[3] = []
all_prob[4] = []
all_prob[5] = []
all_y={}
all_y[1] = []
all_y[2] = []
all_y[3] = []
all_y[4] = []
all_y[5] = []
LeaveOneOut()
Figure = plt.figure()
def plot_roc_curve(labels, probality, legend_text, auc_tag = True):
    #fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text +' (AUC=%6.3f) ' %roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text )

for name, model in models:
    
    ytests = []
    ypreds = []
    yprob=[]
    class_index=class_index+1
    model.fit(X,Y)
    pred=model.predict(X)
    ae_y_pred_prob=[]
    p=[]
    los=log_loss(Y,pred)
    if name=='svm':
         clf=fit_model(X,Y)
    else:
        clf = model

    for train_index, test_index in loo.split(X,Y):
        x, x_ = X[train_index], X[test_index]
        y, y_ = Y[train_index], Y[test_index]
        clf.fit(x,y)
        ae_y_pred_prob = clf.predict(x_)
        p=clf.predict_proba(x_)[:,1]
        ytests += [val for val in y_]
        ypreds += [val for val in ae_y_pred_prob]
        yprob += [val for val in p]
    f=f1_score(ytests,ypreds)
    pr=precision_score(ytests,ypreds)
    rec=recall_score(ytests,ypreds)
    aucs=0
    try:
         aucs=roc_auc_score(ytests,yprob)
    except ValueError:
         print('')
    fpr, tpr, thresholds = roc_curve(ytests,yprob)
    roc_auc = auc(fpr, tpr)
#        print('auac',aucs,f,pr,rec)
    precision1, recall, pr_threshods = precision_recall_curve(ytests, yprob)
    aupr_score = auc(recall, precision1)
    all_F_measure=np.zeros(len(pr_threshods))
    for k in range(0,len(pr_threshods)):
          
           if (precision1[k]+precision1[k])>0:
             all_F_measure[k]=2*precision1[k]*recall[k]/(precision1[k]+recall[k])
           else:
              all_F_measure[k]=0
    max_index=all_F_measure.argmax()
    predicted_score=np.zeros(len(ytests))
    threshold=pr_threshods[max_index]
    predicted_score[ypreds>threshold]=1
    f=f1_score(ytests,predicted_score)
    recall=recall_score(ytests, predicted_score)
    precision1=precision_score(ytests, predicted_score)
    c=confusion_matrix(ytests,ypreds)
    print(c,name)
    loss=log_loss(ytests,ypreds)
    plot_roc_curve(ytests, yprob, name)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LOOCVROC')
plt.legend(loc="lower right")
Figure.savefig('LOOCV.png') 
plt.show() 

        
        

