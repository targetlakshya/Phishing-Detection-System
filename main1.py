import pandas as pd
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split#,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import auc, roc_curve
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings('always')
from warnings import simplefilter
from sklearn.exceptions import DataConversionWarning
simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import warnings
warnings.simplefilter("ignore")
warnings.simplefilter("ignore", UserWarning) 



df=pd.read_csv("phishing.csv")
print(df.head())
print(df.info())
df.isnull().sum()

X= df.drop(columns='class')
Y=df['class']
Y=pd.DataFrame(Y)
X.describe()

pd.value_counts(Y['class']).plot.bar()
plt.show()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X=scaler.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)
pca.get_covariance()
explained_variance=pca.explained_variance_ratio_
explained_variance

with plt.style.context('dark_background'):
    plt.figure(figsize=(10, 8))

    plt.bar(range(31), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    
pca=PCA(n_components=16)
X_new=pca.fit_transform(X)
pca.get_covariance()
explained_variance=pca.explained_variance_ratio_
explained_variance

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(16), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()    
    
train_X,test_X,train_Y,test_Y = train_test_split(X_new,Y,test_size=0.24,random_state=2)



'''ANN'''
print("Artificial Neural Network")
print()
classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(train_X, train_Y, batch_size = 10, epochs = 20,verbose = 1)



ann_pred=classifier.predict(test_X)
ann_prob = ann_pred[:,0]
ann_pred=ann_pred>0.5
ann_cr=classification_report(ann_pred,test_Y)
print(ann_cr)
print()
ann_cm=confusion_matrix(ann_pred,test_Y)
print(ann_cm)
print()
ann_tn = ann_cm[0][0]
ann_fp = ann_cm[0][1]
ann_fn = ann_cm[1][0]
ann_tp = ann_cm[1][1]
Total_TP_FP=ann_cm[0][0]+ann_cm[0][1]
Total_FN_TN=ann_cm[1][0]+ann_cm[1][1]
specificity = ann_tn / (ann_tn+ann_fp)
ann_specificity=format(specificity,'.3f')
print('ANN_specificity:',ann_specificity)
print()
plt.figure()
sns.heatmap(ann_cm,annot=True,fmt='.2f')
plt.show()
#Print Area Under Curve
false_positive_rate, recall, thresholds = roc_curve(test_Y, ann_prob)
roc_auc = auc(false_positive_rate, recall)

plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b')
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()