import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

mushroom_data = pandas.read_csv("agaricus-lepiota.data", header=None)

#print(data.head())
# bo cot chi co 1 du lieu
#print(mushroom_data[16].unique())
mushroom_data.drop(16, inplace=True, axis=1)
#mushroom_data.drop(11, inplace=True, axis=1)
col_with_missing = [col for col in mushroom_data.columns if mushroom_data[col].isnull().any()]

#columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']


#chuan hoa du lieu
mushroom_data[0] = [1 if i == "p" else 0 for i in mushroom_data[0]]

#đổi dữ liệu trong mushrooms từ chữ thành số
for column in mushroom_data.drop([0], axis=1).columns:# cho columns chạy hết các cột trong data trừ cột target
    value = 0
    step = 1/(len(mushroom_data[column].unique())-1)#lấy số lượng các ký tự xuất hiện trong cột
    for i in mushroom_data[column].unique():
        mushroom_data[column] = [value if letter == i else letter for letter in mushroom_data[column]]#đổi lại thành số
        value += step
#print(mushroom_data.head())
data = mushroom_data.drop(0, axis=1)
target = mushroom_data[0]

#chia tap du lieu huan luyen
X_train, X_test, Y_train, Y_test = train_test_split(data,target, test_size=0.3, random_state=42)

# cay quyet dinh
from sklearn import tree
dt = tree.DecisionTreeClassifier(criterion='gini')
dt.fit(X_train, Y_train)
# du doan nhan
Y_pred_dt = dt.predict(X_test)
#print(Y_pred)

print("do tin cay mo hinh cay quyet dinh:",accuracy_score(Y_test, Y_pred_dt))
#danh gia mo hinh cay quyet dinh
nFold = 10
score_dt = cross_val_score(dt, data, target, cv=nFold)
print("do chinh xac mo hinh cay quyet dinh voi K = %d la %.3f"%(nFold, numpy.mean(score_dt)))

#K lang gieng
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
#du doan nhan
Y_pred_knn = knn.predict(X_test)
print("do tin cay mo hinh K lang gieng:", accuracy_score(Y_test, Y_pred_knn))

score_knn = cross_val_score(knn, data, target, cv=nFold)
print("do chinh xac mo hinh K lang gieng voi K = %d la %.3f"%(nFold, numpy.mean(score_knn)))

#naive bayer

from sklearn import svm
nb = svm.SVC()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)
print("do tin cay mo hinh SVM:", accuracy_score(Y_test, Y_pred_nb))

score_nb = cross_val_score(nb, data, target, cv=nFold)
print("do chinh xac mo hinh SVM voi K = %d la %.3f"%(nFold, numpy.mean(score_nb)))

# ve bieu do ma tran hon loan
cm_dt = pandas.DataFrame(confusion_matrix(Y_test, Y_pred_dt), index=["khong doc", "co doc"], columns=["khong doc", "co doc"])
cm_knn = pandas.DataFrame(confusion_matrix(Y_test, Y_pred_knn), index=["khong doc", "co doc"], columns=["khong doc", "co doc"])
cm_nb = pandas.DataFrame(confusion_matrix(Y_test, Y_pred_nb), index=["khong doc", "co doc"], columns=["khong doc", "co doc"])

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_dt,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("du doan")
plt.ylabel("thuc te")
plt.title("cay quyet dinh")
plt.show()
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_knn,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("du doan")
plt.ylabel("thuc te")
plt.title("K lang gieng")
plt.show()
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_nb,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("du doan")
plt.ylabel("thuc te")
plt.title("SVM")
plt.show()