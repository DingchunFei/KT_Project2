from matplotlib import pyplot
import scipy as sp
import numpy as np
# from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from numpy import *

# ========SVM========#
def SvmClass(x_train, y_train):
    from sklearn.svm import SVC
    # 调分类器
    clf = SVC(kernel='linear', probability=True)  # default with 'rbf'
    clf.fit(x_train, y_train)  # 训练，对于监督模型来说是 fit(X, y)，对于非监督模型是 fit(X)
    return clf


# =====NB=========#
def NbClass(x_train, y_train):
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=0.01).fit(x_train, y_train)
    return clf


# ========Logistic Regression========#
def LogisticClass(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2')
    clf.fit(x_train, y_train)
    return clf


# ========KNN========#
def KnnClass(x_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    return clf


# ========Decision Tree ========#
def DccisionClass(x_train, y_train):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    return clf


# ========Random Forest Classifier ========#
def random_forest_class(x_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=8)  # 参数n_estimators设置弱分类器的数量
    clf.fit(x_train, y_train)
    return clf


# ========准确率召回率 ========#
def Precision(clf):
    doc_class_predicted = clf.predict(x_test)
    print(np.mean(doc_class_predicted == y_test))  # 预测结果和真实标签
    # 准确率与召回率
    #precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
    answer = clf.predict_proba(x_test)[:, 1]
    report = answer > 0.5
    print("分类器结果：",doc_class_predicted)
    print("标签结果  ：",y_test)
    print(classification_report(y_test, doc_class_predicted, target_names=['negative','neutral', 'positive']))
    print("--------------------")
    from sklearn.metrics import accuracy_score
    print('准确率: %.2f' % accuracy_score(y_test, doc_class_predicted))


if __name__ == '__main__':
    train_data = []
    train_labels = []
    with open("dataset/train-labels.txt", "r" ,encoding='UTF-8')as file:
        for line in file:
            line = line[19:-1]
            if(line=="negative"):                       # negative => -1
                train_labels.append(-1)
            elif(line=="neutral"):                      # neutral  => 0
                train_labels.append(0)
            elif(line=="positive"):                     # postive  => 1
                train_labels.append(1)

    with open("dataset/train-tweets.txt", "r", encoding='UTF-8')as file:
        for line in file:
            words=[]
            line = line[19:-1]
            words = line.split(" ")
            flag=1
            for word in words:
                if(word.isalpha()):
                    if(flag==1):
                        newline=word
                        flag=0
                    else:
                        newline=newline+" "+word
            train_data.append(newline)

    eval_data = []
    eval_labels = []
    with open("dataset/eval-labels.txt", "r", encoding='UTF-8')as file:
        for line in file:
            line = line[19:-1]
            if (line == "negative"):  # negative => -1
                eval_labels.append(-1)
            elif (line == "neutral"):  # neutral  => 0
                eval_labels.append(0)
            elif (line == "positive"):  # postive  => 1
                eval_labels.append(1)

    with open("dataset/eval-tweets.txt", "r", encoding='UTF-8')as file:
        for line in file:
            words=[]
            line = line[19:-1]
            words = line.split(" ")
            flag=1
            for word in words:
                if(word.isalpha()):
                    if(flag==1):
                        newline=word
                        flag=0
                    else:
                        newline=newline+" "+word
            eval_data.append(newline)


    x_train = np.array(train_data)
    y_train = [int(i) for i in train_labels]

    x_test = np.array(eval_data)
    y_test = [int(i) for i in eval_labels]

    # print (x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)

    #movie_target = labels
    # 转换成空间向量
    count_vec = TfidfVectorizer(binary=False)
    # 加载数据集，切分数据集80%训练，20%测试
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    x_train = count_vec.fit_transform(x_train)
    x_test = count_vec.transform(x_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print('**************支持向量机************  ')
    Precision(SvmClass(x_train, y_train))
    print('**************朴素贝叶斯************  ')
    Precision(NbClass(x_train, y_train))
    print('**************最近邻KNN************  ')
    Precision(KnnClass(x_train, y_train))
    print('**************逻辑回归************  ')
    Precision(LogisticClass(x_train, y_train))
    print('**************决策树************  ')
    Precision(DccisionClass(x_train, y_train))
    print('**************逻辑森林************  ')
    Precision(random_forest_class(x_train, y_train))