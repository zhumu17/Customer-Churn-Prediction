import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def loadData():
    df = pd.read_csv('./DATA/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # print(df.head())
    df_dummies_part1 = pd.get_dummies(df.loc[:, ['gender', 'Partner','Dependents']])
    df_dummies_part2 = pd.get_dummies(df.loc[:,'PhoneService' : 'PaymentMethod'])
    df_dummies = pd.concat([df_dummies_part1,df_dummies_part2], axis = 1)

    df = pd.concat([df.SeniorCitizen, df.tenure, df.MonthlyCharges, df.TotalCharges, df_dummies, df.Churn], axis = 1)
    # print(df.head())
    print(df.shape)
    for i in df.index:
        if df.loc[i,'Churn'] == 'Yes':
            df.loc[i, 'Churn'] = 1
        else:
            df.loc[i, 'Churn'] = 0
        # if df.loc[i,'TotalCharges'] == " ":
        #     df.loc[i,'TotalCharges'] = df.loc[i,'MonthlyCharges']
    print(df.head())
    # clean data type
    df.loc[:,'TotalCharges'] = pd.to_numeric(df.loc[:,'TotalCharges'], errors = 'coerce') # errors converts " " to NaN

    # double check if any null values exist
    if df.isnull().values.any() == True:
        df = df.dropna(axis = 0, how = 'any') # drop whole row if any NaN exists


    return df

def dataProcess(df):
    X = df.iloc[:,:-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle= False)

    featureImportance(X_train, y_train, df)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test

def featureImportance(X, y, df):
    forest = RandomForestClassifier()
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print(indices.shape)
    # Print the feature ranking
    print("Feature ranking:")
    featureList = list(df.columns.values)
    for f in range(X.shape[1]):
        print("%d. feature %d named %s (%f)" % (f + 1, indices[f], featureList[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    indices = indices[:10] # top 10 important feature indices
    plt.bar(range(10), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(10), indices)
    plt.xlim([-1, 10])
    plt.show()
    plt.close()


def MLmodel(modelName, X_train, y_train):
    if modelName == 'LR':
        model = LogisticRegression()

        print(X_train[:5,:])
        print(y_train[:5])
        model.fit(X_train, y_train)
        return model

def MLcrossValidation(model, X_train, y_train):
    crossValidationScore = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring ='roc_auc')
    print(crossValidationScore)

def MLpredict(model, X_test):
    y_test_predict = model.predict(X_test)
    y_test_score = model.decision_function(X_test)
    y_test_prob = model.predict_proba(X_test)
    # print(y_test_predict)
    return y_test_predict, y_test_score

def MLevaluate(y_test, y_test_score):
    fpr, tpr, threshold = roc_curve(y_test, y_test_score)
    auc_score = auc(fpr, tpr)
    print("AUC:", auc_score)



if __name__ == '__main__':
    df = loadData()
    X_train, X_test, y_train, y_test = dataProcess(df)
    model = MLmodel('LR', X_train, y_train)
    MLcrossValidation(model, X_train, y_train)
    _, y_test_score = MLpredict(model, X_test)
    MLevaluate(y_test, y_test_score)
