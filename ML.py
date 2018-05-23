import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def loadData():
    df = pd.read_csv('./DATA/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # clean data type
    df.loc[:, 'TotalCharges'] = pd.to_numeric(df.loc[:, 'TotalCharges'], errors='coerce')  # errors converts " " to NaN
    # double check if any null values exist
    if df.isnull().values.any() == True:
        df = df.dropna(axis=0, how='any')  # drop whole row if any NaN exists
    return df

def cleanFeature(df):
    # clean data type
    df.loc[:, 'TotalCharges'] = pd.to_numeric(df.loc[:, 'TotalCharges'], errors='coerce')  # errors converts " " to NaN

    # print(df.head())
    # print(df)
    df_dummies_part1 = pd.get_dummies(df.loc[:, ['Gender', 'Partner','Dependents']])
    df_dummies_part2 = pd.get_dummies(df.loc[:,'PhoneService' : 'PaymentMethod'])
    df_dummies = pd.concat([df_dummies_part1,df_dummies_part2], axis = 1)
    df_features = pd.concat([df.SeniorCitizen, df.Tenure, df.MonthlyCharges, df.TotalCharges, df_dummies], axis = 1)
    # df = pd.concat([df.SeniorCitizen, df.Tenure, df.MonthlyCharges, df.TotalCharges, df_dummies, df.Churn], axis = 1)

    # print(df.head())
    # print(df.shape)
    return df_features


def cleanLabel(df,df_features):
    df = pd.concat([df_features, df.Churn], axis = 1)
    for i in df.index:
        if df.loc[i,'Churn'] == 'Yes':
            df.loc[i, 'Churn'] = 1
        else:
            df.loc[i, 'Churn'] = 0
        # if df.loc[i,'TotalCharges'] == " ":
        #     df.loc[i,'TotalCharges'] = df.loc[i,'MonthlyCharges']
    # print(df.head())
    return df



def dataProcess(df):
    X = df.iloc[:,:-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle= False)

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    return X_train, X_test, y_train, y_test


def visualizeData(df):
    print('plotting seaborn plot')
    cols = ['Tenure', 'MonthlyCharges','TotalCharges']
    sns.set(style = 'whitegrid', context = 'notebook')
    sns.pairplot(df[cols], size = 2.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./static/figures/CovTenureMonthTotal.png', dpi=300)
    plt.close()

    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1)
    sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./static/figures/heatMapTenureMonthTotal.png', dpi=300)
    plt.close()


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
    # plt.figure()
    # plt.title("Feature importances")
    # indices = indices[:10] # top 10 important feature indices
    # plt.bar(range(10), importances[indices], color="r", yerr=std[indices], align="center")
    # plt.xticks(range(10), indices)
    # plt.xlim([-1, 10])
    # # plt.show()
    # plt.savefig('./static/figures/featureImportance.png', dpi=300)
    # plt.close()


def MLmodel(modelName, X_train, y_train):
    if modelName == 'LR':
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model
    elif modelName == 'RF':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model
    elif modelName == 'GB':
        model = GradientBoostingClassifier()
        grid_vals = {'learning_rate':[0.1, 0.3, 0.5], 'max_depth':[3, 4, 5], 'n_estimators': [100,300,500]}
        # model = GridSearchCV(estimator = model, scoring = 'roc_auc', param_grid = grid_vals, cv = None, verbose = 2)

        model.fit(X_train, y_train)
        # print(model.best_estimator_)
        return model

def MLcrossValidation(model, X_train, y_train):
    crossValidationScore = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10, scoring ='roc_auc')
    print('avg cross validation score:',crossValidationScore.mean())

def MLpredict(model, X_test):
    y_test_predict = model.predict(X_test)
    y_test_score = model.decision_function(X_test)
    y_test_proba = model.predict_proba(X_test)
    # print(y_test_predict)
    # print(y_test_proba)
    # print(y_test_predict)
    return y_test_predict, y_test_score, y_test_proba

def MLevaluate(y_test, y_test_score):
    fpr, tpr, threshold = roc_curve(y_test, y_test_score)
    auc_score = auc(fpr, tpr)
    print("AUC:", auc_score)
    return fpr, tpr, auc_score

def visualizeROC(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    plt.savefig('./static/figures/roc.png', dpi=300)
    plt.close()

def storeModel(model, modelName):
    import pickle
    pickle.dump(model, open("./Models/" + modelName + ".pkl", "wb"))
    print()


if __name__ == '__main__':
    df= loadData()
    df_features = cleanFeature(df)
    df = cleanLabel(df, df_features)
    X_train, X_test, y_train, y_test = dataProcess(df)
    # visualizeData(df)
    featureImportance(X_train, y_train, df)
    modelName = 'GB'
    model = MLmodel(modelName, X_train, y_train)
    MLcrossValidation(model, X_train, y_train)
    y_test_predict, y_test_score, y_test_proba = MLpredict(model, X_test)
    fpr, tpr, auc_score = MLevaluate(y_test, y_test_score)
    visualizeROC(fpr, tpr)
    storeModel(model, modelName)