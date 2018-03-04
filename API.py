from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd
import numpy as np
import ML
app = Flask(__name__)

app.secret_key = 'key'

model = pickle.load(open("./Models/GB.pkl","rb"))
df = ML.loadData()
featureList = list(ML.cleanFeature(df).columns.values)
df = pd.DataFrame(np.zeros(len(featureList)).reshape(1,-1), columns = featureList)
# print(featureList)
# print(df)

@app.route('/', methods = ['GET', 'POST'])
def index():
    # session.pop('proba')
    # session.pop('category')
    # session.pop('submitted')

    if 'submitted' in session:
        print("submitted in session:", session['submitted'])
    else:
        session['submitted'] = []
    if 'proba' in session:
        print("proba in session:",session['proba'])
    else:
        session['proba'] = []
    if 'category' in session:
        print("category in session:", session['category'])
    else:
        session['category'] = []
    if 'answer' in session:
        print("answer in session:", session['answer'])
    else:
        session['answer'] = []



    # model.predict()
    print(request.method)
    # session['proba'] = 0
    # session['answer'] = 'TBD'
    if request.method == 'POST' and request.form['submit'] == 'predict':
        gender = request.form['gender']
        seniorCitizen = request.form['seniorCitizen']
        partner = request.form['partner']
        dependents = request.form['dependents']
        tenure = request.form['tenure']
        phoneService = request.form['phoneService']
        multipleLines = request.form['multipleLines']
        internetService = request.form['internetService']
        onlineSecurity = request.form['onlineSecurity']
        onlineBackup = request.form['onlineBackup']
        deviceProtection = request.form['deviceProtection']
        techSupport = request.form['techSupport']
        streamingTV = request.form['streamingTV']
        streamingMovie = request.form['streamingMovie']
        contract = request.form['contract']
        paperlessBilling = request.form['paperlessBilling']
        paymentMethod = request.form['paymentMethod']
        monthlyCharges = request.form['monthlyCharges']
        totalCharges = request.form['totalCharges']
        df_input= pd.DataFrame({'CustomerID':[1], 'Gender':[gender], 'SeniorCitizen':[seniorCitizen], 'Partner':[partner],
                      'Dependents':[dependents], 'Tenure':[tenure], 'PhoneService':[phoneService], 'MultipleLines':[multipleLines],
                      'InternetService':[internetService], 'OnlineSecurity':[onlineSecurity], 'OnlineBackup':[onlineBackup],
                      'DeviceProtection':[deviceProtection], 'TechSupport':[techSupport], 'StreamingTV':[streamingTV],
                      'StreamingMovie':[streamingMovie], 'Contract':[contract], 'PaperlessBilling':[paperlessBilling],
                      'PaymentMethod':[paymentMethod], 'MonthlyCharges':[monthlyCharges], 'TotalCharges':[totalCharges]},
                                columns =['CustomerID','Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure', 'PhoneService',
                                          'MultipleLines', 'InternetService', 'OnlineSecurity',  'OnlineBackup', 'DeviceProtection',
                                          'TechSupport', 'StreamingTV', 'StreamingMovie', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                                          'MonthlyCharges', 'TotalCharges'])
        # print(df_online)

        df_partial = ML.cleanFeature(df_input)
        # print(df_partial)
        for i in df_partial.columns.values:
            for j in df.columns.values:
                if i == j:
                    df.loc[0,j] = df_partial.loc[0,i]

        # print(df)
        df.SeniorCitizen = pd.to_numeric(df.SeniorCitizen)
        df.Tenure = pd.to_numeric(df.Tenure)
        df.MonthlyCharges = pd.to_numeric(df.MonthlyCharges)
        # X = df.loc[0,:].values.reshape(1,-1)
        X = df.loc[0,:].values.reshape(1,-1)
        # print(X)
        category = model.predict(X)
        proba = model.predict_proba(X)

        session['submitted'] = True
        if category == [1]:
            answer = 'likely to quit'
        else:
            answer = 'not likely to quit'

        session['answer'] = answer
        session['proba'] = int(list(proba)[0][1] * 100)
        return redirect(url_for("index"))

    if request.method == 'POST' and request.form['submit'] == 'analysis':
        return redirect(url_for("analysis"))


    submitted = session['submitted']
    session['submitted'] = False
    proba = session['proba']
    answer = session['answer']
    return render_template('index.html', submitted = submitted, proba = proba, answer = answer)


@app.route("/analysis", methods = ['GET', 'POST'])
def analysis():
    print(request.method)

    if request.method == 'POST':
        return redirect(url_for('index'))
    return render_template("analysis.html")




if __name__ == "__main__":
    app.run(debug = True)