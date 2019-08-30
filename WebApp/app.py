from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():

    return render_template('upload.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    target = os.path.join(APP_ROOT, 'csv/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        global filename
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        preprocess(filename)

    return render_template("complete.html")

@app.route('/final.html')
def view():

    return render_template("final.html")

def preprocess(filename):
    """
    Transform raw input file into an ingestable form for ML model
    """
    df = pd.read_csv("csv/"+filename)
    for column in df.columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    unique_custID = df['customer_ID'].unique()
    unique_custID = list(unique_custID)

    df_clean = pd.DataFrame(columns=['customer_ID','num_quotes','state','location','group_size','homeowner','car_age',
                                'car_value','risk_factor','age_oldest','age_youngest','married_couple','C_previous',
                                'duration_previous','A_min','A_max','A_mode','A_median','B_min','B_max','B_mode',
                                'B_median','C_min','C_max','C_mode','C_median','D_min','D_max','D_mode','D_median',
                                'E_min','E_max','E_mode','E_median','F_min','F_max','F_mode','F_median',
                                'G_min','G_max','G_mode','G_median','cost_min','cost_max','cost_mode','cost_median',
                                'A_final','B_final','C_final','D_final','E_final','F_final','G_final'])

    for i in unique_custID:
        curr_ID = df.groupby(['customer_ID']).get_group(i)
        maxNo = curr_ID.shape[0] - 1
        customer_ID = i
        num_quotes = curr_ID['shopping_pt'].max()
        state = curr_ID['state'].iloc[0]
        location = curr_ID['location'].iloc[0]
        group_size = curr_ID['group_size'].iloc[0]
        homeowner = curr_ID['homeowner'].iloc[0]
        car_age = curr_ID['car_age'].iloc[0]
        car_value = curr_ID['car_value'].iloc[0]
        risk_factor = curr_ID['risk_factor'].iloc[0]
        age_oldest = curr_ID['age_oldest'].iloc[0]
        age_youngest = curr_ID['age_youngest'].iloc[0]
        married_couple = curr_ID['married_couple'].iloc[0]
        C_previous = curr_ID['C_previous'].iloc[0]
        duration_previous = curr_ID['duration_previous'].iloc[0]
        A_min = curr_ID['A'].min()
        A_max = curr_ID['A'].max()
        A_mode = curr_ID['A'].mode()[0]
        A_median = curr_ID['A'].median()
        B_min = curr_ID['B'].min()
        B_max = curr_ID['B'].max()
        B_mode = curr_ID['B'].mode()[0]
        B_median = curr_ID['B'].median()
        C_min = curr_ID['C'].min()
        C_max = curr_ID['C'].max()
        C_mode = curr_ID['C'].mode()[0]
        C_median = curr_ID['C'].median()
        D_min = curr_ID['D'].min()
        D_max = curr_ID['D'].max()
        D_mode = curr_ID['D'].mode()[0]
        D_median = curr_ID['D'].median()
        E_min = curr_ID['E'].min()
        E_max = curr_ID['E'].max()
        E_mode = curr_ID['E'].mode()[0]
        E_median = curr_ID['E'].median()
        F_min = curr_ID['F'].min()
        F_max = curr_ID['F'].max()
        F_mode = curr_ID['F'].mode()[0]
        F_median = curr_ID['F'].median()
        G_min = curr_ID['G'].min()
        G_max = curr_ID['G'].max()
        G_mode = curr_ID['G'].mode()[0]
        G_median = curr_ID['G'].median()
        cost_min = curr_ID['cost'].min()
        cost_max = curr_ID['cost'].max()
        cost_mode = curr_ID['cost'].mode()[0]
        cost_median = curr_ID['cost'].median()
        A_final = curr_ID['A'].iloc[maxNo]
        B_final = curr_ID['B'].iloc[maxNo]
        C_final = curr_ID['C'].iloc[maxNo]
        D_final = curr_ID['D'].iloc[maxNo]
        E_final = curr_ID['E'].iloc[maxNo]
        F_final = curr_ID['F'].iloc[maxNo]
        G_final = curr_ID['G'].iloc[maxNo]

        curr_ID_clean = pd.DataFrame({'customer_ID': [customer_ID],
                                      'num_quotes': [num_quotes],
                                      'state': [state],
                                      'location': [location],
                                      'group_size':[group_size],
                                      'homeowner': [homeowner],
                                      'car_age': [car_age],
                                      'car_value': [car_value],
                                      'risk_factor': [risk_factor],
                                      'age_oldest':[age_oldest],
                                      'age_youngest':[age_youngest],
                                      'married_couple': [married_couple],
                                      'C_previous': [C_previous],
                                      'duration_previous': [duration_previous],
                                      'A_min': [A_min],
                                      'A_max': [A_max],
                                      'A_mode': [A_mode],
                                      'A_median': [A_median],
                                      'B_min': [B_min],
                                      'B_max': [B_max],
                                      'B_mode': [B_mode],
                                      'B_median':[B_median],
                                      'C_min': [C_min],
                                      'C_max': [C_max],
                                      'C_mode':[C_mode],
                                      'C_median': [C_median],
                                      'D_min': [D_min],
                                      'D_max': [D_max],
                                      'D_mode': [D_mode],
                                      'D_median': [D_median],
                                      'E_min': [E_min],
                                      'E_max': [E_max],
                                      'E_mode': [E_mode],
                                      'E_median': [E_median],
                                      'F_min': [F_min],
                                      'F_max': [F_max],
                                      'F_mode': [F_mode],
                                      'F_median': [F_median],
                                      'G_min':[G_min],
                                      'G_max': [G_max],
                                      'G_mode': [G_mode],
                                      'G_median': [G_median],
                                      'cost_min': [cost_min],
                                      'cost_max': [cost_max],
                                      'cost_mode': [cost_mode],
                                      'cost_median': [cost_median],
                                      'A_final': [A_final],
                                      'B_final': [B_final],
                                      'C_final': [C_final],
                                      'D_final': [D_final],
                                      'E_final': [E_final],
                                      'F_final': [F_final],
                                      'G_final': [G_final]})
        df_clean = df_clean.append(curr_ID_clean, ignore_index=True)
        #end for loop

    count = list(range(0, df_clean.shape[0]))

    for i in count:
        if(df_clean['car_value'][i] == 'a'):
            df_clean.at[i, 'car_value'] = 1
        if(df_clean['car_value'][i] == 'b'):
            df_clean.at[i, 'car_value'] = 2
        if(df_clean['car_value'][i] == 'c'):
            df_clean.at[i, 'car_value'] = 3
        if(df_clean['car_value'][i] == 'd'):
            df_clean.at[i, 'car_value'] = 4
        if(df_clean['car_value'][i] == 'e'):
            df_clean.at[i, 'car_value'] = 5
        if(df_clean['car_value'][i] == 'f'):
            df_clean.at[i, 'car_value'] = 6
        if(df_clean['car_value'][i] == 'g'):
            df_clean.at[i, 'car_value'] = 7
        if(df_clean['car_value'][i] == 'h'):
            df_clean.at[i, 'car_value'] = 8
        if(df_clean['car_value'][i] == 'i'):
            df_clean.at[i, 'car_value'] = 9

    df_clean["car_value"] = df_clean['car_value'].astype(int)

    df_clean = pd.concat([df_clean,pd.get_dummies(df_clean['state'], prefix='state')],axis=1)
    df_clean = df_clean.drop(['state'], axis = 1)

    df_clean.to_csv("process_test2.csv")

    custMatch()


def custMatch():
    """
    Predict whether a customer will match their final quote given from insurance
    provider. Split into matched dataframe and a further processing dataframe
    """
    df_test = pd.read_csv("process_test2.csv")
    cust_match = pickle.load(open('GBM_Pred.txt', 'rb'))
    cust_match_pred = cust_match.predict(df_test)
    df_test['chosen_match_final'] = pd.Series(cust_match_pred, index=df_test.index)

    done = df_test.groupby(['chosen_match_final']).get_group(1)
    done.to_csv("done_df.csv")
    predictCat = df_test.groupby(['chosen_match_final']).get_group(0)
    predictCat.to_csv("predictCat_df.csv")
    catPrediction()


def catPrediction():
    """
    Further process subsetted dataframe to predict individual insurance coverage
    categories
    """
    #import data to predict
    df_pred_cat = pd.read_csv("predictCat_df.csv")
    df_pred_cat = df_pred_cat.drop(['Unnamed: 0.1', 'chosen_match_final'], axis = 1)

    #import models
    G_pred_rf_model = pickle.load(open('GBM_Pred_G.txt', 'rb'))

    #predict coverage categories
    A_pred = df_pred_cat['A_final']
    B_pred = df_pred_cat['B_final']
    C_pred = df_pred_cat['C_final']
    D_pred = df_pred_cat['D_final']
    E_pred = df_pred_cat['E_final']
    F_pred = df_pred_cat['F_final']
    G_pred = G_pred_rf_model.predict(df_pred_cat)

    #input category predictions into dataframe
    df_pred_cat['A_chosen'] = pd.Series(A_pred, index=df_pred_cat.index)
    df_pred_cat['B_chosen'] = pd.Series(B_pred, index=df_pred_cat.index)
    df_pred_cat['C_chosen'] = pd.Series(C_pred, index=df_pred_cat.index)
    df_pred_cat['D_chosen'] = pd.Series(D_pred, index=df_pred_cat.index)
    df_pred_cat['E_chosen'] = pd.Series(E_pred, index=df_pred_cat.index)
    df_pred_cat['F_chosen'] = pd.Series(F_pred, index=df_pred_cat.index)
    df_pred_cat['G_chosen'] = pd.Series(G_pred, index=df_pred_cat.index)

    #save dataframe with predictions
    df_pred_cat.to_csv("df_predicted_cat.csv")
    join()


def join():
    """
    Format 'matched' dataframe and joine with predicted dataframe. Format into
    submission csv format
    """
    #import match dataframe
    cust_matched_final = pd.read_csv("done_df.csv")
    cust_matched_final = cust_matched_final.drop(['Unnamed: 0.1'], axis = 1)

    #create new 'chosen' columns based on final customer quote received
    cust_matched_final['A_chosen'] = cust_matched_final['A_final']
    cust_matched_final['B_chosen'] = cust_matched_final['B_final']
    cust_matched_final['C_chosen'] = cust_matched_final['C_final']
    cust_matched_final['D_chosen'] = cust_matched_final['D_final']
    cust_matched_final['E_chosen'] = cust_matched_final['E_final']
    cust_matched_final['F_chosen'] = cust_matched_final['F_final']
    cust_matched_final['G_chosen'] = cust_matched_final['G_final']
    cust_matched_final = cust_matched_final.drop(['chosen_match_final'], axis = 1)

    df_predicted_cat = pd.read_csv("df_predicted_cat.csv", index_col = None)
    df_predicted_cat = df_predicted_cat.drop(['Unnamed: 0.1'], axis = 1)
    joined_df = cust_matched_final.append(df_predicted_cat, ignore_index=True)
    joined_df.sort_values(by=['customer_ID'])

    #combine plan choices into a single string
    cols = ['A_chosen', 'B_chosen','C_chosen','D_chosen', 'E_chosen','F_chosen','G_chosen']
    for i in cols:
        joined_df[i] = joined_df[i].astype(str)
    joined_df['plan'] = joined_df[cols].apply(lambda x: ''.join(x), axis = 1)

    #create submission dataframe with customerID and concatenated plan choices string
    submission_df = joined_df.filter(['customer_ID','plan'], axis = 1)
    submission_df.to_csv("submission.csv", index = False)

    submission_df.to_html("final.html", index=False)
    submission_df.to_html("templates/final.html", index=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
