import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn import metrics
from pickle import dump
import joblib
import altair as alt

st.write(""" 
# APLIKASI DIABETES PREDICTION Dibuat Oleh A. Makmun alji | 210411100241""")

import_data, preprocessing, modeling, implementation, evaluation = st.tabs(["Import Data", "PreProcessing", "Modeling", "Implementation", "Evaluation"])

with import_data:
    st.write("# IMPORT DATA")
    uploaded_files = st.file_uploader("Upload Data Set yang Mau Digunakan", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        data = pd.read_csv(uploaded_file)
        st.write("Nama Dataset:", uploaded_file.name)
        st.write(data)

with preprocessing:
    st.write("# PREPROCESSING")
    encoding = st.checkbox("Encoding (Category to Numeric)")
    normalisasi = st.checkbox("Normalisasi dengan MinMaxScallar")

    if encoding:
        st.write("## Kamu Memilih Encoding (Category to Numeric)")
        data_baru = data.drop(columns=["Outcome"])
        data_baru['Pregnancies']= (data_baru["Pregnancies"]== "Y").astype(int)
        st.write("Menampilkan data tanpa ID, kita tidak memerlukan kolom ID dan melakukan encode atau merubah data categorical ke numeric pada kolom Hepatomegaly, data dengan nilai Y akan bernilai 1 dan jika N maka akan bernilai 0")
        st.dataframe(data_baru)
    if normalisasi:
        st.write("## Kamu Memilih Normalisasi")
        st.write("Melakukan Normalisasi pada semua fitur kecuali Hepatomegaly karena Hepatomegaly akan digunakan sebagai data class sebagai output impelentasi nantinya")
        data_baru = data.drop(columns=["Outcome"])
        data_baru['Hepatomegaly']= (data_baru["Hepatomegaly"]== "M").astype(int)
        sebelum_dinormalisasi = ['Bilirubin', "Cholesterol","Albumin", "Copper", "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"]
        setelah_dinormalisasi = ["norm_Bilirubin", "norm_Cholesterol","norm_Albumin", "norm_Copper", "norm_Alk_Phos", "norm_SGOT", "norm_Tryglicerides", "norm_Platelets", "Prothrombin"]

        normalisasi_fitur = data[sebelum_dinormalisasi]
        st.dataframe(normalisasi_fitur)

        scaler = MinMaxScaler()
        scaler.fit(normalisasi_fitur)
        fitur_ternormalisasi = scaler.transform(normalisasi_fitur)
        
        # save normalisasi
        joblib.dump(scaler, 'normal')

        fitur_ternormalisasi_df = pd.DataFrame(fitur_ternormalisasi, columns = setelah_dinormalisasi)

        st.write("Data yang telah dinormalisasi")
        st.dataframe(fitur_ternormalisasi)

        data_sudah_normal = data_baru.drop(columns=sebelum_dinormalisasi)
        
        data_sudah_normal = data_sudah_normal.join(fitur_ternormalisasi_df)

        st.write("data yang sudah dinormalisasi dan sudah disatukan dalam 1 sata frame")
        st.dataframe(data_sudah_normal)

with modeling:
    st.write("# MODELING")

    Y = data_sudah_normal['Hepatomegaly']
    # st.dataframe(Y)
    X = data_sudah_normal.iloc[sebelum_dinormalisasi]
    # st.dataframe(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

    ### Dictionary to store model and its accuracy

    model_accuracy = OrderedDict()

    ### Dictionary to store model and its precision

    model_precision = OrderedDict()

    ### Dictionary to store model and its recall

    model_recall = OrderedDict()
    
    # Naive Bayes
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train, y_train)
    Y_pred_nb = naive_bayes_classifier.predict(X_test)

    # decision tree
    clf_dt = DecisionTreeClassifier(criterion="gini")
    clf_dt = clf_dt.fit(X_train, y_train)
    Y_pred_dt = clf_dt.predict(X_test)
    
    # Bagging Decision tree
    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10, random_state=0).fit(X_train, y_train)
    rsc = clf.predict(X_test)
    c = ['Naive Bayes']
    tree = pd.DataFrame(rsc,columns = c)

    # save model dengan akurasi tertinggi
    joblib.dump(clf, 'bagging_decisionT')

    # K-Nearest Neighboor
    k_range = range(1,26)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        Y_pred_knn = knn.predict(X_test)

    naive_bayes_accuracy = round(100 * accuracy_score(y_test, Y_pred_nb), 2)
    decision_tree_accuracy = round(100* metrics.accuracy_score(y_test, Y_pred_dt))
    model_accuracy['Gaussian Naive Bayes'] = naive_bayes_accuracy
    bagging_Dc = round(100 * accuracy_score(y_test, tree), 2)
    knn_accuracy = round(100 * accuracy_score(y_test, Y_pred_knn), 2)
    

    st.write("Pilih Metode : ")
    naive_bayes_cb = st.checkbox("Naive Bayes")
    decision_tree_cb = st.checkbox("Decision Tree")
    bagging_tree_cb = st.checkbox("Bagging Decision Tree")
    knn_cb = st.checkbox("K-Nearest Neighboor")

    if naive_bayes_cb:
        st.write('Akurasi Metode Naive Bayes {} %.'.format(naive_bayes_accuracy))
    if decision_tree_cb:
        st.write('Akurasi Metode Decision Tree {} %.'.format(decision_tree_accuracy))
    if bagging_tree_cb:
        st.write('Akurasi Metode Bagging Decision Tree {} %.'.format(bagging_Dc))
    if knn_cb:
        st.write('Akurasi Metode KNN {} %.'.format(knn_accuracy))


with implementation:
    st.write("# IMPLEMENTATION")
    nama_pasien = st.text_input("Masukkan Nama")
    Bilirubin = st.number_input("Masukkan Rata-rata Bilirubin", min_value=0.5000, max_value=30.0000)
    Cholesterol = st.number_input("Masukkan rata-rata Cholesterol", min_value=0.0, max_value=99.0000)
    Albumin = st.number_input("Masukkan rata-rata Albumin", min_value=0.0, max_value=5.0000)
    Copper = st.number_input("Masukkan rata-rata Copper", min_value=0.0, max_value=300.0000)
    Alk_Phos = st.number_input("Masukkan Rata-rata Alk_Phos", min_value=0.0, max_value=99.0000)
    SGOT = st.number_input("Masukkan rata-rata SGOT", min_value=0.0, max_value=300.0000)
    Tryglicerides = st.number_input("Masukkan rata-rata Tryglicerides", min_value=0.0, max_value=300.0000)
    Platelets = st.number_input("Masukkan rata-rata Platelets", min_value=0.0, max_value=500.0000)
    Prothrombin = st.number_input("Masukkan rata-rata Prothrombin", min_value=0.0, max_value=20.0000)

    st.write("Cek apakah liver ini termasuk kategori normal atau cirrhosis")
    cek_bagging_tree = st.button('Cek Liver')
    inputan = [[Bilirubin, Cholesterol, Albumin, Copper, Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin]]

    scaler_jl = joblib.load('normal')
    scaler_jl.fit(inputan)
    inputan_normal = scaler.transform(inputan)

    FIRST_IDX = 0
    bagging_decision_tree = joblib.load("bagging_decisionT")
    if cek_bagging_tree:
        hasil_test = bagging_decision_tree.predict(inputan_normal)[FIRST_IDX]
        if hasil_test == 0:
            st.write("Nama Customer ", nama_pasien , "Mengidap Liver Normal Berdasarkan Model bagging decision tree")
        else:
            st.write("Nama Customer ", nama_pasien , "Mengidap Liver Cirrhosis Berdasarkan Model bagging decision tree")

with evaluation:
    st.write("# EVALUATION")
    bagan = pd.DataFrame({'Akurasi ' : [naive_bayes_accuracy,decision_tree_accuracy, bagging_Dc, knn_accuracy], 'Metode' : ["Naive Bayes", "Decision Tree", "Bagging Decision Tree", "K-Nearest Neighboor"]})

    bar_chart = alt.Chart(bagan).mark_bar().encode(
        y = 'Akurasi ',
        x = 'Metode',
    )

    st.altair_chart(bar_chart, use_container_width=True)

