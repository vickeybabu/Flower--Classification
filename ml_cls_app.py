import pickle
import streamlit as st
# streamlit is aliased as st

# Load saved model

lr = pickle.load(open('lr_model.pkl','rb'))  #rb = read the binary files
dt = pickle.load(open('dt_model.pkl','rb'))
knn = pickle.load(open('knn_model.pkl','rb'))
rf = pickle.load(open('rf_model.pkl','rb'))

st.header('Iris Flower Classification ML Web App')

ml_model = ['Logistic Regression','KNeighbours Classifier','Decision Tree Classifier','RandomForest Classifier']

option = st.sidebar.selectbox('select one of the ML Model',ml_model)

sl = st.slider('Sepal Length',0.0,10.0)
sw = st.slider('Sepal Width',0.0,10.0)
pl = st.slider('petal Length',0.0,10.0)
pw = st.slider('petal Width',0.0,10.0)


test = [[sl,sw,pl,pw]]

st.write('Test data',test)

# calling the ML model for prediction

if st.button('Run Classifier'):
    if option=='Logistic Regression':
        st.success(lr.predict(test)[0])
    elif option== 'KNeighbors Classifier':
        st.success(knn.predict(test)[0])
    elif option== 'DecisionTree Classifier':
        st.success(dt.predict(test)[0])
    else :
          st.success(dt.predict(test)[0])




# To run this file in terminal
# streamlit run ml_cls_app.py
 #To stop the server - write the following
# ctrl + c
# To clear the screen 
# cl