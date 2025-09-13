import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
st.html("<h1 style='color:blue; font-size:45x;'>Score Prediction Based On Hours Of Study</h1>")
df=pd.DataFrame(pd.read_csv("student_scores.csv"))
df.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
X=df[['hours']]
y=df['scores']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=22)

poly=PolynomialFeatures(degree=4)

X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)
model=LinearRegression()

model.fit(X_train_poly,y_train)
y_pred=model.predict(X_test_poly)

sc=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
st.html(f"""<span style='color:red;'><b>Model r2 score: </b></span>
        <span style='color:green;'><b>{sc*100:.2f}</b></span>""")
st.subheader("Sample Prediction Of Score")
df=pd.DataFrame({'Actual Score':y_test.values[:5],'Predicted Score':y_pred[:5]})
st.dataframe(df)
a=st.number_input("Hours Of Study:")
if st.button("Predict"):
    apoly=poly.transform([[a]])
    pred=model.predict(apoly)[0]
    st.success(f"Score: {pred}")
