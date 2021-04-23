# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:44:03 2021

@author: ksawant
"""

import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
import numpy as np

st.title("Predict sales based on Youtube Marketing")
#st.header("Application to predict sales based on Youtube Marketing")
st.sidebar.header("user inputs")

def user_inputs():
    youtube_budget=st.sidebar.number_input("Enter the youtube budget")
    youtube_Sq=youtube_budget*youtube_budget
    inputdata={"youtube":youtube_budget,"youtube_Sq":youtube_Sq}
    inputdata=pd.DataFrame(inputdata,index=[0])
    return inputdata
   
inputs=user_inputs()

st.write(inputs)

#build the moidel#

train=pd.read_csv("marketing.csv")

train["youtube_Sq"] = train.youtube*train.youtube

model=smf.ols("np.log(sales)~youtube+youtube_Sq",data=train).fit()
              
prediction=np.exp(model.predict(inputs))







st.write("Estimated sales for the given youtube Budget")
st.write(prediction)


#for interval


st.write("Estimated sales for the given youtube Budget may would lie in the range")
from scipy import stats
#ci=pd.DataFrame(stats.norm.interval(0.95,prediction))

#st.write(ci)

#or
ci=(stats.norm.interval(0.95,prediction))
st.write(ci[0],ci[1])







