import numpy as np
import streamlit as st
from sklearn.datasets import make_regression
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing
from numpy import log as ln
from sklearn.metrics import mean_squared_error


def model_with_threshold(x:np.ndarray,y:np.ndarray,alpha:float=10**-7*9,threshold:int=3,num_of_iter:int=2000):
    beta = np.random.random(2)
    
    y_ln=ln(y)

    st.header("Algorithm")
    st.latex(r"L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i)/threshold)^2 }")
    st.latex(r"Derivative for  \beta_0 =-2* \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i)/threshold^2)^2 }") 
    st.latex(r"Derivative for  \beta_1 =-2*x\sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i)/threshold^2)^2 }") 

    st.write(" price cannot be -  so I use ln(price) then return e^result to overcome that stuation ")
    st.write("in this case mse=1.1949060618195373 ")
    st.write("When ı train model without consider price cannot be negatif number , then mse=0.7011311504478739")

     
    loss=[]
    b0=[]
    b1=[]
    regloss = pd.DataFrame(dict( x=x, y=y_ln))
   
    fig = px.scatter(regloss, x="x", y="y",title="MedInc - Price")

    st.plotly_chart(fig, use_container_width=True)

    for i in range(num_of_iter):
        y_pred: np.ndarray = (beta[0] + beta[1] * x)      
        g_b0 = -2 * ((y_ln - y_pred)/threshold*2).sum()
        g_b1 = -2 * (x * (y_ln- y_pred)/threshold*2).sum()

        b0.append(beta[0])
        b1.append(beta[1])
        loss.append(np.power((y - beta[0] - beta[1] * x )/threshold, 2).sum())
        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}")
        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

    # _df = pd.DataFrame(dict(b0=b0,b1=b1,loss=loss))
    # fig = px.scatter(_df, x="b0", y="loss",title="Loss-b0")
    # st.plotly_chart(fig, use_container_width=True)
    # fig = px.scatter(_df, x="b1", y="loss",title="Loss-b1")
    # st.plotly_chart(fig, use_container_width=True)
    

  

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='data points'))
    fig.add_trace(go.Scatter(x=x, y=beta[0] + beta[1] * x, mode='lines', name='regresssion'))
    st.plotly_chart(fig, use_container_width=True)
    y_pred_ln = beta[0] + beta[1] * x
    y_pred_actual= np.exp(y_pred_ln)
   
    return beta,y_pred_actual


def main():
    cal_housing = fetch_california_housing() 
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names) 
    y = cal_housing.target 

    df = pd.DataFrame( 
            dict(MedInc=X['MedInc'], Price=cal_housing.target))  
    x=df["MedInc"]
  
    beta,y_pred=model_with_threshold(x.to_numpy(),y,alpha=10**-7*9,threshold=3,num_of_iter=2000)
    st.subheader(r"End of train beta values :")
    st.latex(fr"\beta_0: {beta[0]} \\ \beta_1: {beta[1]} ")
    st.subheader(f"Mean square error: {mean_squared_error(y,y_pred)}")
if __name__ == '__main__':
    main()
