import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
data = pd.read_csv("water_potability.csv")

st.write(data)


def Potability():
    ps=data.groupby(by=["Potability"]).size().reset_index(name="counts")
    ps["counts"] = ps["counts"].values.ravel()
    figure=px.bar(data_frame=ps,x="Potability",y="counts",color=["Unsafe","Safe"],title="Distribution of Unsafe and Safe Water")
    plt.figure(figsize=(15, 10))
    st.plotly_chart(figure)

def PH():
    figure = px.histogram(data, x = "ph", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: PH")
    st.plotly_chart(figure)

def Hardness():
    figure = px.histogram(data, x = "ph", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: PH")
    st.plotly_chart(figure)

def solids():
    figure = px.histogram(data, x = "Solids", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Solids")
    st.plotly_chart(figure)

def Cho():
    figure = px.histogram(data, x = "Chloramines", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Chloramines")
    st.plotly_chart(figure)

def Sulphate():
    figure = px.histogram(data, x = "Sulfate", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Sulfate")
    st.plotly_chart(figure)

def Conductivity():
    figure = px.histogram(data, x = "Conductivity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Conductivity")
    st.plotly_chart(figure)

def Organic_carbon():   
    figure = px.histogram(data, x = "Organic_carbon", 
                        color = "Potability", 
                        title= "Factors Affecting Water Quality: Organic Carbon")
    st.plotly_chart(figure)

def Thm():   
    figure = px.histogram(data, x = "Trihalomethanes", 
                        color = "Potability", 
                        title= "Factors Affecting Water Quality: Trihalomethanes")
    st.plotly_chart(figure)

def Turbidity():
    figure = px.histogram(data, x = "Turbidity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Turbidity")
    st.plotly_chart(figure)


gp=st.selectbox("Visulization",["Unsafe and safe water","PH","Hardness","Solids","Chloramines","Sulphate","Conductivity"
                                ,"Organic_carbon","Trihalomethanes","Turbidity"])

if gp=="Unsafe and safe water":
    Potability()
elif gp=="PH":
    PH()
elif gp=="Hardness":
    Hardness()
elif gp=="Solids":
    solids()
elif gp=="Chloramines":
    Cho()
elif gp=="Sulphate":
    Sulphate()
elif gp=="Conductivity":
    Conductivity()
elif gp=="Organic_carbon":
    Organic_carbon()
elif gp=="Trihalomethanes":
    Thm()
elif gp=="Turbidity":
    Turbidity()

x=data.drop("Potability",axis=1)
y=data[["Potability"]]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
pre=model.predict(x_test)
acc=confusion_matrix(y_test,pre)
acc1=accuracy_score(y_test,pre)

st.write("Accuracy Score",acc1)


labels = ['True Neg'],['False Pos'],['False Neg'],['True Pos']
labels = np.asarray(labels).reshape(2,2)
fig=px.imshow(acc,color_continuous_scale='Viridis',aspect="auto")
fig.update_traces(text=labels,texttemplate="%{text}")
fig.update_xaxes(side="top")
st.plotly_chart(fig)

st.write("Write data to find water potability")
a = float(st.text_input("Enter the Ph", 0))
b = float(st.text_input("Enter the Hardness", 0))
c = float(st.text_input("Enter the Solids", 0))
d = float(st.text_input("Enter the Chloramines", 0))
e = float(st.text_input("Enter the Sulphate", 0))
f = float(st.text_input("Enter the Conductivity", 0))
g = float(st.text_input("Enter the Organic_carbon", 0))
h = float(st.text_input("Enter the Trihalomethanes", 0))
i = float(st.text_input("Enter the Turbidity", 0))

sample = np.array([[a, b, c, d, e, f, g, h, i]])

Ans = model.predict(sample)

st.write(f"The predicted potability is: {Ans[0]}")