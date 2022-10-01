import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
from plotly.graph_objs import Scatter, Layout
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

model = pickle.load(open('model_reg.pkl','rb'))
model1 = pickle.load(open('model_reg_Alcohol.pkl','rb'))
model2 = pickle.load(open('model_reg_Amphet.pkl','rb'))
model3 = pickle.load(open('model_reg_Amyl.pkl','rb'))
model4 = pickle.load(open('model_reg_Caff.pkl','rb'))
model5 = pickle.load(open('model_reg_Cannabis.pkl','rb'))
model6 = pickle.load(open('model_reg_Choc.pkl','rb'))
model7 = pickle.load(open('model_reg_Coke.pkl','rb'))
model8 = pickle.load(open('model_reg_Crack.pkl','rb'))
model9 = pickle.load(open('model_reg_Ecstasy.pkl','rb'))
model10 = pickle.load(open('model_reg_Heroin.pkl','rb'))
model11 = pickle.load(open('model_reg_Ketamine.pkl','rb'))
model12 = pickle.load(open('model_reg_Legalh.pkl','rb'))
model13 = pickle.load(open('model_reg_LSD.pkl','rb'))
model14 = pickle.load(open('model_reg_Meth.pkl','rb'))
model15 = pickle.load(open('model_reg_Mushrooms.pkl','rb'))
model16 = pickle.load(open('model_reg_Nicotine.pkl','rb'))
model17 = pickle.load(open('model_reg_Semer.pkl','rb'))
model18 = pickle.load(open('model_reg_VSA.pkl','rb'))

# st.title("Drug usage prediction")

def fun(drug,num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model):

    res = model.predict([[num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12]])
    if(res[0]==0):
        st1 = "Do they use " + drug +"? No"
        st.subheader(st1)
    else:
        st1 = "Do they use " + drug +"? Yes"
        st.subheader(st1)

def main():

    new_title = '<p style="font-family:sans-serif; font-size: 50px;"><strong>Drug Usage Prediction</strong></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.image("tabs.jpg")

    st.subheader("Enter the details")
    
    option = st.selectbox(
        "Enter Age: ",
        ('18 - 24','25 - 34','35 - 44','45 - 54','55 - 64','65+'))
    if(option=="18 - 24"):
        num1 = 0
    elif(option=="25 - 34"):
        num1 = 1
    elif(option=="35 - 44"):
        num1 = 2
    elif(option=="45 - 54"):
        num1 = 3
    elif(option=="55 - 64"):
        num1 = 4
    else:
        num1 = 5

    st.write("\n")

    option = st.selectbox(
        "Enter Gender: ",
        ('Female','Male'))
    if(option=="Female"):
        num2 = 1
    else:
        num2 = 0

    st.write("\n")

    option = st.selectbox(
        "Enter Education: ",
        ('Left School Before 16 years','Left School at 16 years','Left School at 17 years','Left School at 18 years','Some College, No Certificate Or Degree', 'Professional Certificate/ Diploma', 'University Degree', 'Masters Degree', 'Doctorate Degree'))
    if(option=="Left School Before 16 years"):
        num3 = 0
    elif(option=="Left School at 16 years"):
        num3 = 1
    elif(option=="Left School at 17 years"):
        num3 = 2
    elif(option=="Left School at 18 years"):
        num3 = 3
    elif(option=="Some College,No Certificate Or Degree"):
        num3 = 4
    elif(option=="Professional Certificate/ Diploma"):
        num3 = 5
    elif(option=="University Degree"):
        num3 = 6
    elif(option=="Masters Degree"):
        num3 = 7
    elif(option=="Doctorate Degree"):
        num3 = 8

    st.write("\n")

    option = st.selectbox(
        "Enter Country: ",
        ('Australia','Canada','New Zealand','Other','Republic of Ireland','UK','USA'))
    if(option=="Australia"):
        num4 = 3
    elif(option=="Canada"):
        num4 = 5
    elif(option=="New Zealand"):
        num4 = 1
    elif(option=="Other"):
        num4 = 2
    elif(option=="Republic of Ireland"):
        num4 = 4
    elif(option=="UK"):
        num4 = 6
    elif(option=="USA"):
        num4 = 0

    st.write("\n")

    option = st.selectbox(
        "Enter Ethnicity: ",
        ('Asian','Black','Mixed-Black/Asian','Mixed-White/Asian','Mixed-White/Black','Other','White'))
    if(option=="Asian"):
        num5 = 1
    elif(option=="Black"):
        num5 = 0
    elif(option=="Mixed-Black/Asian"):
        num5 = 2
    elif(option=="Mixed-White/Asian"):
        num5 = 5
    elif(option=="Mixed-White/Black"):
        num5 = 3
    elif(option=="Other"):
        num5 = 4
    elif(option=="White"):
        num5 = 6

    st.write("\n")

    num6 = int(st.number_input('Enter Nscore (0-49): ')) 
    st.write("\n")
    num7 = int(st.number_input('Enter Escore (0-41): '))
    st.write("\n")
    num8 = int(st.number_input('Enter Oscore (0-34): '))
    st.write("\n")
    num9 = int(st.number_input('Enter Ascore (0-40): '))
    st.write("\n")
    num10 = int(st.number_input('Enter Cscore (0-40): '))
    st.write("\n")
    num11 = int(st.number_input('Enter Impulsive (0-9): '))
    st.write("\n")
    num12 = int(st.number_input('Enter Sensation seeing (0-10): '))
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("--------------------------------------------------------------------------")
    st.write("\n")

    new_title = '<p style="font-family:sans-serif; font-size: 50px;"><strong>Result</strong></p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.image("result.jpg")

    option = st.selectbox(
        'Choose the testing drug',
        ('Alcohol','amphetamines', 'amyl nitrite', 'benzodiazepine', 'caffeine', 'Cannabis', 'chocolate', 'cocaine', 'Crack',
           'Ecstasy', 'Heroin', 'Ketamine', 'legal highs', 'LSD', 'Meth', 'Mushrooms','Nicotine', 'Semer', 'volatile substance abuse'))

    if(option=="Alcohol"):
        fun("Alcohol",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model1)
    elif(option=="amphetamines"):
        fun("amphetamines",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model2)
    elif(option=="amyl nitrite"):
        fun("amyl nitrite",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model3)
    elif(option=="benzodiazepine"):
        fun("benzodiazepine",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model)
    elif(option=="caffeine"):
        fun("caffeine",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model4)
    elif(option=="Cannabis"):
        fun("Cannabis",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model5)
    elif(option=="chocolate"):
        fun("chocolate",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model6)
    elif(option=="cocaine"):
        fun("cocaine",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model7)
    elif(option=="Crack"):
        fun("Crack",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model8)
    elif(option=="Ecstasy"):
        fun("Ecstasy",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model9)
    elif(option=="Heroin"):
        fun("Heroin",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model10)
    elif(option=="Ketamine"):
        fun("Ketamine",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model11)
    elif(option=="legal highs"):
        fun("legal highs",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model12)
    elif(option=="LSD"):
        fun("LSD",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model13)
    elif(option=="Meth"):
        fun("Meth",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model14)
    elif(option=="Mushrooms"):
        fun("Mushrooms",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model15)
    elif(option=="Nicotine"):
        fun("Nicotine",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model16)
    elif(option=="Semer"):
        fun("Semer",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model17)
    elif(option=="volatile substance abuse"):
        fun("volatile substance abuse",num1,num2,num3,num4,num5,num6,num7,num8,num9,num10,num11,num12,model18)    

if __name__ == "__main__":
    main()

