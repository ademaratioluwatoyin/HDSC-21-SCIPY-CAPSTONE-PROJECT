#!/usr/bin/env python
# coding: utf-8

# In[8]:



from IPython import get_ipython
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
 
# loading the trained model
pickle_in = open('model.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Name, Platform, Year, Genre, Publisher):   
 
    # Pre-processing user input    
    if Name != "":
        Name = encoder.fit_transform(Name)
        Name = Name.reshape(-1, 1)
    else:
        Name = 0
     
    if Platform != "":
        Platform = encoder.fit_transform(Platform).reshape(-1, 1)
        
    else:
        Platform = 0
        
    '''if Year != "":
        Year = encoder.fit_transform(Year)
    else:
        Year = 0'''
        
    if Genre != "":
        Genre = encoder.fit_transform(Genre).reshape(-1, 1)
    else:
        Genre = 0
        
    if Publisher != "":
        Publisher = encoder.fit_transform(Publisher).reshape(-1, 1)
    else:
        Publisher = 0
    
    
    # Making predictions 
    prediction = classifier.predict( 
        Name, Platform, Year, Genre, Publisher)
     
    output = round(prediction, 2)

    return 'Sales should be $ {}'.format(output)

      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:black;text-align:center;">VideoGame_Sales_Predictions</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Name = st.text_input("Name")
    Year = st.number_input("Year") 
    Platform = st.text_input("Platform")
    Genre = st.text_input("Genre")
    Publisher = st.text_input("Publisher")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction (Name, Platform, Year, Genre, Publisher)
        st.success('Sales should be {}'.format(result))
        
     
if __name__=='__main__': 
    main()


# In[ ]:




