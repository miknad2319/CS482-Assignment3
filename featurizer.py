import streamlit as st
import pandas as pd
import numpy as np
import os
from os import path
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.neighbors import NearestNeighbors

feature_vecs = pd.read_csv("face_feature_vecs.csv")
feature_vecs = feature_vecs.iloc[:, 1:]
feature_vecs_array = feature_vecs.to_numpy()

neighborhood = NearestNeighbors(n_neighbors=11)
neighborhood.fit(feature_vecs_array)

face_labels_df = pd.read_csv("face_labels.csv")
face_labels_df = pd.DataFrame({"Name" : face_labels_df["0"]})
face_labels = face_labels_df["Name"].values
current_dir = os.getcwd()
faces_dir = os.path.join(current_dir, "faces/")

# test_indices = np.random.randint(0, len(face_labels), 10)
# neighborhood = test_indices

print(faces_dir)
with st.form(key="init_form"):
   
    choice = st.selectbox("Choose Picture", face_labels)
    face_index = np.where(face_labels == choice)[0]
    img_path = os.path.join(faces_dir, choice)
    # st.image(img_path)
    st.image(img_path) 

    neighbors = neighborhood.kneighbors(feature_vecs_array[face_index].reshape(1,-1))
    neighbors = neighbors[-1][0][1:]
    
    face_paths = [os.path.join(faces_dir, face_labels[index]) for index in neighbors]
    # The index of choice in model_pointers will access the models list
    # and select the Hugging Face model path at index.  
    analyze = st.form_submit_button("Analyze")

if analyze:
    with st.spinner("Analyzing..."):
        
        st.write("Nothing Yet")
        

        cols = cycle(st.columns(5)) # st.columns here since it is out of beta at the time I'm writing this
        for idx, face in enumerate(face_paths):
            next(cols).image(face, width=150, caption=face_labels[neighbors[idx]])
            
        # sentiment_pipeline = pipeline(model=user_picked_model)
        # sentiment_results=sentiment_pipeline(input_text)
        # st.write(f"Sentiment: {sentiment_results[0]['label']}")
        # st.write(f"Score: {sentiment_results[0]['score']}")
else:
    st.write("no input detected")

    