import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow as tf

# Function to read data from CSV files
def read_data(details_csv_path, images_csv_path):
    details_df = pd.read_csv(details_csv_path)
    images_df = pd.read_csv(images_csv_path)
    return details_df, images_df

def display_images_with_products(details_df, images_df, items_per_page=54, num_columns=3):
    num_images = len(images_df)
    num_pages = (num_images - 1) // items_per_page + 1

    st.markdown('<p style="font-weight: bold; color: red;">Enter a Page Number:</p>', unsafe_allow_html=True)
    page_number = st.number_input('', min_value=1, max_value=num_pages, value=1)

    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, num_images)

    rows = (end_idx - start_idx + num_columns - 1) // num_columns

    selected_image = st.session_state.get('selected_image', None)

    for row in range(rows):
        cols = st.columns(num_columns)
        for col_idx, col in enumerate(cols):
            idx = start_idx + row * num_columns + col_idx
            if idx < end_idx:
                link = images_df.loc[idx, 'link']
                product_details = details_df.iloc[idx]

                with col:
                    st.markdown(f"""
                        <div class="product-image-box">
                            <img src="{link}" width="200" style="border-radius: 8px;">
                            <p style="margin-top: 5px; font-weight: bold; text-align: center;">{product_details['productDisplayName']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    if col.button('Select', key=f'select_{idx}'):
                        st.session_state.selected_image = images_df.loc[idx, 'filename']
                        st.session_state.page = "Nearest Image"
                        st.success(f'Selected {product_details["productDisplayName"]}')
                        st.experimental_rerun()
        st.markdown("<hr style='margin: 10px 0;font-size:20px;border-top: 1px solid #ccc;'>", unsafe_allow_html=True)

# Load CSV files

details_csv_path = 'D:\\ML\\project\\fashion recommendation system\\data\\details.csv'
images_csv_path = 'D:\\ML\\project\\fashion recommendation system\\data\\images.csv'

details_df, images_df = read_data(details_csv_path, images_csv_path)

# Streamlit app - Homepage with All Products
def homepage():
    st.markdown('<div class="product-details-box"> <h2>Products </h2> </div>', unsafe_allow_html=True)

    display_images_with_products(details_df, images_df)

# Function to load embeddings and filenames
def load_data(embeddings_path, filenames_path):
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    with open(filenames_path, 'rb') as f:
        filenames = pickle.load(f)

    return embeddings, filenames

# Function to extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    features = features.flatten()
    norm_features = features / np.linalg.norm(features)
    return norm_features

# Function to load ResNet50 model
def load_pretrained_model(input_shape=(224, 224, 3)):
    base_model = Sequential()
    base_model.add(tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape))
    base_model.add(GlobalMaxPooling2D())
    return base_model

def display_image_with_details(image_path, details_df, images_df, title="Image"):
    image_name = os.path.basename(image_path)
    image_link = images_df[images_df['filename'] == image_name]['link'].values[0]
    product_display_name = details_df[images_df['filename'] == image_name]['productDisplayName'].values[0]
    st.markdown(f'<div class="product-details-box"> <h2>{title} </h2> </div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
            <div class="product-image-box">
                <img src="{image_link}" width="200" style="border-radius: 8px;">
                <p style="margin-top: 5px; font-weight: bold;">{product_display_name}</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        image_details = details_df[images_df['filename'] == image_name]
        if not image_details.empty:
            st.markdown('<div class="product-details-box"> <h2>Product Details </h2> </div>', unsafe_allow_html=True)
            for col_name, col_value in image_details.iloc[0].items():
                st.markdown(f'<div class="product-details-box"> <p><strong>{col_name}:</strong> {col_value}</p> </div>', unsafe_allow_html=True)

        else:
            st.warning(f"No details found for {image_name}.")

# Function to display nearest images and their details
def display_nearest_images_with_details(nearest_image_paths, details_df, images_df):
    st.markdown('<div class="product-details-box"> <h1>Similar Product Details: </h1> </div>', unsafe_allow_html=True)

    for image_path in nearest_image_paths:
        image_name = os.path.basename(image_path)
        image_link = images_df[images_df['filename'] == image_name]['link'].values[0]
        product_display_name = details_df[images_df['filename'] == image_name]['productDisplayName'].values[0]
        display_image_with_details(image_path, details_df, images_df, title=product_display_name)

# Load embeddings and filenames
embeddings_path = "D:\\ML\\project\\embeddings.pkl"
filenames_path = 'D:\\ML\\project\\filenames.pkl'

embeddings, filenames = load_data(embeddings_path, filenames_path)

# Convert embeddings to numpy array
embeddings = np.array(embeddings)

# Fit NearestNeighbors model
neighbors_model = NearestNeighbors(n_neighbors=5, algorithm='brute',metric='euclidean').fit(embeddings)

# Load ResNet50 model
model = load_pretrained_model()

# Streamlit app - Nearest Image Page
def nearest_image_page():
    if 'selected_image' not in st.session_state:
        st.warning('No image selected. Please go to the homepage and select an image.')
        return

    selected_image = st.session_state.selected_image

    try:
        # Read details.csv and images.csv
        details_csv_path = 'D:\\ML\\project\\fashion recommendation system\\data\\details.csv'
        images_csv_path = 'D:\\ML\\project\\fashion recommendation system\\data\\images.csv'

        # Read details.csv and filter out unwanted columns
        details_df = pd.read_csv(details_csv_path, usecols=lambda x: x not in ['id','Unnamed: 10', 'Unnamed: 11'])
        images_df = pd.read_csv(images_csv_path)

        # Example: Display selected image and its details
# fashion.py

        selected_image_path = os.path.join('D:\\ML\\project\\fashion recommendation system\\data\\images', selected_image)
        display_image_with_details(selected_image_path, details_df, images_df, title="Selected Image")

        # Extract features for selected image
        input_features = extract_features(selected_image_path, model)

        # Find nearest images
        distances, indices = neighbors_model.kneighbors([input_features])
        nearest_image_paths = [filenames[idx] for idx in indices[0] if idx < len(filenames)]

        # Display nearest images and their details
        display_nearest_images_with_details(nearest_image_paths, details_df, images_df)

        # Back button to return to homepage
        if st.button("Back"):
            st.session_state.page = "Home"
            st.experimental_rerun()

    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {e}")

def add_custom_css():
    st.markdown("""
        <style>
        /* Custom CSS for the app */
        body {
            background-color: #2E2E2E;  /* Dark background */
            color: #F0F0F0;
            background-image: url('https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3Vwd2s2MTg0MzExOC13aWtpbWVkaWEtaW1hZ2Uta293YzVmbDcuanBn.jpg');
            background-size: cover;  /* Cover the entire background */
            background-repeat: no-repeat;  /* Do not repeat the background image */
            background-attachment: fixed;  /* Make the background image fixed */
            height: 100vh;  /* Full height */
            background-position: center;  /* Center the background image */
        }
        .stApp {
            background-color: transparent;
        }
        .stButton>button {
            background-color: #4CAF50;  /* Green button background */
            color: #F0F0F0;  /* Light text color */
            border: none;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
            border-radius: 12px;  /* Rounded corners */
        }
        .stButton>button:hover {
            background-color: #45a049;  /* Darker green on hover */
        }

        .product-image-box {
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white background */
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Shadow effect */
            text-align: center;
            margin-bottom: 20px;
        }
        .product-details-box {
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white background */
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Shadow effect */
            margin-bottom: 20px;
        }
        .product-details-box p {
            color: #000000; /* Adjust text color for better visibility */
        }
                
        </style>
    """, unsafe_allow_html=True)
st.set_page_config(page_title="Fashion Recommendation System", page_icon="👗")

# Main Streamlit app
def main():
    add_custom_css()
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    # Add a header and footer
    st.markdown("<p style='text-align: center;font-size:45px; color:red;background-color: rgba(255, 255, 255, 0.8);padding: 10px; border-radius: 10px;box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);margin-bottom: 20px;'>Fashion Recommendation System</p>", unsafe_allow_html=True)
    pages = {
        "Home": homepage,
        "Nearest Image": nearest_image_page,
    }

    page = st.sidebar.selectbox("Select Page", tuple(pages.keys()), index=list(pages.keys()).index(st.session_state.page))

    if st.session_state.page != page:
        st.session_state.page = page

    pages[page]()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("<p style='text-align: center;'>Made with ❤️ by Divya Teja Venkat</p>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style='text-align: center;'>
       <a href='https://www.linkedin.com/in/divya-teja-venkat-kaliboyina-b023a2219/' target='_blank'>LinkedIn</a> | 
       <a href='https://github.com/DivyaTejaVenkat29/' target='_blank'>GitHub</a> | 
       <a href='mailto:divyatejavenkatk@gmail.com'>Email</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

