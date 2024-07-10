# Fashion Recommendation System

## Table of Contents
**1. [Abstract](#abstract)**  
**2. [Introduction](#introduction)**  
**3. [Problem Statement](#problem-statement)**  
**4. [Motivation](#motivation)**  
**5. [Key Objectives](#key-objectives)**  
**6. [Project Setup](#project-setup)**  
**7. [Usage](#usage)**  
**8. [Methodology](#methodology)**  
**9. [Results](#results)**  
**10. [Conclusion](#conclusion)**  
**11. [License](#license)**  

## Abstract
The Fashion Recommendation System is designed to help users discover fashion items based on visual similarities. Utilizing machine learning techniques for image feature extraction and similarity measurement, the system recommends products that are visually similar to a selected item, providing a seamless and intuitive shopping experience.

## Introduction
Fashion recommendation systems are increasingly crucial in online retail, helping users find products that match their preferences. This project leverages advanced neural networks for image processing and recommendation, aiming to enhance the user experience by suggesting similar fashion items based on a chosen image.

## Problem Statement
Online shoppers often face difficulty in finding fashion items that match their taste due to the vast number of available products. Traditional search methods are limited and may not capture the nuances of style and visual appeal. This project aims to address this issue by developing a recommendation system that uses image similarity to suggest relevant fashion items.

## Motivation
The motivation behind this project stems from the need to improve the online shopping experience. By providing recommendations based on image similarity, users can easily discover products that align with their preferences, potentially increasing customer satisfaction and sales for retailers.

## Key Objectives
- **Develop a feature extraction model**: Utilize a pre-trained ResNet50 neural network to extract features from fashion item images.
- **Implement a recommendation algorithm**: Use cosine similarity to find and suggest visually similar fashion items.
- **Create an interactive web interface**: Build a user-friendly interface with Streamlit for users to select images and view recommendations.
- **Optimize performance and accuracy**: Ensure the system provides accurate and relevant recommendations in a timely manner.

## Project Setup
### Prerequisites
- Python 3.9.7
- Streamlit
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/fashion-recommendation-system.git
    cd fashion-recommendation-system
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the necessary data files (`details.csv`, `images.csv`, `embeddings.pkl`, `filenames.pkl`) and place them in the appropriate directories.

## Usage
To run the Streamlit application:
```bash
streamlit run app.py
```
This will start the Streamlit server, and you can interact with the application through your web browser.

## Main Components:

    Homepage: Displays all products with options to select an image.
    Nearest Image Page: Shows the selected image and recommends similar products based on visual similarity.

## Methodology

The project follows these key steps:
    
    1.Data Collection: Gather images of fashion items along with their metadata.
    
    2.Data Preprocessing: Clean and prepare the data, including resizing images and extracting relevant details.
    
    3.Feature Extraction: Use a pre-trained ResNet50 model to extract feature vectors from the images.
    
    4.Similarity Measurement: Implement cosine similarity to measure the similarity between the feature vectors.
    
    5.Model Training: Train a NearestNeighbors model using the extracted features.
    
    6.Web Interface Development: Develop an interactive interface using Streamlit for users to select images and view recommendations.

## Results

The system successfully provides recommendations for fashion items based on visual similarity. The Streamlit interface allows users to easily select an image and view similar products, enhancing the shopping experience.

## Conclusion

The Fashion Recommendation System effectively uses image processing and machine learning techniques to recommend fashion items. By leveraging the pre-trained ResNet50 neural network for feature extraction and cosine similarity for similarity measurement, the system offers accurate and relevant recommendations, demonstrating the potential for improving online shopping experiences.

## License
  This project is licensed under the MIT Licens
