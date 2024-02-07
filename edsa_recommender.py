"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# Other dependencies
import pickle

with open('resources/models/JL2_SVD.pkl', 'rb') as file:
    model = pickle.load(file)

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", "About","Contact Us", "FAQ", "For The Tech Geeks"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")
                              

    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        # st.write("Describe your winning approach on this page")

        st.subheader("1. Importing Packages")
        st.write("Begin by importing the necessary Python packages, including libraries such as NumPy, Pandas, and scikit-learn, to facilitate data manipulation, analysis, and modeling.")
    
        st.subheader("2. Loading Data")
        st.write("Load the dataset into your working environment, ensuring that it is accessible and ready for analysis. This step may involve reading data from files, databases, or APIs.")

        st.subheader("3. Exploratory Data Analysis (EDA)")
        st.write("Perform a comprehensive EDA to gain insights into the dataset's characteristics.")
        st.write("Visualize and summarize key statistics, identify patterns, and detect outliers to inform subsequent preprocessing steps.")

        st.subheader("4. Preprocessing")
        st.write("Clean and prepare the data for model training.")
        st.write("This may include handling missing values, encoding categorical variables, scaling numerical features, and other tasks to ensure the data is suitable for machine learning algorithms.")

        st.subheader("5. Model Training")
        st.write("Select and train machine learning models on the preprocessed data.")
        st.write("This step involves splitting the dataset into training and testing sets, choosing appropriate algorithms, and fine-tuning model parameters for optimal performance")

        st.subheader("6. Model Evaluation")
        st.write("Assess the trained models' performance using evaluation metrics such as accuracy, precision, recall, and F1 score. Utilize techniques like cross-validation to ensure robust evaluation and avoid overfitting.")

        st.subheader("7. Best Model Selection")
        st.write("Identify the model that performs best based on the evaluation results.")
        st.write("Consider factors like accuracy, interpretability, and computational efficiency when choosing the final model for deployment.")

        st.subheader("8. Best Model Explanation")
        st.write("Provide an explanation for why the selected model is considered the best.")
        st.write("This may involve interpreting feature importance, visualizing decision boundaries, or explaining how the model captures patterns in the data.")

        st.subheader("9. Conclusion")
        st.write("Summarize the key findings, insights, and outcomes of the entire process.")
        st.write("Reflect on the success of the chosen model and its potential applications.")
        st.write("Consider any limitations and suggest future improvements or areas of exploration.")


    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
        
    if page_selection == "About":
        st.title("About Intelliscape Analytics")
        st.write("### Intelligence behind film production landscapes")
        st.write("With our intelligent algorithms, Netflix users can enjoy a personalized entertainment experience. No more sifting through endless options – our technology guides viewers to discover hidden gems, enhancing daily content consumption with a tailored and thoroughly satisfying impact.")
        st.image('resources/imgs/about.png',use_column_width=True)

    if page_selection == "Contact Us":
        st.title("For further assistance, contact us at:")
        st.write('<i class="fas fa-phone"></i> +27 11 000 5555', unsafe_allow_html=True)
        st.write('<i class="fas fa-envelope"></i> hello@intelliscapeanalytics.com', unsafe_allow_html=True)

        st.title("Get In Touch")
        name = st.text_input("Name")
        email = st.text_input("Email Address")
        phone_number = st.text_input("Phone Number")
        comment = st.text_input("Comment or Message")
        submitted = st.button("Submit")

        if submitted:
            st.write(f"Name: {name}")
            st.write(f"Email Address: {email}")
            st.write(f"Phone Number: {phone_number}")
            st.write(f"Comment or Message: {comment}")


    if page_selection == "FAQ":
        st.title("Intelliscape Analytics FAQ")
        st.header("General Questions")

        st.subheader("Q: What is the Intelliscape Analytics app?")
        st.write("A: TheIntelliscape Analytics app is designed to make recommendations tailored to each user’s needs and preferences,providing a better viewing experience.")

        st.subheader("Q: How can I download the app?")
        st.write("A: You can download the Intelliscape Analytics app from AppStore, GooglePlayStore.")

        st.subheader("Q: Is the app free to use?")
        st.write("A: Yes, theIntelliscape Analytics app is free to download and use.")

        st.header("Account and Security")

        st.subheader("Q: How do I create an account?")
        st.write("A: To create an account, simply download the app from your favourite app downloader and all you'll need to provide is your email address to set up an account.")

        st.subheader("Q: How can I reset my password?")
        st.write("A: Head to the Login Page click on 'Reset Password' and you will be prompted to enter 'username' or 'email address'.")
        st.write("After submitting your email or username, check your email inbox for a message from Intelliscape Analytics")
        st.write("This email will contain instructions and a link to reset your password.")


        st.header("Technical Support")

        st.subheader("Q: I'm experiencing technical issues with the app. What should I do?")
        st.write("A: If you're facing technical issues, please reach out to our technical support team using any of our contact details and immediate assistance will be provided.")

        st.header("Submit a Question")

        st.write("If you have a question that is not addressed in the FAQ, feel free to submit it using the form below.")

        new_question = st.text_area("Your Question:")
        submit_button = st.button("Submit Question")

        if submit_button:
            print("New question submitted:", new_question)
            st.success(f"Thank you for your question! We will get back to you as soon as possible.")
    
    if page_selection == "For The Tech Geeks":
        st.title("FOR THE TECH ENTHUSIASTS")
        st.write("### WHY USE SVD MODEL FOR INTELLISCAPE ANALYTICS APP???")
        st.image('resources/imgs/bar_graph.png',use_column_width=True)

        st.subheader("Lowest RMSE")
        st.write("The primary reason for selecting the Singular Value Decompositon (SVD)  model is that it has the lowest RMSE among the models being compared.")
        st.write("RMSE is a measure of the accuracy of a predictive model, and lower values indicate better predictive performance.")
        st.write("In this comparison, Singular Value Decompositon (SVD)  provides the most accurate predictions on the test data.")
        
        st.subheader("Robustness and Stability")
        st.write("The Singular Value Decompositon (SVD)  model is less prone to overfitting and can be more robust in scenarios with limited data.")
        st.write("Singular Value Decompositon (SVD) , by focusing on baseline user and item biases, tends to be more stable and less sensitive to outliers.")
        
        st.subheader("Simplicity and Interpretability")
        st.write("Simplicity is often desirable as it leads to easier interpretation and maintenance.")
        st.write("Singular Value Decompositon (SVD)  captures user and item biases, providing a straightforward understanding of how predictions are made.")

        st.subheader("Computational Efficiency")
        st.write("This efficiency can be crucial, especially for large-scale recommendation systems like Netflix, where the goal is to make real-time or near-real-time recommendations to a massive user base")
        
        st.subheader("Ease of Integration")
        st.write("The simplicity and interpretability of the Singular Value Decompositon (SVD)  model make it easier to integrate into existing recommendation systems.")
        st.write("It may be more straightforward to implement and maintain, allowing for a smoother integration into Netflix's existing infrastructure")


if __name__ == '__main__':
    main()
