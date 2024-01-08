
# Rental Property Price Prediction <img src="https://cdn-icons-png.flaticon.com/512/13644/13644020.png" alt="Car Price Prediction" width="50" height="50">


This project provides a comprehensive pipeline for predicting rental property prices based on various features. It encompasses data cleaning, model training, evaluation, and a user-friendly interface for predictions.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Code Layout](#code-layout)
- [Getting Started](#getting-started)
- [Training the Model](#training-the-model)
- [Save the Model](#save-the-model)
- [Prediction](#prediction)
- [Dockerized Web App](#dockerized-web-app)
- [License](#license)

## Overview
The Rental Property Price Prediction project harnesses machine learning to estimate rental property prices accurately. It features meticulous data processing methods and implements a variety of regression models, including Linear Regression, Random Forest, Gradient Boosting, ElasticNet, XGBoost, and a GridSearchCV-based selection. A user-friendly Streamlit web app allows users to input property details and promptly receive real-time price predictions.

## Project Structure
The project structure is organized as follows:

- `data/`: Contains the dataset (`car_data.csv`) used for training and predictions.
- `pipelines/`: Includes ZenML pipelines for data cleaning, model training, and evaluation.
- `steps/`: Custom Python scripts for data loading, model training, and evaluation.
- `src/`: Source code files, including data cleaning strategies, model development, and utilities.
- `saved_models/`: Stores trained machine learning models.
- `utils.py`: Utility functions for model saving and loading.
- `app.py`: Streamlit-based web application for predicting car prices.
- `requirements.txt`: Python dependencies for the project.
- `Dockerfile`: Docker configuration for containerizing the web app.

## Code Layout
![Rental Property Price_layout](https://github.com/iampraveens/House-Rent-Prediction/assets/125688218/b00f24f8-38b0-462d-85af-db2823b0b125)

## Getting Started
To get started with the project, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/iampraveens/House-Rent-Prediction.git
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```
## Training the Model

```bash
python ./steps/run_pipeline.py
```
- This command will execute the data cleaning, model training, and evaluation process

## Save the Model

```bash
python ./steps/save_model.py
```

## Prediction

```bash
streamlit run app.py
```

## Dockerized Web App
You can also deploy the Rental Property Price Prediction web application using Docker. Build the Docker image and run the container:
```bash
docker build -t your_docker_username/rental-property-price-prediction-app .
```
- To build a docker image.

```bash
docker run -d -p 8501:8501 your_docker_username/rental-property-price-prediction-app
```
- To run as a container.

Access the web app at `http://localhost:8501` or `your_ip_address:8501` in your web browser.
Else if you want to access my pre-built container, here is the code below to pull from docker hub(Public).
```bash
docker pull iampraveens/rent-price-app:latest
```
## License 
This project is licensed under the MIT License - see the [License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt) file for details.

This README provides an overview of the project, its structure, how to get started, how to train the model, make predictions, tracking the model and even deploy a Dockerized web app.

