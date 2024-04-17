# Question Classification with Metaflow

## Overview

This project focuses on classifying questions into primary categories using the Experimental Data for Question Classification dataset available in the nltk.corpus package. The primary categories are:

- ABBR: Denotes abbreviations
- ENTY: Stands for entities
- DESC: Denotes descriptions and abstract concepts
- HUM: Denotes human beings
- LOC: Denotes locations
- NUM: Stands for numeric values

The classification process is implemented using Metaflow, a framework for building and managing real-life data science projects.

## Project Structure

The project consists of the following components:

1. **Text Classification Flow**: A Metaflow flow defined in `text_classification_flow.py`. This flow handles the end-to-end process of preprocessing the data, performing feature engineering, training a text classification model, and evaluating its performance.

2. **Preprocessing**: Data preprocessing is carried out in the `preprocessing_data` step of the Metaflow flow. It involves tokenizing the text data, encoding labels, and splitting the dataset into training, validation, and test sets.

3. **Feature Engineering**: The `feature_engineering` step involves transforming text data into numerical features using TF-IDF vectorization.

4. **Model Training**: Model training is performed in the `train_model` step. It involves building and training a neural network model using PyTorch, with the Adam optimizer and Cross-Entropy Loss.

5. **Evaluation**: Model evaluation is conducted in the `evaluate` step. Metrics such as accuracy, precision, recall, and F1 score are calculated on the test dataset to assess the model's performance.

## CI/CD Pipeline

The project includes a CI/CD pipeline defined in the GitHub Actions workflow file `.github/workflows/ci_cd.yml`. The pipeline consists of the following jobs:

### 1. Pre-commit

This job runs on every push to the `main` branch. It checks the code using pre-commit hooks to ensure code quality and consistency.

### 2. Terraform

This job runs Terraform to provision a Kubernetes cluster on DigitalOcean. It initializes Terraform and applies the configuration.

### 3. Push to Registry

This job builds a Docker image of the project, tags it with metadata, and pushes it to Docker Hub. It extracts metadata such as tags and labels and uses them to tag the Docker image.

### 4. Deploy

This job deploys the application to a DigitalOcean Kubernetes cluster. It installs `doctl` to interact with DigitalOcean, saves the cluster's kubeconfig, and deploys the application using Kubernetes manifests. In each pod of the Kubernetes cluster, the model is deployed with a FastAPI endpoint for inference. This allows for real-time prediction of question categories.
