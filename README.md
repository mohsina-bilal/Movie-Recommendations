# Movie Recommendation System using Neural Collaborative Filtering
This project implements a movie recommendation system using the Item-based Neural Collaborative Filtering (NCF) technique. The code is written in Python and uses PyTorch and TensorFlow libraries for deep learning.

## Project Overview
The project aims to recommend movies to users based on their historical movie ratings.
It uses the MovieLens dataset, which contains user ratings for movies.
The code is divided into several sections, including data preprocessing, model architecture, training, and evaluation.

Item-based neural collaborative filtering has been utilised in order to create a movie recommendation system that generates recommendations based on implicit user feedback. Recommender systems built using implicit feedback also allows us to tailor recommendations in real time, with every click and interaction.

Item-based neural collaborative filtering systems employ neural networks to develop latent representations of objects and users. These systems are founded on the premise that similar items are favoured by similar users, and that a user's preferences may be anticipated by looking at the preferences of other users who like similar movies.

The input to the neural network in an item-based neural collaborative filtering system is a matrix of user-item evaluations. The matrix's rows and columns each represent a user and an item. The matrix entries are the ratings provided to the goods by the users.

The neural network is divided into two layers: the item embedding layer and the user embedding layer. Based on the user-item ratings matrix, the item embedding layer develops a low-dimensional representation of each item. Based on the items that were assessed, the user embedding layer establishes a low-dimensional representation of each user.

The rating prediction model is then built by combining the item embedding layer with the user embedding layer. This model takes the user embedding and item embedding for a particular user and item as input and produces a predicted rating.

The neural network learns the item and user embeddings during training by minimising the gap between the expected and actual ratings in the training data. Typically, a loss function such as mean squared error or binary cross-entropy is used.

Once trained, the neural network may be utilised to give suggestions to users. Given a user, the system may identify and propose goods that are most comparable to the items that the user has rated highly.

Overall, item-based neural collaborative filtering systems are a strong and versatile method to suggest that may be applied in a wide range of applications such as e-commerce, entertainment, and social media. Even in the presence of limited and noisy data, these algorithms can learn complicated patterns in user-item interactions and generate accurate predictions.

![image](https://github.com/mohsina-bilal/Movie-Recommendation-System/assets/99142580/e2f78da7-3f9d-4bf7-aba4-81f1bdd2c4ba)

## Prerequisites
### Before running the code, make sure you have the following dependencies installed:
- Pandas
- NumPy
- tqdm
- Torch
- PyTorch Lightning

### Hardware and Software Requirements
#### Hardware Requirement:
- Processor (CPU): Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz 2.11 GHz
- Graphics Processing Unit (GPU): NVIDIA GTX 1050 Ti or AMD Radeon RX 570 GPU, NVIDIA RTX 2060 or AMD Radeon RX 6700 XT (recommended)
- RAM: 8 GB of RAM (ideally, 16-32 GB RAM)
- Storage: 256 GB (ideally, 512 GB or above)
- Software and Libraries: Python is the primary programming language used for deep learning, and various libraries such as TensorFlow, PyTorch, and Keras are commonly used.
#### Software Requirement:
- Operating System: Windows 7 or later (64-bit)
- GPU Drivers: Compatible NVIDIA GPU with CUDA support, along with the latest NVIDIA drivers for acceleration purposes.
- Python Version: PyTorch supports Python 3.6, 3.7, 3.8, and 3.9.
- CPU Requirements: PyTorch can run on CPUs (significantly slower performance)
- 4 GB of RAM


### Getting Started
- Clone or download the repository.
- Install the required dependencies using the command: pip install -r requirements.txt.
- Download the MovieLens dataset and save it as "ratings.csv" in the project directory.

### Usage
- Open the Python script containing the code.
- Adjust any necessary hyperparameters or settings based on your requirements.
- Run the script to train the NCF model.
- Once the model is trained, it will evaluate the recommendation performance by calculating the Hit Ratio @ 10 metric.
- The results will be displayed in the console.

## Dataset References
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

## Conclusion
This project demonstrates how to implement a movie recommendation system using Neural Collaborative Filtering.
Item-based neural collaborative filtering is a highly effective approach for building movie recommendation systems that can provide personalised recommendations to users based on their preferences and past behaviour. By using deep learning techniques to learn from large data sets, these systems can provide accurate and useful recommendations that can help users discover new movies and improve their movie-watching experience.


