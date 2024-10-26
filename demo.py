import warnings
import os
import pickle
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
import librosa  # Using librosa for MFCC extraction

warnings.filterwarnings('ignore')

def extract_mfcc(full_audio_path):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(full_audio_path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    return mfcc_features.T  # Transpose to get time x features

def build_dataset(dir):
    """Build a dataset of MFCC features from audio files in a directory."""
    file_list = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    
    for file_name in file_list:
        # Assuming filename format: "something_label.wav"
        label = file_name.split('.')[0].split('_')[1]  
        feature = extract_mfcc(os.path.join(dir, file_name))  # Full path
        
        if label not in dataset:
            dataset[label] = []
        dataset[label].append(feature)

    return dataset

def train_gmm_hmm(dataset):
    """Train GMM-HMM models for each label in the dataset."""
    gmm_hmm_models = {}
    states_num = 5  # Number of hidden states

    for label, features in dataset.items():
        # Create the GaussianHMM model
        model = hmm.GaussianHMM(n_components=states_num, covariance_type='diag', n_iter=10)
        
        # Stack features for training and calculate lengths
        lengths = [feature.shape[0] for feature in features]
        train_data = np.vstack(features)  # Stack features for training

        # Train the model
        model.fit(train_data, lengths=lengths)  
        
        # Save the model
        gmm_hmm_models[label] = model

    return gmm_hmm_models

def main():
    train_dir = './train_audio/'
    train_data_set = build_dataset(train_dir)
    print("Finished preparing the training data.")

    hmm_models = train_gmm_hmm(train_data_set)
    print("Finished training the GMM-HMM models.")

    # Save the trained models to a pickle file
    with open("gmm_hmm_models.pkl", "wb") as f:
        pickle.dump(hmm_models, f)

    print("Models saved to gmm_hmm_models.pkl")

if __name__ == '__main__':
    main()
