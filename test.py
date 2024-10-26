import os
import pickle
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm
import librosa

def extract_mfcc(full_audio_path):
    y, sr = librosa.load(full_audio_path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    return mfcc_features.T  # Transpose to get time x features

def build_test_dataset(dir):
    file_list = [f for f in os.listdir(dir) if os.path.splitext(f)[1] == '.wav']
    test_dataset = {}

    for file_name in file_list:
        label = file_name.split('.')[0].split('_')[1]  # Assuming filename format: "something_label.wav"
        feature = extract_mfcc(os.path.join(dir, file_name))  # Full path
        
        test_dataset[label] = feature

    return test_dataset

def test_gmm_hmm(test_dataset, hmm_models):
    score_cnt = 0
    for true_label, features in test_dataset.items():
        score_list = {}
        
        for model_label, model in hmm_models.items():
            score = model.score(features)  # Use the HMM model to score the features
            score_list[model_label] = score
        
        predicted_label = max(score_list, key=score_list.get)  # Get the label with the highest score
        print(f"True label: {true_label}, Predicted label: {predicted_label}")
        
        if predicted_label == true_label:
            score_cnt += 1

    print(f"Final recognition rate: {100.0 * score_cnt / len(test_dataset)}%")

if __name__ == '__main__':
    test_dir = './test_audio/'  # Directory containing test audio files
    test_data_set = build_test_dataset(test_dir)
    print("Finished preparing the test data.")

    # Load the trained models
    with open("gmm_hmm_models.pkl", "rb") as f:
        hmm_models = pickle.load(f)

    print("Loaded models from gmm_hmm_models.pkl")

    # Test the models
    test_gmm_hmm(test_data_set, hmm_models)


