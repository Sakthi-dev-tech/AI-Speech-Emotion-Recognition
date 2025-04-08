import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split

"""

Filename identifiers

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).

Vocal channel (01 = speech, 02 = song).

Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).

Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.

Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").

Repetition (01 = 1st repetition, 02 = 2nd repetition).

Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

"""

# Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

def extract_feature(file_name, mfcc, chroma, mel):
    """
    Extracts features from the audio file.
    :param file_name: Path to the audio file
    :param mfcc: Boolean indicating whether to extract MFCC features
    :param chroma: Boolean indicating whether to extract Chroma features
    :param mel: Boolean indicating whether to extract Mel features
    :param contrast: Boolean indicating whether to extract Contrast features
    :param tonnetz: Boolean indicating whether to extract Tonnetz features
    :return: Extracted features as a numpy array
    """
    # Load the audio file
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))

    return result

def load_data(test_size=0.2):
    """
    Loads the dataset and extracts features from the audio files.
    :param test_size: Proportion of the dataset to include in the test split
    :return: Tuple of features and labels
    """
    x, y = [], []
    for file in glob.glob("C:\\Users\\Sakthi\\Documents\\GitHub\AI-Speech-Emotion-Recognition\\speech_samples\\Actor_*\\*.wav"):
        # Extracting features from the audio file
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]

        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=42)


x_train,x_test,y_train,y_test=load_data(test_size=0.25)
print((x_train.shape[0], x_test.shape[0]))