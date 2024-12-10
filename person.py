import pandas as pd
import librosa
import os
import numpy as np
from typing import Dict
from lpc import LPC, MagsAndPeaks

SAMPLE_RATE = 44100

class Person:
    name: str
    initials: str
    gender: str
    ground_truth: pd.DataFrame
    audio_dict: Dict[str, np.array]
    def __init__(self, name: str, initials: str, csv_loc: str, audio_loc: str, gender: str = None):
        self.name = name
        self.initials = initials
        self.ground_truth = pd.read_csv(csv_loc)
        self.audio_dict = self.get_audio_dict(audio_loc)
        self.gender = gender if gender is not None else self.set_gender_from_df()
    
    def get_audio_dict(self, audio_loc):
        audio_dict = {}
        for file in os.listdir(audio_loc):
            file = os.path.join(audio_loc, file)
            audio, sr = librosa.load(file, sr=SAMPLE_RATE)
            assert sr == SAMPLE_RATE
            vowel = file[-6:-4]
            audio_dict[vowel] = audio
        return audio_dict
    
    def set_gender_from_df(self):
        gender_row = self.ground_truth.iloc[0]
        initials = self.initials
        return gender_row[initials]
    
