from person import Person
from lpc import LPC, MagsAndPeaks
import os
import numpy as np
import pandas as pd
from IPython.display import display
from tqdm import tqdm


cwd = os.getcwd()
audio_dir = os.path.join(cwd, 'audio')
csv_dir = os.path.join(cwd, 'csv')

def make_persons(audio_dir: str, 
                 csv_dir: str, 
                 dict_loc: str = None):
    """Generate Person objects given audio/vowel measurements

    Args:
        audio_dir (os.path): loc of audio files (formatted 'FirstName_LastInitial')
        csv_dir (os.path): loc of vowel ground truth measurements (formatted 'Initials.csv')
    """
    dict_loc = os.path.join(os.getcwd(), 'people_data.npy')
    if os.path.exists(dict_loc):
        people_dict = np.load(dict_loc, allow_pickle=True)
    
    people_dict = {}
    for audio_folder in tqdm(os.listdir(audio_dir)):
        initials = audio_folder[0] + audio_folder[-1]
        for csv_file in os.listdir(csv_dir):
            if initials == csv_file[-6:-4]:
                csv_file = os.path.join(os.getcwd(), 'csv', csv_file)
                audio_folder = os.path.join(os.getcwd(), 'audio', audio_folder)
                people_dict[initials] = Person(name = audio_folder,
                                               initials=initials,
                                               csv_loc=csv_file,
                                               audio_loc=audio_folder)
          
    # Save People Dictionary for future testing      
    with open(os.path.join(os.getcwd(), 'people_data.npy'), 'wb') as f:
        np.save(f, people_dict)
        
    return people_dict


def main():
    audio_dir = os.path.join(os.getcwd(), 'audio')
    csv_dir = os.path.join(os.getcwd(), 'csv')
    dict_loc = os.path.join(os.getcwd(), 'people_data.npy')
    people_dict = make_persons(audio_dir, csv_dir, dict_loc)
    display(people_dict)

if __name__ == '__main__':
    main()              