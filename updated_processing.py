import os
import numpy as np
from classes import Person

dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
audio_dir = os.path.join(dataset_dir, 'audio')
csv_dir = os.path.join(dataset_dir, 'csv')
gci_dir = os.path.join(dataset_dir, 'gci_estimates')

output_file = os.path.join(os.getcwd(), 'people_dict.npy')

people_dict = {}
for name in os.listdir(audio_dir):
    if name == 'Mark':
        continue
    
    audio_loc = os.path.join(audio_dir, name)
    gci_loc = os.path.join(gci_dir, name)
    
    initials = name[0] + name[-1]
    
    for csv_file in os.listdir(csv_dir):
        if initials == csv_file[-6:-4]:
            csv_loc = os.path.join(csv_dir, csv_file)
            person = Person(name=name,
                            initials=initials,
                            csv_loc=csv_loc,
                            audio_loc=audio_loc,
                            gci_loc=gci_loc)
            people_dict[initials] = person
            
with open(output_file, 'wb') as f:
    np.save(f, people_dict)
                