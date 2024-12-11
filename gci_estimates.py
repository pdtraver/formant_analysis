import os
from FCN_GCI import run_prediction

dataset_audio = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'dataset', 'audio')
estimate_folder = os.path.join(os.path.dirname(dataset_audio), 'GCI_estimates')

for folder in os.listdir(dataset_audio):
    person_name = folder
    if person_name == '.DS_Store':
        continue
    
    output_path = os.path.join(estimate_folder, person_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    person_data = os.path.join(dataset_audio, person_name)
    
    # sound_files = []
    # for file in os.listdir(person_data):
    #     sound_files.append(os.path.join(person_data, file))
    
    run_prediction(person_data, output_path)