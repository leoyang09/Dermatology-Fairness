import pandas as pd
import numpy as np

try:
    data = pd.read_pickle('DDI-results/DeepDerm-evaluation.pkl')
    print("Keys:", data.keys())
    
    if 'web_path' in data:
        print("Type of web_path:", type(data['web_path']))
        if isinstance(data['web_path'], (list, np.ndarray)):
            print("Length of web_path:", len(data['web_path']))
            print("First 3 web_paths:", data['web_path'][:3])
        else:
            print("web_path content:", data['web_path'])

    if 'predicted_labels' in data:
        vals = np.array(data['predicted_labels'])
        print("Shape of predicted_labels:", vals.shape)
        print("First 3 predicted_labels:", vals[:3])

    if 'images' in data:
        imgs = data['images']
        print("Type of images:", type(imgs))
        print("Length of images:", len(imgs))
        print("First 3 images:", imgs[:3])

    if 'true_labels' in data:
        t_vals = np.array(data['true_labels'])
        print("Shape of true_labels:", t_vals.shape)
        print("First 3 true_labels:", t_vals[:3])
except Exception as e:
    print("Error:", e)
