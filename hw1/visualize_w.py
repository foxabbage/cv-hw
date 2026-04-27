import numpy as np
import matplotlib.pyplot as plt

def visualize_w1(path):
    data = np.load(path)
    w0:np.ndarray = data['w_0']
    length = w0.shape[1]

    features = w0.T.reshape(length, 28, 28)
    
    fig, axes = plt.subplots(length//16, 16, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(length):
        img = features[i]
        img_min, img_max = img.min(), img.max()
        if img_max == img_min:
            img_norm = np.zeros_like(img)
        else:
            img_norm = (img - img_min) / (img_max - img_min)
            
        axes[i].imshow(img_norm, cmap='gray')
        axes[i].axis('off')
        
    plt.tight_layout(pad=0.5)
    plt.show()

visualize_w1('./grid_search_results_3/best_model.npz')