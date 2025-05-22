import torch
import matplotlib.pyplot as plt
from scipy.ndimage import label
import numpy as np

def plot_similarity(similarity_matrix: torch.Tensor, save_path: str):

    similarity_matrix = similarity_matrix.detach().cpu().numpy()
    

    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    
    vmin, vmax = similarity_matrix.min(), similarity_matrix.max()
    plt.clim(vmin, vmax)
    
    plt.colorbar()  
    plt.title(f'Similarity Matrix for 0th Batch min:{vmin} max:{vmax}')
    plt.xlabel('N')
    plt.ylabel('N')

    plt.savefig(save_path)
    plt.close()  