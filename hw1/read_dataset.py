import gzip
import numpy as np

def read_mnist_images(file_path)->np.ndarray:
    with gzip.open(file_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')
        
        print(f"Magic: {magic}, Images: {num_images}, Size: {rows}x{cols}")
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(num_images, rows, cols)
        return images

def read_mnist_labels(file_path)->np.ndarray:
    with gzip.open(file_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        
        print(f"Magic: {magic}, Labels: {num_labels}")
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels
