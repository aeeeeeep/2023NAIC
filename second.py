import os
import glob
import numpy as np
from submit_compress import compress
from submit_reconstruct import reconstruct

def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def read_fea(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        fea = np.frombuffer(f.read(), dtype='<f4')
    return fea

def evaluate(byte_rate: str):
    query_fea_dir = 'datas/testX'
    reconstructed_query_fea_dir = 'reconstructed_feature_stream/{}'.format(byte_rate)

    query_fea_paths = glob.glob(os.path.join(query_fea_dir, '*.*'))
    reconstructed_query_fea_paths = glob.glob(os.path.join(reconstructed_query_fea_dir, '*.*'))

    distances = []
    for query_fea_path, reconstructed_query_fea_path in zip(query_fea_paths, reconstructed_query_fea_paths):
        query_basename = get_file_basename(query_fea_path)
        query_fea = read_fea(query_fea_path)
        reconstructed_fea = read_fea(reconstructed_query_fea_path)
        distance = np.linalg.norm(query_fea - reconstructed_fea)
        distances.append(distance)
    distances = np.array(distances).mean()
    return distances

scores = []
for byte in ['4000', '8000', '16000']:
    compress('./datas/testX.zip', byte)
    reconstruct(byte)
    score = evaluate(byte)
    scores.append(score)
score = np.array(scores).mean()
print(score)
