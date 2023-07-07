from torchvision import transforms
from typing import Optional
from tqdm import tqdm
import h5py
import torch
from pathlib import Path
import numpy as np

from .dataset import OnlineTripletImageDataset

class EmbeddingRetriever:
    def __init__(self, dataset: OnlineTripletImageDataset,
                path: Optional[Path] = None,
                model: Optional[torch.nn.Module] = None,
                device = torch.device("cpu")):
        self.device = device
        self.dataset = dataset
        self.model = model
        self.path = path

        self.names = []
        self.embeddings = []
        if self.path is not None:
            print("Loading saved embedding from: ", str(self.path))
            self.names, self.embeddings = self.load_embedding()
    
    def export_embeddings(self):
        if self.path is None:
            self.path = Path('netvlads.h5')
        self.path.parent.mkdir(exist_ok=True, parents = True)
        if (len(self.names) == 0) or (len(self.embeddings) == 0) or (len(self.names) == len(self.embeddings)):
            self._calculate_embeddings(self.model) 
        with h5py.File(str(self.path), "a", libver="latest") as f:
            for i, name  in enumerate(self.names):
                try:
                    if name in f:
                        del f[name]
                    f[name] = self.embeddings[i]
                except OSError as error:
                    if 'No space left on device' in error.args[0]:
                        del f[name]
                    raise error
    
    def load_embeddings(self):
        embeddings = []
        with h5py.File(str(self.path), 'r', libver="latest") as r:
            names = self._get_dataset_keys(r)
            for name in names:
                embeddings.append(r[name][()])
            embeddings = np.array(embeddings)
        return names, embeddings

    def _calculate_embedding(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        embedding = self.model(img_tensor)
        return embedding.view(-1).detach().cpu().numpy()

    def _calculate_embeddings(self):
        if self.model == None:
            raise ValueError("No model found!!")

        for i, (img, label) in enumerate(tqdm(self.dataset)):
            full_path_name = str(self.dataset.root/self.dataset.names[i])
            v = self._calculate_embedding(img.unsqueeze(0))
            self.names.append(full_path_name)
            self.embeddings.append(v)
        self.embeddings = np.array(self.embeddings)
    
    def _get_dataset_keys(self, f):
        keys = []
        f.visit(lambda key : keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
        return keys
