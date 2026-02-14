#%%
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
# %%
dataset = LeRobotDataset("lerobot/libero_10")
# %%
for k in dataset[0].keys():
    print(k)
# %%
t2i = lambda t: Image.fromarray(
    (256*t.to("cpu").numpy().transpose(1, 2, 0)).astype(np.uint8))
t2i(dataset[0]['observation.images.image'])
# %%
keep_keys = ['frame_index', 'episode_index', 'index', 'task_index', 'task']
metadata = []
for sample in tqdm(dataset):
    metadata += [{k: sample[k].item() if torch.is_tensor(sample[k]) else sample[k] for k in keep_keys}]
metadata = pd.DataFrame(metadata)
metadata
# %%
# %%
metadata.to_csv("cache/metadata/lerobot__libero_10.csv", index=False)
# %%
# metadata['task'].unique()
metadata.groupby(['task', 'episode_index']).count()