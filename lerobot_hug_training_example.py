#%%
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import os, sys
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# %%
dataset = LeRobotDataset("lerobot/libero_10")
# %%
for k in dataset[0].keys():
    print(k)
# %%
t2i = lambda t: Image.fromarray(
    (256*t.to("cpu").numpy().transpose(1, 2, 0)).astype(np.uint8))
t2i(dataset[0]['observation.images.image'])
#%%
metadata_path = "cache/metadata/lerobot__libero_10.csv"
# %%
if not os.path.exists(metadata_path):
    keep_keys = ['frame_index', 'episode_index', 'index', 'task_index', 'task']
    metadata = []
    for sample in tqdm(dataset):
        metadata += [{k: sample[k].item() if torch.is_tensor(sample[k]) else sample[k] for k in keep_keys}]
    metadata = pd.DataFrame(metadata)
    metadata


    metadata.to_csv("cache/metadata/lerobot__libero_10.csv", index=False)
else:
    metadata = pd.read_csv(metadata_path)
metadata.head()
# %%
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)
# %%
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

# %%
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": t2i(dataset[10]['observation.images.image']).convert("RGB"),
                "image": t2i(dataset[39800]['observation.images.image']).convert("RGB"),
            },
            {"type": "text", "text": "Is claw of the robot open or closed? Answer in one word: 'open' or 'closed'."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
# %%
t2i(dataset[39800]['observation.images.image']).convert("RGB")
# %%
task0 = metadata[metadata['task_index'] == 0]
ep0 = task0[task0['episode_index'] == 0]
ep0
# %%
limit_per_task = 10
images_open = []
images_closed = []
for task_index in metadata['task_index'].unique():
    task0 = metadata[metadata['task_index'] == task_index]
    print(task0['task'].iloc[0])
    counter = 0

    open_ls = []
    closed_ls = []
    episodes = task0['episode_index'].unique()
    for ep_index in episodes[:10]:
        ep0 = task0[task0['episode_index'] == ep_index]

        for didx in tqdm(ep0.index):
            claw_state = dataset[didx]['action'][-1]
            if claw_state == -1:
                open_ls += [didx]
            elif claw_state == 1:
                closed_ls += [didx]

    images_open += np.random.choice(open_ls, 100).tolist()
    images_closed += np.random.choice(closed_ls, 100).tolist()

        # nframes = len(ep0)
        # # print(task_index, ep_index, len(ep0))
        # images_open += [ep0.iloc[int(nframes*0)]['index'].item()]
        # images_closed += [ep0.iloc[int(nframes*0.9)]['index'].item()]
        # counter += 1
        # if counter == limit_per_task:
        #     break
len(images_open), len(images_closed)
# %%
tiled = torch.cat([
    torch.cat([dataset[images_closed[np.random.choice(1000)]]['observation.images.image'] for i in range(10)], dim=2),
    torch.cat([dataset[images_open[np.random.choice(1000)]]['observation.images.image'] for i in range(10)], dim=2)
], dim=1)
t2i(tiled)
# %%

class RobotHoldingDataset(Dataset):

    def __init__(self, images_open, images_closed, dataset, processor):
        prompt = "Is the robot holding onto an object? Answer in one word: yes or no"
        self.return_labels = False
        self.samples = []
        for img_idx in images_open:
            image = t2i(dataset[img_idx]['observation.images.image']).convert("RGB")
            self.samples.append({
                "image": image,
                "prompt": prompt,
                "label": 0,  # no
            })
        for img_idx in images_closed:
            image = t2i(dataset[img_idx]['observation.images.image']).convert("RGB")
            self.samples.append({
                "image": image,
                "prompt": prompt,
                "label": 1,  # yes
            })
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["prompt"]},
                ],
            }
        ]
        proc = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        yes_token = self.processor.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_token = self.processor.tokenizer.encode("no", add_special_tokens=False)[0]
        label_token = yes_token if sample["label"] == 1 else no_token

        # Build labels so that only the last position (next token after prompt) is evaluated
        labels = torch.full_like(proc["input_ids"], fill_value=-100)
        labels[0, -1] = label_token



        data = {
            "input_ids": proc["input_ids"].squeeze(0),
            "attention_mask": proc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "pixel_values": proc.get("pixel_values", None).squeeze(0),
            "image_grid_thw": proc.get("image_grid_thw", None).squeeze(0),
        }
        if self.return_labels:
            data['_label'] = sample["label"]
        return data

def data_collator(features):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [f["input_ids"] for f in features], batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [f["attention_mask"] for f in features], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [f["labels"] for f in features], batch_first=True, padding_value=-100
    )
    if features[0]["pixel_values"] is not None:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        # image_grid_thw: (num_images, 3) per sample; stack to (batch_size, 3)
        image_grid_thw = torch.stack([f["image_grid_thw"] for f in features])
    else:
        pixel_values = None
        image_grid_thw = None

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }
    if image_grid_thw is not None:
        result["image_grid_thw"] = image_grid_thw
    return result
#%%
# Prepare dataset and dataloader
neval = 100
train_dataset = RobotHoldingDataset(images_open[:-neval], images_closed[:-neval], dataset, processor)
eval_dataset = RobotHoldingDataset(images_open[-neval:], images_closed[-neval:], dataset, processor)

# Define training args
training_args = TrainingArguments(
    output_dir="saved/task0-qwen3vl-finetuned",
    per_device_train_batch_size=16,
    num_train_epochs=1,
    learning_rate=1e-6,
    weight_decay=0.01,
    save_strategy="epoch",
    bf16=torch.cuda.is_available(),
    logging_steps=10,
    report_to="none",
)
#%%
loss_history = []
eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False, collate_fn=data_collator)
for epochs in range(3):

    model.eval()

    labels = []
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            # Move tensors to device
            batch = { k: t.to(model.device) for k, t in batch.items() }
            outputs = model(**batch)

            # For language models, prediction is typically obtained via argmax or directly decoding logits
            predictions = torch.argmax(outputs.logits[:, -1:], dim=-1)

            for j in range(batch['input_ids'].size(0)):
                input_text = processor.tokenizer.decode(batch['input_ids'][j], skip_special_tokens=True)
                pred_text = processor.tokenizer.decode(predictions[j], skip_special_tokens=True)
                labels += [batch['labels'][j, -1].item()]
                preds += [pred_text]

    pairs = [('yes' if y == 9693 else 'no', yh) for y, yh in zip(labels, preds) if yh in {'yes','no'}]
    labels = ['yes', 'no']
    cm = confusion_matrix(*zip(*pairs), labels=labels)
    print(cm)

    if epochs == 2:
        break
    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

    history = pd.DataFrame(trainer.state.log_history)
    loss_history += history['loss'].values.tolist()
# %%
plt.figure(figsize=(5, 3))
plt.title('Loss (last token "yes" or "no")')
plt.plot(loss_history)
plt.xlabel('Epochs')
plt.show()
# %%
