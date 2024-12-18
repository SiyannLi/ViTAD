
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from ViT import ViT

logging.basicConfig(level=logging.INFO)
# Training settings
batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)
logging.info(f"Using device {device}")
logging.info(f"Using seed {seed}")

# Data loading
all_dataset = MyDataset("./CoVLA_dataset.pkl")
logging.info(f"Dataset size: {len(all_dataset)}")
train_size = int(0.7 * len(all_dataset))
val_size = int(0.15 * len(all_dataset))

train_dataset = all_dataset[:train_size]
valid_dataset = all_dataset[train_size:train_size+val_size]
test_dataset = all_dataset[train_size+val_size:]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
logging.info("Data loaded successfully")
# Model

model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 4,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# Training

for epoch in range(epochs):
    epoch_loss = 0
    best_val_loss = float("inf")
    for image, (current_state, target_state) in tqdm(train_loader):
        image = image.to(device)
        current_state = current_state.to(device)
        target_state = target_state.to(device)


        output = model(image, current_state)
        loss = criterion(output, target_state)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_loss = 0
        for image, (current_state, target_state) in tqdm(valid_loader):
            image = image.to(device)
            current_state = current_state.to(device)
            target_state = target_state.to(device)

            val_output = model(image, current_state)
            val_loss = criterion(val_output, target_state)


            epoch_val_loss += val_loss / len(valid_loader)

        # save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_model.pth")

    logging.info(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - val_loss : {epoch_val_loss:.4f}\n"
    )

    scheduler.step()

# Testing

with torch.no_grad():
    epoch_test_accuracy = 0
    epoch_test_loss = 0
    for image, (current_state, target_state) in tqdm(test_loader):
        image = image.to(device)
        current_state = current_state.to(device)
        target_state = target_state.to(device)

        test_output = model(image, current_state)
        test_loss = criterion(test_output, target_state)

        epoch_test_loss += test_loss / len(test_loader)

    logging.info(
        f"Test loss : {epoch_test_loss:.4f} - Test acc: {epoch_test_accuracy:.4f}\n"
    )
