import torch
import logging
import random
import numpy as np
import os

from trainer import Trainer
from ViT import ViT
from ViT_dataset import CoVLADataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.tensorboard as tensorboard

logging.basicConfig(level=logging.INFO)

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

def init_model():
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
    return model

dataset = CoVLADataset()
save_path = "saved_models" # model name: "best_model_{scene_id}.pth"
writer = tensorboard.SummaryWriter(log_dir="logs")

start_scene_id = 0
if not os.path.exists(save_path):
    os.makedirs(save_path)
# if best model exists, load it
models = os.listdir(save_path)
# check if best model exists, starts with "best_model_"
best_model = [model for model in models if model.startswith("best_model_")]
if best_model:
    model = torch.load(os.path.join(save_path, best_model[-1]))
    start_scene_id = int(best_model[-1].split("_")[-1].split(".")[0]) + 1
else:
    model = init_model()
    
# loss function
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

best_test_loss = float('inf')
for scene_id in range(start_scene_id, 10000):
    
    # 示例数据集
    training_data = dataset.get_scene_data(scene_id)

    # 初始化 Trainer
    trainer = Trainer(model, optimizer, criterion, training_data)

    if not scene_id % 10 == 0:
        logging.info(f"Training Scene ID: {scene_id}")

        train_loss = trainer.train_scene()

        val_loss = trainer.validate_scene()

        writer.add_scalar("Loss/train", train_loss, scene_id)
        writer.add_scalar("Loss/val", val_loss, scene_id)
        scheduler.step()

    else:
        logging.info(f"Testing Scene ID: {scene_id}")
        test_loss = trainer.test_scene()
        writer.add_scalar("Loss/test", test_loss, scene_id)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model, os.path.join(save_path, f"best_model_{scene_id}.pth"))
