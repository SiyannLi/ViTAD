import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging


class Trainer:

    def __init__(self, model, optimizer, criterion, dataset, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.dataset = dataset
        self.save_folder = "saved_models"

        # 配置日志
        logging.basicConfig(level=logging.INFO)

        # 划分数据集
        train_part = int(0.8 * len(self.dataset))
        val_part = int(0.2 * len(self.dataset))
        train_set = self.dataset[:train_part]
        val_set = self.dataset[train_part:]

        test_set = self.dataset

        # 创建 DataLoader
        self.train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    def train_scene(self):
        self.model.train()  # 设置模型为训练模式
        epoch_loss = 0
        for image, current_state, target_state in tqdm(self.train_loader, desc="Training"):
            image = image.to(self.device)
            current_state = current_state.to(self.device)
            target_state = target_state.to(self.device)

            # 前向传播
            output = self.model(image, current_state)
            loss = self.criterion(output, target_state)

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        # 返回平均损失
        return epoch_loss / len(self.train_loader)

    def validate_scene(self):
        self.model.eval()  # 设置模型为评估模式
        epoch_loss = 0
        with torch.no_grad():
            for image, current_state, target_state in tqdm(self.val_loader, desc="Validating"):
                image = image.to(self.device)
                current_state = current_state.to(self.device)
                target_state = target_state.to(self.device)

                # 前向传播
                output = self.model(image, current_state)
                loss = self.criterion(output, target_state)

                epoch_loss += loss.item()

        # 返回平均损失
        return epoch_loss / len(self.val_loader)

    def test_scene(self):
        self.model.eval()  # 设置模型为评估模式
        epoch_loss = 0
        with torch.no_grad():
            for image, current_state, target_state in tqdm(self.test_loader, desc="Testing"):
                image = image.to(self.device)
                current_state = current_state.to(self.device)
                target_state = target_state.to(self.device)

                # 前向传播
                output = self.model(image, current_state)
                loss = self.criterion(output, target_state)

                epoch_loss += loss.item()

        # 返回平均损失
        return epoch_loss / len(self.test_loader)


