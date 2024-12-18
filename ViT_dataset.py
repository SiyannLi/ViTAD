import json
from datasets import load_dataset
import numpy as np
from torchvision import transforms
import torch
class CoVLADataset:

    def __init__(self):
        self.dataset_video = load_dataset(
            "turing-motors/CoVLA-Dataset",
            cache_dir="/home/avsaw1/Documents/siyan/unsloth/data/CoVLA",
            data_dir="videos",
            split="train",
        )

        self.dataset_state = load_dataset(
            "turing-motors/CoVLA-Dataset",
            cache_dir="/home/avsaw1/Documents/siyan/unsloth/data/CoVLA",
            data_files="states.tar.gz",
            split="train",
        )

    def get_scene_data(self, scene_id):

        # Extract video frames as PIL images
        transform = transforms.Resize((256, 256))

        frames = [
            transform(torch.from_numpy(
                self.dataset_video[scene_id]["video"]
                .get_batch([x]).asnumpy()[0]
                .transpose(2, 0, 1)))
                .to(torch.float32)
            for x in range(600)
        ]

        # Parse states into a list of dictionaries
        states = [
            json.loads(self.dataset_state[scene_id]["jsonl"].splitlines()[x]) for x in range(600)
        ]

        # Function to fetch data for a single frame
        def get_data(frame_id):
            return (
                frames[frame_id],
                torch.tensor(
                [round(states[frame_id][str(frame_id)]["vEgo"], 2),
                round(states[frame_id][str(frame_id)]["steeringAngleDeg"], 2),
                round(states[frame_id][str(frame_id)]["brake"], 2),
                round(states[frame_id][str(frame_id)]["gas"], 2)], dtype=torch.float32),
                torch.tensor(
                [round(states[frame_id + 1][str(frame_id + 1)]["vEgo"], 2),
                round(states[frame_id + 1][str(frame_id + 1)]["steeringAngleDeg"], 2),
                round(states[frame_id + 1][str(frame_id + 1)]["brake"], 2),
                round(states[frame_id + 1][str(frame_id + 1)]["gas"], 2)], dtype=torch.float32)
            )
            
        # Construct the conversation dictionary
        conversation = [
            get_data(x)
        for x in range(599)]

        return conversation

