import numpy as np
import os
import cv2
import torch
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

class PlantDetector:
     def __init__(self, config_path, model_path):
        # Initialize essentials
        self.logger = Logger(local_rank=0, use_tensorboard=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load config and model
        load_config(cfg, config_path)
        self.model = build_model(cfg.model)
        
        # Safe weight loading
        ckpt = torch.load(model_path, 
                        map_location=self.device,
                        weights_only=True)
        
        load_model_weight(self.model, ckpt, self.logger)
        self.model = self.model.to(self.device).eval()
        
        # Pipeline setup
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

     def inference(self, img_path):
        # Read image with error handling
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # Prepare metadata
        img_info = {
            "id": 0,
            "file_name": os.path.basename(img_path),
            "height": img.shape[0],
            "width": img.shape[1]
        }
        
        meta = {"img_info": img_info, "raw_img": img, "img": img}
        meta = self.pipeline(None, meta, cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        
        # Memory-safe inference
        with torch.no_grad():
            results = self.model.inference(meta)
            
        return meta, results[0]
     
def get_plant_detector(config_path, model_path):
    """
    Initialize the PlantDetector with the given config and model paths.
    
    Args:
        config_path (str): Path to the configuration file.
        model_path (str): Path to the model weights file.
        
    Returns:
        PlantDetector: An instance of the PlantDetector class.
    """
    return PlantDetector(config_path, model_path)