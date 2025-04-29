from model import ViTTiny
import torch
class WeedDetector:
    def __init__(self, model_path):
        # Initialize the model
        self.model = ViTTiny(
            img_size=32,            # Input image size
            patch_size=4,           # Patch size
            in_chans=3,             # Input channels (RGB)
            num_classes=17,         # Number of classes
            embed_dim=128,          # Embedding dimension
            depth=6,                # Number of transformer blocks
            num_heads=2,            # Number of attention heads
            mlp_ratio=4.,           # MLP expansion ratio
            qkv_bias=True,          # Use bias for QKV projection
            drop_rate=0.2,          # Dropout rate
            attn_drop_rate=0.2      # Attention dropout rate
        )
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, batch):
        # Preprocess the batch and make predictions on 17 classes
        id_to_class = {
            0: 'CROP',
            1: 'GRASSES',
            2: 'AMARANTH',
            3: 'GOOSEFOOT',
            4: 'KNOTWEED',
            5: 'CORN SPURRY',
            6: 'CHICKWEED',
            7: 'SOLANALES',
            8: 'POTATO WEED',
            9: 'CHAMOMILE',
            10: 'THISTLE',
            11: 'MERCURIES',
            12: 'GERANIUM',
            13: 'CRUCIFER',
            14: 'POPPY',
            15: 'PLANTAGO',
            16: 'LABIATE'
        }
        with torch.no_grad():
            batch = batch.to(self.device)
            outputs = self.model(batch)
            _, predicted = torch.max(outputs, 1)

        predictions = [id_to_class[pred.item()] for pred in predicted]

        return predictions
        

