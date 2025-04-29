from VIT_files.vit_inference import WeedDetector
from nanodet_files.ViTnanodet.nanodet_inference import PlantDetector
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def batcher(bbox_predictions, vit_ids, image):
    transform = transforms.Compose([
        transforms.ToTensor(),                # Convert to tensor [C, H, W]
        # transforms.Permute([2, 0, 1]),        # Rearrange from [H, W, C] to [C, H, W]
        transforms.Resize((32,32)),        # Resize all images to 224x224
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    transformed_patches = []
    for idx in vit_ids:
        detection = bbox_predictions[idx][1]
        x1, y1, x2, y2 = map(int,detection[:4])
        patch = image[y1:y2, x1:x2]
        transformed_patches.append(transform(patch))

    if len(transformed_patches) > 0:
        batch = torch.stack(transformed_patches)
    else:
        batch = None
    return batch

def annotate(image, final_predictions):
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(image)
    
    for idx, prediction in final_predictions.items():
        label, bbox = prediction.values()
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), width, height, 
            linewidth=2, 
            edgecolor="green", 
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x_min, y_min - 5, 
            label, 
            color="green", 
            fontsize=12, 
            bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2}
        )
    ax.set_xticks([])
    ax.set_yticks([])
    
    
    plt.tight_layout()
    # Convert the figure to a numpy array
    fig.canvas.draw()
    annotated_image = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Close the figure to free memory
    plt.close(fig)
    
    return annotated_image

def crop_and_weed_pipeline(plant_detector, weed_classifier, image_path):
    bbox_predictions = {}
    vit_ids = []
    final_predictions = {}

    
    meta, results = plant_detector.inference(image_path)
    image = meta["raw_img"][0].copy()
    
    for idx, (class_id, detection) in enumerate(results.items()):
        bbox_predictions[idx] = [class_id, detection]

    for idx, result in bbox_predictions.items():
        class_id, detection = result
        if class_id == 0:
            final_predictions[idx] = {"label": "CROP",
                                      "bbox" : detection[:4],
                                     }
        else:
            if len(detection) == 5 and detection[4] >= 0.35:
                final_predictions[idx] = {"label": "WEED",
                                          "bbox" : detection[:4]
                                         }
                vit_ids.append(idx)
    
    batch = batcher(bbox_predictions, vit_ids, image)
    if batch:
        predictions = weed_classifier.predict(batch)
        for idx, prediction in zip(vit_ids, predictions):
            final_predictions[idx]["label"] += f"_{prediction}"
    
    annotated_img = annotate(image, final_predictions)
    plt.imsave("annotated_image.jpg", annotated_img)
    plt.figure(figsize=(12, 12))
    plt.imshow(annotated_img)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Paths to the model files
    plant_detector_config_path = "nanodet_files/ViTnanodet/config/nanodet-plus-m_416-yolo-cpu.yml"
    plant_detector_model_path = "nanodet_files/ViTnanodet/saved_models/nanodet_model_best.pth"
    weed_classifier_model_path = "VIT_files/model_weights/vit_tiny.pth"

    # Initialize the detectors
    plant_detector = PlantDetector(plant_detector_config_path, plant_detector_model_path)
    weed_classifier = WeedDetector(weed_classifier_model_path)

    # Path to the image
    image_path = "images/agri_0_1083.jpeg"

    # Run the pipeline
    crop_and_weed_pipeline(plant_detector, weed_classifier, image_path)
    # Example usage
