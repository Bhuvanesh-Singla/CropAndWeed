# CropAndWeed
IT357 CV Project
To use the pipeline, clone the git repo to your local system.
Download the dataset from the following link
```bash
https://www.kaggle.com/datasets/bhuvaneshsingla16/cleaned-cv-dataset/
```
Navigate to the folder containing the cloned repo.
Navigate to `Final Flow` folder containing all code
Install the requirements using 
```bash
pip install -r requirements.txt
```
Run the pipeline.py with command line arguments as follows
```bash
python pipeline.py --plant_detector_config nanodet_files/ViTnanodet/config/nanodet-plus-m_416-yolo-cpu.yml \
--plant_detector_model nanodet_files/ViTnanodet/saved_models/nanodet_model_best.pth \
--weed_classifier_model VIT_files/vit_tiny.pth \
--image_path "path to your input image"
```
