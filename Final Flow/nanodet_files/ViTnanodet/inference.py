import os

# Inference without GPU
command = (
    r'set PYTHONPATH=C:\Users\anush\Desktop\CODING\Artificial Intelligence\CV\Project\ViTnanodet && '
    r'python demo/demo.py image '
    r'--config "C:\Users\anush\Desktop\CODING\Artificial Intelligence\CV\Project\ViTnanodet\config\nanodet-plus-m_416-yolo-cpu.yml" '
    r'--model "C:\Users\anush\Desktop\CODING\Artificial Intelligence\CV\Project\ViTnanodet\saved_models\nanodet_model_best.pth" '
    r'--path "C:\Users\anush\Desktop\CODING\Artificial Intelligence\CV\Project\ViTnanodet\images\agri_0_1083.jpeg"'
)

## Inference with GPU

# command = (
#     r'set PYTHONPATH=C:\Users\anush\Desktop\CODING\Artificial Intelligence\CV\Project\ViTnanodet && '
#     r'python demo/demo.py image '
#     r'--config "C:\Users\anush\Desktop\CODING\Artificial Intelligence\CV\Project\ViTnanodet\config\nanodet-plus-m_416-yolo-gpu.yml" '
#     r'--model "C:\Users\anush\Desktop\CODING\Artificial Intelligence\CV\Project\ViTnanodet\saved_models\nanodet_model_best.pth" '
#     r'--path "C:\Users\anush\Desktop\CODING\Artificial Intelligence\CV\Project\ViTnanodet\images\agri_0_1028.jpeg"'
# )


os.system(command)



