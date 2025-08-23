import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils.data_loaders import denormalize

def grad_cam(model, input_image, target_class, layer_name):  ###############################QULCHE MODIFICA PURE QUA
    model.eval()

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    # Ottieni il layer dal nome
    target_layer = dict(model.named_modules())[layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward
    output = model(input_image)
    if target_class is None:
        target_class = output.argmax(dim=1).item()  # Predizione del modello
    loss = output[0, target_class]

    # Backward
    model.zero_grad()
    loss.backward()

    # Recupera attivazioni e gradienti
    act = activations['value'].detach()
    grad = gradients['value'].detach()

    # Calcola i pesi medi
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    # Normalizza
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Rimuovi gli hook
    forward_handle.remove()
    backward_handle.remove()

    return cam

def plot_grad_cam(input_image, grad_cam_map, student_name, output_dir): ############################ MODIFICATO
    
    # Step 1: Converti tensore torch in array numpy RGB (C, H, W) -> (H, W, C)
    input_image_denorm = denormalize(input_image.squeeze().cpu().detach())  # (3, 64, 64)
    input_image_np = input_image_denorm.numpy()
    input_image_np = np.transpose(input_image_np, (1, 2, 0))       # -> (64, 64, 3)
    input_image_np = np.clip(input_image_np, 0, 1)                 # clamp tra 0 e 1

    # Step 2: Resize della heatmap Grad-CAM da (x, x) a (64, 64)
    grad_cam_map_resized = cv2.resize(grad_cam_map, (64, 64))      # (64, 64)
    grad_cam_map_resized = np.clip(grad_cam_map_resized, 0, 1)     # clamp tra 0 e 1

    # Step 3: Crea heatmap colorata
    heatmap = np.uint8(255 * grad_cam_map_resized)                 # [0,255] uint8
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)         # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)             # RGB

    # Step 4: Sovrapposizione trasparente (alpha blending)
    input_image_rgb = (input_image_np * 255).astype(np.uint8)      # da float [0,1] a uint8 [0,255]
    #superimposed_img = cv2.addWeighted(src1=input_image_rgb, alpha=0.6,
    #                                   src2=heatmap, beta=0.4, gamma=0)

    
    # Plot imgs
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(input_image_np)
    axs[0].set_title("Immagine originale")
    axs[0].axis('off')

    axs[1].imshow(input_image_rgb)
    axs[1].imshow(heatmap, alpha=0.3)  # Sovrapposizione della heatmap
    axs[1].set_title("Grad-CAM")
    axs[1].axis('off')

    img_dir = os.path.join(output_dir, "gradcam_images")
    os.makedirs(img_dir, exist_ok=True)

    plt.suptitle(f"Grad-CAM for {student_name}", fontsize=16)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(img_dir, f"{student_name}_grad_cam.png"))
    plt.close(fig)

    

