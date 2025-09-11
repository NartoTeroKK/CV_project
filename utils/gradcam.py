import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils.data_loaders import denormalize

def grad_cam(model, input_image, target_class, layer_name): 
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

def plot_grad_cam(input_image, grad_cam_map, student_name, labels, extra_label, output_dir): ############################ MODIFICATO
    
    input_image_np, input_image_rgb, heatmap = set_grad_cam_img(input_image, grad_cam_map)
    
    true_label, pred_label = labels

    # Plot imgs
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    fig.subplots_adjust(bottom=0.2)

    axs[0].imshow(input_image_np)
    axs[0].set_title("Immagine originale", fontsize=15)
    axs[0].set_xlabel(true_label, fontsize=14)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    

    axs[1].imshow(input_image_rgb)
    axs[1].imshow(heatmap, alpha=0.25)  # Sovrapposizione della heatmap
    axs[1].set_title("Grad-CAM", fontsize=15)
    axs[1].set_xlabel(pred_label, fontsize=14)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    img_dir = os.path.join(output_dir, "gradcam_images", student_name)
    img_label = f"{student_name}_{extra_label}"
    os.makedirs(img_dir, exist_ok=True)

    plt.suptitle(f"Grad-CAM for {img_label}", fontsize=17)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(img_dir, f"{img_label}.png"))
    plt.close(fig)

def set_grad_cam_img(input_image, grad_cam_map):
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

    return input_image_np, input_image_rgb, heatmap

    
def get_data_for_grad_cam(student, test_loader, device):
    # Inizializza le variabili per salvare i dati
    correct_data = None
    wrong_data = None

    student.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = student(inputs)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                img = inputs[i].unsqueeze(0)  # aggiungi batch dim

                # Caso predizione corretta
                if correct_data is None and true_label == pred_label:
                    correct_data = (img.clone(), true_label, pred_label)

                # Caso predizione errata
                if wrong_data is None and true_label != pred_label:
                    wrong_data = (img.clone(), true_label, pred_label)

                # Se li ho trovati entrambi, li ritorno
                if correct_data is not None and wrong_data is not None:
                    return correct_data, wrong_data 
    
    return correct_data, wrong_data


def grad_cam_comparison(test_loader, student_best, student_worst, device):

    student_best.eval()
    student_worst.eval()

    selected_data = None

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs_best = student_best(inputs)
            preds_best = torch.argmax(outputs_best, dim=1)

            outputs_worst = student_worst(inputs)
            preds_worst = torch.argmax(outputs_worst, dim=1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_best = preds_best[i].item()
                pred_worst = preds_worst[i].item()
                img = inputs[i].unsqueeze(0)

                if pred_best == true_label and pred_worst != true_label:
                    selected_data = (img.clone(), true_label, pred_best, pred_worst)
                    return selected_data  
    return selected_data

def plot_grad_cam_comparison(input_image, grad_cam_map_best, grad_cam_map_worst, student_best_name, student_worst_name, labels, output_dir):
    
    input_image_np, input_image_rgb, heatmap_best = set_grad_cam_img(input_image, grad_cam_map_best)
    _, _, heatmap_worst = set_grad_cam_img(input_image, grad_cam_map_worst)

    true_label, pred_best, pred_worst = labels

    # Plot imgs
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    fig.subplots_adjust(bottom=0.2)

    axs[0].imshow(input_image_np)
    axs[0].set_title("Immagine originale", fontsize=15)
    axs[0].set_xlabel(true_label, fontsize=14)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(input_image_rgb)
    axs[1].imshow(heatmap_best, alpha=0.25)  # Sovrapposizione della heatmap
    axs[1].set_title(f"Grad-CAM {student_best_name}", fontsize=15)
    axs[1].set_xlabel(pred_best, fontsize=14)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].imshow(input_image_rgb)
    axs[2].imshow(heatmap_worst, alpha=0.25)  # Sovrapposizione della heatmap
    axs[2].set_title(f"Grad-CAM {student_worst_name}", fontsize=15)
    axs[2].set_xlabel(pred_worst, fontsize=14)
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    img_dir = os.path.join(output_dir, "gradcam_images")
    img_label = f"{student_best_name}_vs_{student_worst_name}"
    os.makedirs(img_dir, exist_ok=True)

    plt.suptitle(f"Grad-CAM Comparison for {img_label}", fontsize=17)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(img_dir, f"{img_label}.png"))
    plt.close(fig)