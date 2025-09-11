from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torchmetrics
import os

def evaluate_model(model, dataloader, criterion, device, num_classes, test=False):
    """Calcola metriche di valutazione per il modello.
    
    Args:
        model (torch.nn.Module): Il modello da valutare.
        dataloader (torch.utils.data.DataLoader): Il dataloader per la valutazione.
        device (torch.device): Il dispositivo di calcolo (CPU/GPU).
        num_classes (int): Numero delle classi.
        test (bool): Se True, calcola tutte le metriche; se False, calcola solo quelle per la validazione.

    Returns:
        dict: Dizionario contenente le metriche richieste.
    """
    
    # Metriche standard
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='none').to(device)
    recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='none').to(device)
    f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='none').to(device)
    confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device) if test else None
    
    model.eval()
    total_loss = 0.0
    total_time = 0.0 if test else None
    num_params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Misura il tempo di inferenza solo in fase di test
            if test:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                ###
                outputs = model(inputs)
                ###
                end_time.record()
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                total_time += inference_time
            else:
                outputs = model(inputs)
            
            # Calcolo della loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calcolo delle metriche
            preds = torch.argmax(outputs, dim=1)
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1_score.update(preds, labels)
            if test:
                confusion_matrix.update(preds, labels)
        
    
    # Computazione delle metriche finali
    avg_loss = total_loss / len(dataloader)
    accuracy_value = accuracy.compute().item()
    precision_value = precision.compute().cpu().numpy()
    recall_value = recall.compute().cpu().numpy()
    f1_score_value = f1_score.compute().cpu().numpy()
    
    metrics = {
        "Loss": round(avg_loss, 4),
        "Accuracy": round(accuracy_value, 4),
        "Precision": round(float(precision_value.mean().item()), 4),
        "Recall": round(float(recall_value.mean().item()), 4),
        "F1 Score": round(float(f1_score_value.mean().item()), 4),
    }
    
    if test:
        avg_inference_time = total_time / len(dataloader.dataset)
        metrics.update({
            "Total Inference Time": round(total_time,4),
            "Single Img Avg Inference Time": round(avg_inference_time, 4),
            "Number of Parameters": num_params,
            "Accuracy/Num Parameters": metrics["Accuracy"] / num_params,
            "Accuracy/Avg Inference Time": round(metrics["Accuracy"] / avg_inference_time, 4),
            "Confusion Matrix": confusion_matrix.compute().cpu().numpy()
        })
    
    # Reset delle metriche
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()
    if test:
        confusion_matrix.reset()
    
    return metrics


def plot_confusion_matrix(conf_matrix, class_names, student_name, output_dir):
    """Visualizza la Confusion Matrix con Seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {student_name}")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    #plt.show()

    img_dir = os.path.join(output_dir, "confusion_matrices")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    plt.savefig(os.path.join(img_dir, f"{student_name}_confusion_matrix.png"))
    plt.close()

