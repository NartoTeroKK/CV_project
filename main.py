import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from torch import nn

from utils.data_loaders import get_data_loaders
from models.teachers import Teacher_efficientnet_b4
from models.students import (
    SimpleNet,
    StudentResNet,
    StudentNetSeparable,
    StudentNetDepthwiseSkip,
    StudentNetLight
)
from utils.training import (
    train_teacher, 
    train_student_with_distillation, 
    distillation_loss,
    calc_learning_rate
)
from utils.evaluation import (
    evaluate_model,
    plot_confusion_matrix
)
from utils.gradcam import (
    grad_cam, 
    plot_grad_cam,
    get_data_for_grad_cam,
    grad_cam_comparison,
    plot_grad_cam_comparison
)

import numbers

def is_number(var):
    return isinstance(var, numbers.Number)

def load_model_weights(model: nn.Module, model_name: str, base_dir: str):
    """
    Carica i pesi di un modello dato il suo nome.

    Args:
        model (nn.Module): L'istanza del modello PyTorch.
        model_name (str): Il nome del modello (es. 'resnet').
        base_dir (str): La directory di base dove sono salvati i pesi.
    
    Returns:
        torch.nn.Module: Il modello con i pesi caricati.
    """
    model_path = os.path.join(base_dir, "student_models", f"{model_name}.pth")
    if not os.path.exists(model_path):
        hydra.utils.log.error(f"File not found: {model_path}")
        return None
    
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    return model

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    # matplotlib backend
    import matplotlib
    matplotlib.use('TkAgg') 

    # Stampa configurazioni
    hydra.utils.log.info("\n=== Configurations ===\n"+OmegaConf.to_yaml(cfg))

    # Calcolo Learning Rate in base alla dimensione del batch
    lr = calc_learning_rate(cfg.batch_size)
    hydra.utils.log.info(f"\nlearning rate: {lr}\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hydra.utils.log.info(f"\ndevice: {device}\n")

    # Imposta il seed per la riproducibilità
    if cfg.seed is not None and is_number(cfg.seed):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    else:
        hydra.utils.log.warning("No seed set!") 


    # Caricamento dei dataloader per training, validation e test
    train_loader, val_loader, test_loader = get_data_loaders(cfg.data_dir, cfg.batch_size)

    # Classi del dataset EuroSAT
    class_names = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
    ]
    num_classes = len(class_names)

    # Modello Teacher
    teacher_model = Teacher_efficientnet_b4(num_classes=num_classes).to(device)

    # Dizionario con tutti i modelli student da addestrare
    student_models = {
        "SimpleNet": SimpleNet(num_classes=num_classes).to(device),
        "StudentResNet": StudentResNet(num_classes=num_classes).to(device),
        "StudentNetSeparable": StudentNetSeparable(num_classes=num_classes).to(device),
        "StudentNetDepthwiseSkip": StudentNetDepthwiseSkip(num_classes=num_classes).to(device),
        "StudentNetLight": StudentNetLight(num_classes=num_classes).to(device)
    }

    # Dizionario con il layer per Grad-CAM per ogni modello
    grad_cam_layers = {
        "SimpleNet": ["conv1", "conv2"],
        "StudentResNet": ["conv1", "conv2", "residual_block", "conv3"],
        "StudentNetSeparable": ["depthwise", "pointwise", "conv2"],
        "StudentNetDepthwiseSkip": ["depthwise", "pointwise", "conv2"],
        "StudentNetLight": ["conv1", "conv2"]
    }

    # Definizione della loss function (CrossEntropy per classificazione multiclass)
    criterion = torch.nn.CrossEntropyLoss()

    # Dizionario dove verranno salvati i risultati finali
    results = {}

    # Directory di output per i risultati
    hydra_oututput_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    hydra.utils.log.info(f"\n=== Training Teacher EfficientNet-B4 ===")
    optimizer_teacher = torch.optim.Adam(teacher_model.parameters(), lr=lr)

    # Training del Teacher
    for epoch in range(cfg.t_epochs):
        loss, accuracy = train_teacher(teacher_model, train_loader, criterion, optimizer_teacher, device)
        hydra.utils.log.info(f"\nEpoch {epoch + 1}/{cfg.t_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Valutazione del Teacher sul test set
    teacher_test_metrics = evaluate_model(teacher_model, test_loader, criterion, device, num_classes)
    hydra.utils.log.info(f"\nTEACHER - Test Metrics: {teacher_test_metrics}")


    # Creazione dello Student
    for s_name, student in student_models.items():
        #student = StudentModel(num_classes=num_classes).to(device)
        optimizer_student = torch.optim.Adam(student.parameters(), lr=lr)

        # Early stopping setup
        best_val_loss = float("inf")
        patience = 4
        epochs_without_improvement = 0
        best_state_dict = None  # Qui salviamo i pesi migliori in memoria (non su file)

        hydra.utils.log.info(f"\n=== Training {s_name} with Teacher EfficientNet-B4 ===")
        for epoch in range(cfg.s_epochs):
            loss, accuracy, val_metrics = train_student_with_distillation(
                student, teacher_model, train_loader, criterion,
                distillation_loss, cfg.alpha, cfg.temperature,
                optimizer_student, device,
                evaluate_model, val_loader, num_classes
            )
            hydra.utils.log.info(f"\nEpoch {epoch + 1}/{cfg.s_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}.\n"
                  f"Validation Metrics:\n {val_metrics}")

            val_loss = val_metrics.get("Loss", None)

            if val_loss is not None:
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state_dict = student.state_dict()  # Salva i pesi migliori
                    epochs_without_improvement = 0
                    hydra.utils.log.info("Miglioramento trovato, modello salvato in memoria")
                else:
                    epochs_without_improvement += 1
                    hydra.utils.log.info(f"⏸ Nessun miglioramento ({epochs_without_improvement}/{patience})")

                if epochs_without_improvement >= patience:
                    hydra.utils.log.info("Early stopping attivato")
                    break

        # Ripristina il modello ai pesi migliori prima del test
        if best_state_dict is not None:
            student.load_state_dict(best_state_dict)
            hydra.utils.log.info("Modello ripristinato ai pesi migliori")

        # Valutazione dello Student dopo KD
        student_test_metrics = evaluate_model(student, test_loader, criterion, device, num_classes, test=True)
        hydra.utils.log.info(f"\n{s_name} - Test Metrics: {student_test_metrics}")

        # Seleziona il layer corrispondente
        layers_names = grad_cam_layers[s_name]

        # Ottieni un'immagine con predizione corretta ed una con predizione errata sulle quali eseguire Grad-CAM
        correct_data, wrong_data = get_data_for_grad_cam(student, test_loader, device)

        # Esegui Grad-CAM e plotting 
        if correct_data:
            img, true_label, pred_label = correct_data
            for layer_name_grad in layers_names:
                grad_cam_map = grad_cam(student, img, target_class=pred_label, layer_name=layer_name_grad)
                true_label_name = class_names[true_label]
                pred_label_name = class_names[pred_label]
                plot_grad_cam(
                    img,
                    grad_cam_map, 
                    student_name=s_name, 
                    labels=(true_label_name,pred_label_name), 
                    extra_label=f"CORRECT_{layer_name_grad}", 
                    output_dir=hydra_oututput_dir
                )
                hydra.utils.log.info(f"{s_name} - Predizione corretta: True={true_label_name}, Pred={pred_label_name} - Layer: {layer_name_grad}")
        if wrong_data:
            img, true_label, pred_label = wrong_data
            for layer_name_grad in layers_names:
                grad_cam_map = grad_cam(student, img, target_class=pred_label, layer_name=layer_name_grad)
                true_label_name = class_names[true_label]
                pred_label_name = class_names[pred_label]
                plot_grad_cam(
                    img, 
                    grad_cam_map, 
                    student_name=s_name, 
                    labels=(true_label_name,pred_label_name), 
                    extra_label=f"WRONG_{layer_name_grad}", 
                    output_dir=hydra_oututput_dir
                )
                hydra.utils.log.info(f"{s_name} - Predizione sbagliata: True={true_label_name}, Pred={pred_label_name} - Layer: {layer_name_grad}")

        # Salva i risultati
        results[s_name] = {
            "Student Test Metrics": student_test_metrics
        }

        #Salva i pesi del modello
        model_path = os.path.join(hydra_oututput_dir, "student_models")
        os.makedirs(model_path, exist_ok=True)
        model_path = os.path.join(model_path, f"{s_name}.pth")
        torch.save(student.state_dict(), model_path)
        hydra.utils.log.info(f"{s_name} model weights saved")
    
    # Confronto tra il miglior e il peggior modello in termini di accuracy
    best_student_name = max(results, key=lambda k: results[k]['Student Test Metrics']['Accuracy'])
    worst_student_name = min(results, key=lambda k: results[k]['Student Test Metrics']['Accuracy'])

    hydra.utils.log.info(f"\nBest Student: {best_student_name} with Accuracy: {results[best_student_name]['Student Test Metrics']['Accuracy']}")
    hydra.utils.log.info(f"\nWorst Student: {worst_student_name} with Accuracy: {results[worst_student_name]['Student Test Metrics']['Accuracy']}")
    
    best_student = load_model_weights(student_models[best_student_name], best_student_name, hydra_oututput_dir)
    worst_student = load_model_weights(student_models[worst_student_name], worst_student_name, hydra_oututput_dir)
    
    if best_student is None or worst_student is None:
        hydra.utils.log.error("Error loading best or worst student models for Grad-CAM comparison.")
        return
    

    hydra.utils.log.info(f"\n=== Grad-CAM Comparison between Best ({best_student_name}) and Worst ({worst_student_name}) Students ===")
    selected_data = grad_cam_comparison(test_loader, best_student, worst_student, device)

    if selected_data:
        img, true_label, pred_best, pred_worst = selected_data

        layer_best = grad_cam_layers[best_student_name][-1]
        layer_worst = grad_cam_layers[worst_student_name][-1]

        grad_cam_best = grad_cam(best_student, img, target_class=pred_best, layer_name=layer_best)
        grad_cam_worst = grad_cam(worst_student, img, target_class=pred_worst, layer_name=layer_worst)

        plot_grad_cam_comparison(
            img, grad_cam_best, grad_cam_worst,
            student_best_name=best_student_name,
            student_worst_name=worst_student_name,
            labels=(class_names[true_label],class_names[pred_best],class_names[pred_worst]),
            output_dir=hydra_oututput_dir
        )

        hydra.utils.log.info(f"Confronto Grad-CAM: True={class_names[true_label]}, {best_student_name} Pred={class_names[pred_best]}, {worst_student_name} Pred={class_names[pred_worst]}")

    else:
        hydra.utils.log.info(f"Nessuna immagine trovata dove {best_student_name} ha indovinato e {worst_student_name} ha sbagliato.")


    # Stampa risultati finali
    hydra.utils.log.info("\n=== Final Results ===")
    hydra.utils.log.info(f"\nTeacher: {teacher_test_metrics}\n")
    for s_name, result in results.items():
        hydra.utils.log.info(f"\n{s_name}: {result['Student Test Metrics']}")
        plot_confusion_matrix(result['Student Test Metrics']['Confusion Matrix'], class_names, s_name, hydra_oututput_dir)
    
    hydra.utils.log.info("All models trained and evaluated successfully.")


if __name__ == "__main__":
    main()
