import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from utils.data_loaders import get_data_loaders
from models.teachers import Teacher_efficientnet_b4
from models.students import (
    SimpleNet,
    StudentResNet1,
    StudentNetSeparable,
    StudentNetDepthwiseSkip,
    StudentNetLight
)
from utils.training import (
    train_teacher, 
    train_student_with_distillation, 
    distillation_loss,
    calc_hyperparam
)
from utils.evaluation import (
    evaluate_model,
    plot_confusion_matrix
)
from utils.gradcam import grad_cam, plot_grad_cam

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    # matplotlib backend
    import matplotlib
    matplotlib.use('TkAgg') 

    # Stampa configurazioni
    print("\n=== Configurations ===")
    print(OmegaConf.to_yaml(cfg))


    # Imposta il seed per la riproducibilità
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)


    # Fix SSL error
    #import ssl
    #ssl._create_default_https_context = ssl._create_stdlib_context

    # Calcolo Learning Rate in base alla dimensione del batch
    lr = calc_hyperparam(cfg.batch_size)
    print(f"learning rate: {lr}\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
        "StudentResNet1": StudentResNet1(num_classes=num_classes).to(device),
        "StudentNetSeparable": StudentNetSeparable(num_classes=num_classes).to(device),
        "StudentNetDepthwiseSkip": StudentNetDepthwiseSkip(num_classes=num_classes).to(device),
        "StudentNetLight": StudentNetLight(num_classes=num_classes).to(device)
    }

    # Dizionario con il layer per Grad-CAM per ogni modello
    grad_cam_layers = {
        "SimpleNet": "conv2",  # poiché è l'ultimo layer convoluzionale prima del flattening. Ha una buona rappresentazione delle caratteristiche.
        "StudentResNet1": "conv3",  # Questo layer è il terzo layer convoluzionale e si trova più in profondità nella rete, ma non è ancora un fully connected layer. Ha un buon mix di astrazione e rilevamento di caratteristiche semantiche.
        "StudentNetSeparable": "conv2",  # La seconda convoluzione è un buon punto per osservare come la rete stia combinando le caratteristiche dopo la convoluzione separabile.
        "StudentNetDepthwiseSkip": "conv2",  # Conv2 per StudentNetDepthwiseSkip
        "StudentNetLight": "conv2",  # Poiché conv2 è l'ultimo layer convoluzionale, applicare GradCAM qui ti permette di visualizzare quali caratteristiche spaziali siano state apprese prima del passaggio alla fase di classificazione.
    }


    # Definizione della loss function (CrossEntropy per classificazione multiclass)
    criterion = torch.nn.CrossEntropyLoss()

    # Dizionario dove verranno salvati i risultati finali
    results = {}

    # Directory di output per i risultati
    hydra_oututput_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    print(f"\n=== Training Teacher EfficientNet-B4 ===")
    optimizer_teacher = torch.optim.Adam(teacher_model.parameters(), lr=lr)

    # Training del Teacher
    for epoch in range(cfg.t_epochs):
        loss, accuracy = train_teacher(teacher_model, train_loader, criterion, optimizer_teacher, device)
        print(f"Epoch {epoch + 1}/{cfg.t_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Valutazione del Teacher sul test set
    teacher_test_metrics = evaluate_model(teacher_model, test_loader, criterion, device, num_classes)
    print(f"TEACHER - Test Metrics: {teacher_test_metrics}")

    # Creazione dello Student
    for s_name, student in student_models.items():
        #student = StudentModel(num_classes=num_classes).to(device)
        optimizer_student = torch.optim.Adam(student.parameters(), lr=lr)

        print(f"\n=== Training {s_name} with Teacher EfficientNet-B4 ===")
        for epoch in range(cfg.s_epochs):
            loss, accuracy, val_metrics = train_student_with_distillation(student, teacher_model, train_loader, criterion,
                                                distillation_loss, cfg.alpha, cfg.temperature, optimizer_student, device,
                                                evaluate_model, val_loader, num_classes)
            print(f"Epoch {epoch + 1}/{cfg.s_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}.\n Validation Metrics:\n {val_metrics}")

        # Valutazione dello Student dopo KD
        student_test_metrics = evaluate_model(student, test_loader, criterion, device, num_classes, test=True)
        print(f"{s_name} - Test Metrics: {student_test_metrics}")

        # Esegui Grad-CAM per il primo esempio nel test set
        input_image, _ = next(iter(test_loader))  # Prendi un batch dal dataloader
        input_image = input_image[0].unsqueeze(0).to(device)  # Prendi il primo esempio

        # Seleziona il layer corrispondente
        layer_name_grad = grad_cam_layers[s_name]

        # Calcola la classe predetta dallo student per l'immagine
        student.eval()
        with torch.no_grad():
            output = student(input_image)
            predicted_class = torch.argmax(output, dim=1).item()

        # Applica Grad-CAM sulla classe effettivamente predetta
        grad_cam_map = grad_cam(student, input_image, target_class=predicted_class, layer_name=layer_name_grad)
        hydra.utils.log.info(f"Grad-CAM map for {s_name} calculated.")
        
        # Salva e visualizza Grad-CAM
        plot_grad_cam(input_image, grad_cam_map, s_name, hydra_oututput_dir)
        hydra.utils.log.info(f"Grad-CAM map for {s_name} plotted and saved.")

        results[s_name] = {
            "Student Test Metrics": student_test_metrics
        }

    # Stampa risultati finali
    print("\n=== Final Results ===")
    print(f"Teacher: {teacher_test_metrics}")
    for s_name, result in results.items():
        print(f"Student - {s_name}: {result['Student Test Metrics']}")
        plot_confusion_matrix(result['Student Test Metrics']['Confusion Matrix'], class_names, s_name, hydra_oututput_dir)
    
    hydra.utils.log.info("All models trained and evaluated successfully.")


if __name__ == "__main__":
    main()
