import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.data_loaders import get_data_loaders
from models.teachers import Teacher_efficientnet_b4
from models.students import (
    StudentModel,
    StudentCustom,
    StudentResNet
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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # Stampa configurazioni
    print("\n=== Configurations ===")
    print(OmegaConf.to_yaml(cfg))

    # Fix SSL error
    import ssl
    ssl._create_default_https_context = ssl._create_stdlib_context

    # Calcolo Learning Rate in base alla dimensione del batch
    lr = calc_hyperparam(cfg.batch_size)
    print(f"learning rate: {lr}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataloader
    train_loader, val_loader, test_loader = get_data_loaders(cfg.data_dir, cfg.batch_size)

    # Classi del dataset EuroSAT
    class_names = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
    ]
    num_classes = len(class_names)

    # Modello Teacher
    teacher_model = Teacher_efficientnet_b4(num_classes=num_classes).to(device)

    student_models = {
        "SimpleNet": StudentModel(num_classes=num_classes).to(device),
        "StudentResNet1": StudentCustom(num_classes=num_classes).to(device),
        "StudentResNet2": StudentResNet(num_classes=num_classes).to(device)
    }

    # Criterio di perdita
    criterion = torch.nn.CrossEntropyLoss()

    results = {}

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
        print(f"{s_name} Knowledge Distillation - Test Metrics: {student_test_metrics}")

        results[s_name] = {
            "Student Test Metrics": student_test_metrics
        }

    # Stampa risultati finali
    print("\n=== Final Results ===")
    print(f"Teacher: {teacher_test_metrics}")
    for s_name, result in results.items():
        print(f"Student - {s_name}: {result['Student Test Metrics']}")
        #plot_confusion_matrix(result['Student Test Metrics']['Confusion Matrix'], class_names)



if __name__ == "__main__":
    main()
