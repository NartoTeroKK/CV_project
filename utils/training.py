import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_teacher(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calcolo della perdita
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calcolo dell'accuracy
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def train_student_with_distillation(
        student, teacher, train_loader, criterion, distill_loss,
        alpha, T, optimizer, device, evaluate_fn, val_loader, num_classes
    ):

    student.train()
    teacher.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        teacher_outputs = teacher(inputs).detach()
        student_outputs = student(inputs)

        # Calcolo delle perdite
        hard_loss = criterion(student_outputs, labels)
        soft_loss = distill_loss(student_outputs, teacher_outputs, T)
        loss = alpha * soft_loss + (1 - alpha) * hard_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calcolo dell'accuracy
        _, preds = torch.max(student_outputs, dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples

    # Esegui la validazione usando la funzione passata come parametro
    metrics = evaluate_fn(student, val_loader, criterion, device, num_classes)

    # Aggiorna il learning rate in base alla loss di validazione
    val_loss = metrics['Loss']
    scheduler.step(val_loss)
    
    return avg_loss, accuracy, metrics


def distillation_loss(student_outputs, teacher_outputs, T):

    softmax = nn.Softmax(dim=1)

    return nn.KLDivLoss(reduction='batchmean')(
        torch.log(softmax(student_outputs / T)),
        softmax(teacher_outputs / T)
    )

# Funzione di valutazione del modello

def calc_hyperparam(batch_size):
    # learning rate standard per Adam con batch size 32
    lr = 0.001
    if batch_size == 32:
        return lr
    elif batch_size == 64:
        return lr * 2
    elif batch_size == 128:
        return lr * 4