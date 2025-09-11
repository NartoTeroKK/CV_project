from torchvision.datasets import EuroSAT
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter
import numpy as np
import torch

def get_data_loaders(data_dir, batch_size=32, val_split=0.1, test_split=0.1):
    """
    Crea DataLoader per il dataset EuroSAT, suddiviso in train, validation e test set.

    Args:
        data_dir (str): Percorso per scaricare il dataset.
        batch_size (int): Dimensione del batch per i DataLoader.
        val_split (float): Percentuale del dataset da usare per la validation (default: 10%).
        test_split (float): Percentuale del dataset da usare per il test set (default: 10%).

    Returns:
        train_loader, val_loader, test_loader: DataLoader per training, validation e test.
    """
    # Verifica che val_split e test_split siano validi
    assert 0 < val_split < 1, "val_split deve essere tra 0 e 1."
    assert 0 < test_split < 1, "test_split deve essere tra 0 e 1."
    assert val_split + test_split < 1, "La somma di val_split e test_split deve essere inferiore a 1."

    """ train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizzazione standard ImageNet
    ]) """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalizzazione standard ImageNet
    ])


    # Caricamento del dataset completo
    dataset = EuroSAT(root=data_dir, transform=transform, download=True)
    
    # Distribuzione delle classi nel dataset completo:
    #     AnnualCrop: 3000 immagini
    #     Forest: 3000 immagini
    #     HerbaceousVegetation: 3000 immagini
    #     Highway: 2500 immagini
    #     Industrial: 2500 immagini
    #     Pasture: 2000 immagini
    #     PermanentCrop: 2500 immagini
    #     Residential: 3000 immagini
    #     River: 2500 immagini
    #     SeaLake: 3000 immagini


    """
    # Estrazione delle etichette (classi)
    labels = [sample[1] for sample in dataset]

    # Calcola i pesi inversi rispetto alla frequenza
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = [weights[label] for label in labels]

    # Conteggio delle occorrenze delle classi
    class_counts = Counter(labels)

    # Visualizzazione della distribuzione delle classi
    class_names = dataset.classes
    class_distribution = [class_counts[i] for i in range(len(class_names))]
    print("Distribuzione delle classi nel dataset completo:")
    for class_name, count in zip(class_names, class_distribution):
        print(f"  {class_name}: {count} immagini")
    """


    # Calcolo delle dimensioni per train, validation e test set
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size

    # Suddivisione del dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Creazione dei DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def denormalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return image_tensor * std + mean

