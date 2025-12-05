# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os # Necesario tener puntos de partida (checkpointing)

from dataset import BronchoscopyDataset
from model import SequentialPoseTransformer

# CONFIGURACIÓN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

ROOT_DIR = "data/VirtualNavigations" #Pon la ruta correcta (si es que la hiciste diferente)
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 50
SEQUENCE_LENGTH = 5

MODEL_SAVE_PATH = "bronco_model_128px_all_patients.pth"
LOSS_SAVE_PATH = "training_losses_all_patients.npz"

# El archivo de checkpoint que guarda tu progreso
CHECKPOINT_PATH = "training_checkpoint.pth"

# PREPARACIÓN DE DATOS
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = BronchoscopyDataset(
    root_dir=ROOT_DIR, 
    sequence_length=SEQUENCE_LENGTH, 
    transform=data_transform
)

torch.manual_seed(42)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# MODELO, FUNCIÓN de PÉRDIDA Y OPTIMIZADOR
model = SequentialPoseTransformer(num_pose_outputs=7, sequence_length=SEQUENCE_LENGTH).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

history_train_loss = []
history_val_loss = []

# PROCESO DE CHECKPOINT
start_epoch = 0 
if os.path.exists(CHECKPOINT_PATH):
    print(f"Cargando checkpoint desde {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1 
    history_train_loss = checkpoint['history_train_loss']
    history_val_loss = checkpoint['history_val_loss']
    print(f"¡Checkpoint cargado! Reanudando desde la época {start_epoch + 1}")
else:
    print("No se encontró checkpoint. Empezando desde cero.")

# BUCLE DE ENTRENAMIENTO
print("--- Iniciando Entrenamiento Final (Todos los pacientes) ---")

for epoch in range(start_epoch, EPOCHS): 
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS}")
    for images, poses in progress_bar:
        images, poses = images.to(DEVICE), poses.to(DEVICE)
        predicted_poses = model(images)
        loss = criterion(predicted_poses, poses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # Bucle de validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, poses in val_loader:
            images, poses = images.to(DEVICE), poses.to(DEVICE)
            predicted_poses = model(images)
            loss = criterion(predicted_poses, poses)
            val_loss += loss.item()
            
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    history_train_loss.append(avg_train_loss)
    history_val_loss.append(avg_val_loss)
    
    print(f"Época {epoch+1}: Pérdida de Entrenamiento: {avg_train_loss:.4f}, Pérdida de Validación: {avg_val_loss:.4f}")

    print(f"Guardando checkpoint de la época {epoch+1}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history_train_loss': history_train_loss,
        'history_val_loss': history_val_loss,
    }, CHECKPOINT_PATH)
    print("Checkpoint guardado.")

# GUARDADO DE RESULTADOS FINALES
print("\n¡Entrenamiento completado!")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
np.savez(LOSS_SAVE_PATH, train_loss=history_train_loss, val_loss=history_val_loss)

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
 
print(f"Modelo final guardado en: {MODEL_SAVE_PATH}")
print(f"Historial de pérdidas guardado en: {LOSS_SAVE_PATH}")
