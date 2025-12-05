import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# --- IMPORTA TU MODELO Y DATASET AQUÍ ---
# Asegúrate de que estos archivos existan en tu repo
from model import SequentialPoseTransformer 
from dataset import BronchoscopyDataset 
from torchvision import transforms

def get_forward_vector(q):
    """
    Convierte un lote de cuaterniones en vectores de dirección (Forward Vectors).
    
    Asume que el eje óptico de la cámara virtual corresponde al Eje X local (+1, 0, 0),
    basado en el análisis geométrico del dataset BronchoPose.
    
    Args:
        q (torch.Tensor): Tensor de forma (N, 4) con cuaterniones [w, x, y, z].
        
    Returns:
        torch.Tensor: Tensor de forma (N, 3) con los vectores de dirección normalizados.
    """
    # Desempaquetar componentes (asumiendo orden: w, x, y, z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Fórmula para rotar el vector (1, 0, 0) usando el cuaternión q
    # Esta es la primera columna de la matriz de rotación
    vx = 1 - 2 * (y**2 + z**2)
    vy = 2 * (x*y + z*w)
    vz = 2 * (x*z - y*w)
    
    vec = torch.stack([vx, vy, vz], dim=1)
    
    # Asegurar que sea unitario (buena práctica numérica)
    return vec / torch.norm(vec, dim=1, keepdim=True)

def compute_metrics(model, loader, device):
    """
    Calcula el Error de Posición (Euclidiano) y el Error de Dirección (Angular).
    """
    model.eval()
    position_errors = []
    direction_errors = []
    
    print(f"Evaluando modelo en: {device}...")
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Calculando métricas"):
            images = images.to(device)
            labels = labels.to(device) # Shape: (B, 7) -> [x, y, z, qw, qx, qy, qz]
            
            # Inferencia
            outputs = model(images)
            
            # --- 1. ERROR DE POSICIÓN (Euclidiano) ---
            pos_pred = outputs[:, :3]
            pos_true = labels[:, :3]
            
            # Distancia Euclidiana (L2 Norm)
            # Nota: Si tus datos están normalizados, aquí deberías des-normalizarlos
            # multiplicando por la desviación estándar y sumando la media del dataset.
            e_pos = torch.norm(pos_pred - pos_true, dim=1)
            position_errors.extend(e_pos.cpu().numpy())
            
            # --- 2. ERROR DE DIRECCIÓN (Angular) ---
            q_pred = outputs[:, 3:] # [w, x, y, z]
            q_true = labels[:, 3:]  # [w, x, y, z]
            
            # Obtener vectores de vista (Asumiendo cámara en Eje X)
            v_pred = get_forward_vector(q_pred)
            v_true = get_forward_vector(q_true)
            
            # Producto punto (Cosine similarity)
            dot_product = torch.sum(v_pred * v_true, dim=1)
            
            # Clamp para evitar errores numéricos fuera de [-1, 1]
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            
            # Ángulo en grados: arccos(dot) * 180/pi
            e_dir = torch.acos(dot_product) * (180.0 / np.pi)
            direction_errors.extend(e_dir.cpu().numpy())

    return np.array(position_errors), np.array(direction_errors)

if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "bronco_model_128px_all_patients.pth" # Tu mejor checkpoint
    DATA_PATH = "/path/to/your/data" # Ruta a tus datos
    BATCH_SIZE = 64
    SEQ_LENGTH = 5
    
    # --- CARGA DE DATOS ---
    # Transformaciones básicas (ajusta según tu entrenamiento)
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Cargar Dataset (Ajusta include_patients según tu set de prueba)
    test_dataset = BronchoscopyDataset(
        root_dir=DATA_PATH, 
        sequence_length=SEQ_LENGTH, 
        transform=val_transforms,
        include_patients=["LENS_P18_14_01_2016_INSP_CPAP"] # Ejemplo de paciente de prueba
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- CARGA DEL MODELO ---
    model = SequentialPoseTransformer(num_pose_outputs=7, sequence_length=SEQ_LENGTH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    
    # --- EJECUCIÓN ---
    pos_err, dir_err = compute_metrics(model, test_loader, DEVICE)
    
    # --- REPORTE DE RESULTADOS ---
    print("\n" + "="*40)
    print("      RESULTADOS DE EVALUACIÓN      ")
    print("="*40)
    
    print(f"Muestras evaluadas: {len(pos_err)}")
    print("-" * 40)
    print(f"ERROR DE POSICIÓN (mm):")
    print(f"  Media: {np.mean(pos_err):.4f} mm")
    print(f"  Std:   ±{np.std(pos_err):.4f} mm")
    print("-" * 40)
    print(f"ERROR DE DIRECCIÓN (Grados):")
    print(f"  Media: {np.mean(dir_err):.4f} deg")
    print(f"  Std:   ±{np.std(dir_err):.4f} deg")
    print("="*40)
