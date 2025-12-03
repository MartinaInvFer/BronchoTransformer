# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
from glob import glob
from tqdm import tqdm

class BronchoscopyDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, transform=None, include_patients=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length

        all_dfs = []
        all_patient_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        if include_patients is None:
            # Modo "Entrenamiento Final": Carga todos menos el excluido
            exclude_list = ["Lens3_INSP_SIN"]
            patient_folders = [p for p in all_patient_folders if p not in exclude_list]
            print(f"Cargando todos los pacientes excepto: {exclude_list}")
        else:
            # Modo "Prueba/Test": Carga solo los pacientes de la lista
            patient_folders = [p for p in all_patient_folders if p in include_patients]
            print(f"Cargando {len(patient_folders)} pacientes especificados...")

        for patient_folder in patient_folders:
            csv_files = glob(os.path.join(root_dir, patient_folder, '*_clean.csv'))
            if not csv_files:
                csv_files = glob(os.path.join(root_dir, patient_folder, '*.csv'))
                if not csv_files:
                    continue
            
            df = pd.read_csv(csv_files[0])
            df['patient_folder'] = patient_folder
            all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No se pudo cargar ningún archivo CSV para los pacientes seleccionados.")

        self.metadata = pd.concat(all_dfs, ignore_index=True)
        print(f"Metadata inicial cargada con {len(self.metadata)} filas.")
        
        print("Pre-calculando índices de secuencias válidas...")
        self.valid_indices = []
        # Agrupa por paciente para no mezclar secuencias entre pacientes
        for _, group in tqdm(self.metadata.groupby('patient_folder')):
            # Necesitamos 5 imágenes (seq_len) + 1 pose (la 6ta)
            for i in range(len(group) - (self.sequence_length)):
                self.valid_indices.append(group.index[i])

        print(f"Se encontraron {len(self.valid_indices)} secuencias válidas posibles.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_index = self.valid_indices[idx]
        
        image_sequence = []
        for i in range(self.sequence_length):
            current_idx = start_index + i
            row = self.metadata.iloc[current_idx]
            
            filename_completo = row['filename'].strip()
            img_path = os.path.join(self.root_dir, row['patient_folder'], row['lobe'], filename_completo)
            
            image = cv2.imread(img_path)
            if image is not None:
                seq_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else: # Si una imagen falla, intenta con la siguiente secuencia (Esto en caso de que no hubiera un preprocesamietno)
                return self.__getitem__((idx + 1) % len(self))

            if self.transform:
                seq_image = self.transform(seq_image)
            
            image_sequence.append(seq_image)

        images_tensor = torch.stack(image_sequence)
        
        # Obtiene la pose de la 6ta imagen
        pose_index = start_index + self.sequence_length
        final_row = self.metadata.iloc[pose_index]
        
        position = final_row[['pos_x', 'pos_y', 'pos_z']].values.astype('float32')
        orientation = final_row[['q_w', 'q_x', 'q_y', 'q_z']].values.astype('float32')
        pose = torch.tensor(list(position) + list(orientation), dtype=torch.float32)
        
        return images_tensor, pose
