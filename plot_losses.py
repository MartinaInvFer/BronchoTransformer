import numpy as np
import matplotlib.pyplot as plt
import os

LOSS_FILE = "training_losses_all_patients.npz"
OUTPUT_IMAGE = "loss_curve_all_patients_trimmed.png"

if not os.path.exists(LOSS_FILE):
    print(f"Error: No se encontró el archivo '{LOSS_FILE}'")
    print("Asegúrate de que 'train.py' se haya ejecutado y guardado el archivo.")
    exit()

data = np.load(LOSS_FILE)
train_loss = data['train_loss']
val_loss = data['val_loss']

# Definimos el número de épocas a omitir al principio (Esto por si la caída es muy abrupta)
EPOCAS_A_OMITIR = 1

# Recortamos los datos de las pérdidas y las épocas
train_loss_trimmed = train_loss[EPOCAS_A_OMITIR:]
val_loss_trimmed = val_loss[EPOCAS_A_OMITIR:]
epochs_trimmed = range(EPOCAS_A_OMITIR + 1, len(train_loss) + 1)


plt.figure(figsize=(10, 6))
plt.plot(epochs_trimmed, train_loss_trimmed, '#5D8AA8', label='Training losses')
plt.plot(epochs_trimmed, val_loss_trimmed, '#BF6B63', label='Validation losses')
plt.title('Loss curve', fontsize=17)
plt.xlabel('EPOCHS', fontsize=14)
plt.ylabel('Losses (MSE)', fontsize=14)
plt.legend(fontsize=16)
plt.grid(True)

plt.savefig(OUTPUT_IMAGE)

print(f"¡Gráfica de pérdidas recortada y guardada como '{OUTPUT_IMAGE}'!")
