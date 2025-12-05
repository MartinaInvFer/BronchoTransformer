# BronchoTransformer: Sequential Vision Transformer for 7-DoF Pose Estimation

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Este repositorio contiene la implementación oficial de **BronchoTransformer**, una arquitectura basada en **Vision Transformers (ViT) Secuenciales** diseñada para la estimación de pose en broncoscopia virtual.

A diferencia de los métodos tradicionales que utilizan redes recurrentes (RNNs) o restricciones geométricas explícitas, este modelo aprende dependencias temporales globales directamente de secuencias de video, logrando una **precisión de localización superior (3.12 mm)** y una mayor estabilidad sin necesidad de sensores externos.

---

## 📋 Tabla de Contenidos
- [Introducción](#introducción)
- [Preparación del Dataset](#preparación-del-dataset)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
  - [Entrenamiento](#entrenamiento)
  - [Evaluación](#evaluación)
- [Resultados](#resultados)
- [Referencias](#referencias)

---

## Introducción

El objetivo de este proyecto es estimar la pose de la cámara (Posición $x,y,z$ y Orientación $q_w, q_x, q_y, q_z$) dentro del árbol bronquial utilizando únicamente información visual.

El modelo utiliza un enfoque **End-to-End**:
1.  **Extractor Espacial:** Un ViT procesa cada *frame* individualmente.
2.  **Codificador Temporal:** Un Transformer Encoder modela la secuencia de movimiento.
3.  **Cabezal de Regresión:** Predice el vector de pose de 7 grados de libertad (7-DoF).

Para probar el modelo, es necesario descargar el dataset sintético del paper **BronchoPose** de Borrego et al. Puedes obtenerlo aquí:
[Descargar Dataset BronchoPose](https://dataverse.csuc.cat/dataset.xhtml?persistentId=doi:10.34810/data2251)

---

## Preparación del Dataset

El modelo espera una estructura de directorios específica basada en el dataset *VirtualNavigations*.

1.  **Descarga:** Baja los archivos del dataset desde el enlace superior. Nota que llegan en *split zips* (partes divididas).
2.  **Descompresión:** Une los archivos zip para extraer el contenido completo.
3.  **Organización:**
    * Crea una carpeta llamada `data` en la raíz de este repositorio.
    * Mueve la carpeta descomprimida `VirtualNavigations` dentro de `data`.
    
La estructura final debe verse así:
```text
BronchoTransformer/
├── data/
│   └── VirtualNavigations/
│       ├── LENS_P1_14_01_2016_INSP_CPAP/
│       │   ├── Frames/
│       │   └── P1_r1_to_3.csv
│       ├── ...
```
## Preparación del Dataset
1. Clonar el repositorio:
git clone [https://github.com/tu-usuario/BronchoTransformer.git](https://github.com/tu-usuario/BronchoTransformer.git)
cd BronchoTransformer
2. Instalar dependencias: Se recomienda usar un entorno virtual (conda o venv). Ejecuta el siguiente comando para instalar las librerías necesarias:
pip install torch torchvision timm numpy pandas matplotlib opencv-python tqdm

## Estructura del proyecto
