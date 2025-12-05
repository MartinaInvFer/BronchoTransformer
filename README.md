# BronchoTransformer
**Introducción**
Para probar el modelo, es necesario descargar el dataset sintético del paper BronchoPose de Borrego et al. Puedes obtenerlo aquí:
https://dataverse.csuc.cat/dataset.xhtml?persistentId=doi:10.34810/data2251

**Preparación del dataset**
1. Descarga los archivos del dataset, que llegan en split zips.
2. Une los archivos zip para obtener el dataset completo.
3. En caso de que uses otro tipo de dataset (sintético o real), realiza los cambios correspondientes en el código para adaptarlo.
4. El código también puede ajustarse según las capacidades computacionales de la máquina donde entrenarás el modelo.

# Uso con el dataset BronchoPose
**Instalación de librerías**
Ejecuta este comando para instalar todas las librerías necesarias:
pip install torch torchvision timm numpy pandas matplotlib opencv-python tqdm

**Organización de archivos**
Descarga todo el contenido y guárdalo en una carpeta dentro de tu proyecto en Visual Studio Code.

Dentro de esta carpeta crea una subcarpeta llamada data.

Arrastra la carpeta VirtualNavigations dentro de la carpeta data. Esta contiene todos los archivos de pacientes, imágenes y los archivos csv correspondientes.
