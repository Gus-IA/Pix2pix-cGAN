# Pix2Pix GAN con TensorFlow

Este proyecto implementa un **Pix2Pix Conditional GAN** para la generación de imágenes usando TensorFlow y Keras.

---

## 📂 Estructura de carpetas

- `inputFlowers/` → Imágenes de entrada (input)  
- `targetFlowers/` → Imágenes objetivo (target)  
- `checkpoints/` → Carpeta donde se guardan los pesos entrenados del modelo  

---

## 📝 Descripción

El código implementa:

- **Generador:** U-Net para generar imágenes a partir de imágenes de entrada.  
- **Discriminador:** CNN que diferencia imágenes reales de generadas.  
- **Función de pérdida:** Combina pérdida GAN con L1 para preservar detalles.  
- **Data Augmentation:** Redimensionado, recorte aleatorio y flip horizontal.  

El modelo se entrena con **TensorFlow 2** usando `tf.data` para manejar batches de imágenes.

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
