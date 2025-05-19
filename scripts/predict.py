import os
import random
from ultralytics import YOLO

# Carrega o modelo YOLO treinado (melhor versão salva durante o treinamento)
model = YOLO("/home/vinicius_lazzeri/Documents/detector-veiculos/scripts/runs/detect/train/weights/best.pt")

# Define o caminho para o diretório de imagens de teste
test_img_dir = "../dataset/test/images"

# Lista todas as imagens com extensões suportadas (.jpg ou .png)
all_images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))]

# Seleciona aleatoriamente 12 imagens do conjunto de teste
selected_images = random.sample(all_images, 12)
selected_paths = [os.path.join(test_img_dir, f) for f in selected_images]

# Realiza a inferência com o modelo YOLO nas imagens selecionadas
results = model.predict(
    source=selected_paths,  # Lista de caminhos para as imagens selecionadas
    conf=0.5,                # Threshold de confiança mínima para considerar uma detecção válida
    save=True,               # Salva as imagens com as predições (bounding boxes, scores etc.)
    show=False               # Define se as imagens devem ser exibidas na tela (False = não exibe)
)
