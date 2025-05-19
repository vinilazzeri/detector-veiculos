import os
import shutil
import random
import cv2
import numpy as np

# =============================================
# CONFIGURAÇÕES INICIAIS
# =============================================

dataset_dir = "/home/vinicius_lazzeri/Documents/detector-veiculos/dataset"  # Caminho raiz do dataset
train_images_dir = os.path.join(dataset_dir, "train/images")              # Pasta de imagens de treino
train_labels_dir = os.path.join(dataset_dir, "train/labels")              # Pasta de labels de treino

augmentation_percentage = 0.25  # Percentual de imagens que serão aumentadas artificialmente

# Proporções para divisão do dataset
train_ratio = 0.75  # 75% treino (não é usado diretamente, pois o que sobra após val/test permanece como treino)
val_ratio = 0.20    # 20% validação
test_ratio = 0.05   # 5% teste

# =============================================
# FUNÇÕES DE AUGMENTATION BÁSICAS
# =============================================

def add_noise(image):
    """Adiciona ruído gaussiano à imagem para simular baixa qualidade de captura."""
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch)) * 255
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy

def add_blur(image):
    """Aplica desfoque gaussiano para simular perda de foco."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def adjust_lighting(image):
    """Altera brilho e contraste para simular variações de iluminação."""
    alpha = random.uniform(0.7, 1.3)  # Contraste
    beta = random.randint(-30, 30)   # Brilho
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# =============================================
# FUNÇÃO PRINCIPAL: PROCESSA E DIVIDE O DATASET
# =============================================

def process_and_split():
    # Lista imagens originais (sem sufixo "_aug")
    original_images = [f for f in os.listdir(train_images_dir) if f.endswith((".jpg", ".png")) and "_aug" not in f]
    num_original = len(original_images)
    print(f"📊 Imagens originais: {num_original}")

    # Define quantas imagens serão aumentadas
    num_augment = int(augmentation_percentage * num_original)
    augmented_count = 0

    # Aplica aumentos artificiais em imagens selecionadas aleatoriamente
    for img_file in random.sample(original_images, num_augment):
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(train_images_dir, img_file)
        label_path = os.path.join(train_labels_dir, f"{base_name}.txt")

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Seleciona aleatoriamente um tipo de aumento
        aug_img = random.choice([
            add_noise(img),
            add_blur(img),
            adjust_lighting(img)
        ])

        # Salva imagem aumentada
        aug_name = f"{base_name}_aug{augmented_count}.jpg"
        cv2.imwrite(os.path.join(train_images_dir, aug_name), aug_img)

        # Copia o label correspondente (se existir)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(train_labels_dir, f"{base_name}_aug{augmented_count}.txt"))

        augmented_count += 1

    print(f"🔧 Imagens aumentadas: {augmented_count}")

    # Reagrupa todas as imagens após aumento
    all_images = [f for f in os.listdir(train_images_dir) if f.endswith((".jpg", ".png"))]
    random.shuffle(all_images)

    # Define quantidade de imagens para validação e teste
    total_images = len(all_images)
    num_val = int(val_ratio * total_images)
    num_test = int(test_ratio * total_images)
    num_train = total_images - num_val - num_test  # Resto permanece no treino

    # Cria pastas para conjuntos de validação e teste
    splits = {
        "val": os.path.join(dataset_dir, "valid"),
        "test": os.path.join(dataset_dir, "test"),
    }
    for split in splits.values():
        os.makedirs(os.path.join(split, "images"), exist_ok=True)
        os.makedirs(os.path.join(split, "labels"), exist_ok=True)

    # Move as imagens e labels selecionadas para val/test
    for i, img_file in enumerate(all_images):
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(train_images_dir, img_file)
        label_path = os.path.join(train_labels_dir, f"{base_name}.txt")

        if i < num_val:
            split = "val"
        elif i < num_val + num_test:
            split = "test"
        else:
            continue  # Mantém no treino

        # Move imagem para pasta correspondente
        shutil.move(img_path, os.path.join(splits[split], "images", img_file))
        # Move label (se existir)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(splits[split], "labels", f"{base_name}.txt"))

    # Exibe estatísticas finais da divisão
    total_train_final = len(os.listdir(train_images_dir))
    total_val_final = len(os.listdir(os.path.join(splits["val"], "images")))
    total_test_final = len(os.listdir(os.path.join(splits["test"], "images")))

    print("\n✅ DIVISÃO FINALIZADA:")
    print(f"Train: {total_train_final} imagens ({total_train_final / total_images * 100:.1f}%)")
    print(f"Valid: {total_val_final} imagens ({total_val_final / total_images * 100:.1f}%)")
    print(f"Test:  {total_test_final} imagens ({total_test_final / total_images * 100:.1f}%)")

# =============================================
# EXECUÇÃO DIRETA DO SCRIPT
# =============================================
if __name__ == "__main__":
    print("=== PROCESSANDO DATASET ===")
    process_and_split()
