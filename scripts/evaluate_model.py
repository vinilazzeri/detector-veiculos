import numpy as np
from ultralytics import YOLO

# =============================================
# FUNÇÃO UTILITÁRIA
# =============================================
def to_scalar(x):
    """
    Converte valores NumPy para escalares do Python.
    
    Args:
        x (np.ndarray | float): Valor numérico ou array.
    
    Returns:
        float: Valor escalar.
    """
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else x.mean().item()
    return float(x)

# =============================================
# AVALIAÇÃO DO MODELO YOLOv8
# =============================================

# Caminho para o modelo treinado
model_path = "/home/vinicius_lazzeri/Documents/detector-veiculos/scripts/runs/detect/train/weights/best.pt"

# Inicializa o modelo
model = YOLO(model_path)

# Executa a avaliação no conjunto de teste
metrics = model.val(
    data="../configs/data.yaml",   # Caminho para o arquivo de configuração do dataset
    split='test',                  # Subconjunto a ser avaliado
    conf=0.25,                     # Threshold de confiança
    iou=0.6,                       # Threshold de IoU
    save=True,                     # Salvar previsões
    save_txt=False,               
    save_json=True                # Salvar métricas em JSON
)

# =============================================
# MÉTRICAS GERAIS
# =============================================
print("\n📊 Resultados (em média):")
print(f"Precision média:     {to_scalar(metrics.box.p):.4f}")
print(f"Recall média:        {to_scalar(metrics.box.r):.4f}")
print(f"mAP@0.5 média:       {to_scalar(metrics.box.map50):.4f}")
print(f"mAP@0.5:0.95 média:  {to_scalar(metrics.box.map):.4f}")

# =============================================
# MÉTRICAS POR CLASSE (Matriz de Confusão)
# =============================================
conf_matrix = metrics.confusion_matrix.matrix  # Matriz de confusão
class_names = metrics.names                    # Nomes das classes

print("\n🔍 Detalhamento por classe (Precision/Recall calculados):")
for class_idx, class_name in enumerate(class_names):
    tp = conf_matrix[class_idx, class_idx]                         # Verdadeiros Positivos
    fp = conf_matrix[:, class_idx].sum() - tp                      # Falsos Positivos
    fn = conf_matrix[class_idx, :].sum() - tp                      # Falsos Negativos

    # Cálculo das métricas com tratamento de divisão por zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Exibição formatada
    print(f"{class_name:<15} | Precision: {precision:.3f} | Recall: {recall:.3f} | TP: {tp} | FP: {fp} | FN: {fn}")
