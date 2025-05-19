from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.train(
    data="../configs/data.yaml",  # Caminho para o arquivo YAML que define os caminhos das imagens e classes
    epochs=100,                   # Número total de épocas de treinamento (ciclos completos sobre o conjunto de dados)
    batch=4,                     # Tamanho do lote por iteração. Valores menores consomem menos VRAM, ideal para GPUs com 6GB
    imgsz=640,                    # Tamanho das imagens de entrada (redimensionadas). Maior = mais contexto, mas mais consumo de VRAM
    workers=6,                    # Número de subprocessos usados para carregar os dados (ajuste depende do hardware e SO)
    device="0",                   # Especifica a GPU a ser usada (ex: "0" = primeira GPU)
    amp=True,                     # Ativa Mixed Precision (float16), reduz uso de memória e acelera o treino se suportado
    patience=20,                 # Early stopping: para o treino se não houver melhora por 20 épocas consecutivas

    # Hiperparâmetros de otimização
    lr0=0.008,                    # Taxa de aprendizado inicial. Levemente acima do padrão (0.01) para acelerar a convergência
    optimizer="AdamW",           # Otimizador usado. AdamW oferece regularização de peso nativa (melhor que Adam puro em muitos casos)
    weight_decay=0.0003,         # Penaliza pesos grandes, ajudando na regularização e evitando overfitting
    warmup_epochs=3,             # Número de épocas de aquecimento (learning rate começa menor e cresce progressivamente)

    # Técnicas de data augmentation para melhorar generalização
    hsv_h=0.015,                 # Variação de matiz nas imagens (para robustez à iluminação)
    hsv_s=0.7,                   # Variação de saturação (forte para gerar diversidade visual)
    hsv_v=0.4,                   # Variação de brilho (valor). Moderado para simular diferentes exposições
    scale=0.5,                   # Escala aleatória nas imagens, permitindo o modelo aprender objetos em diferentes tamanhos
    degrees=2.0,                 # Rotação aleatória leve nas imagens, simula pequenas inclinações
    shear=1.0,                   # Distorção em forma de "cisalhamento". Moderado para simular deformações geométricas
)
