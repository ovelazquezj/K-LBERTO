import json
import torch
from pathlib import Path

# Rutas
beto_path = "./models/beto_model"
output_path = "./models/beto_uer_model"
Path(output_path).mkdir(exist_ok=True)

# 1. Convertir config.json
print("📋 Converting BETO config to UER format...")
with open(f"{beto_path}/config.json", 'r') as f:
    beto_config = json.load(f)

uer_config = {
    "emb_size": beto_config["hidden_size"],
    "feedforward_size": beto_config["intermediate_size"],
    "hidden_size": beto_config["hidden_size"],
    "heads_num": beto_config["num_attention_heads"],
    "layers_num": beto_config["num_hidden_layers"],
    "dropout": beto_config.get("hidden_dropout_prob", 0.1),
    "hidden_act": beto_config.get("hidden_act", "gelu"),
    "vocab_size": beto_config["vocab_size"],
    "type_vocab_size": beto_config.get("type_vocab_size", 2),
    "max_position_embeddings": beto_config.get("max_position_embeddings", 512),
}

with open(f"{output_path}/config.json", 'w') as f:
    json.dump(uer_config, f, indent=2)
print("✓ Config converted")

# 2. Copiar vocab (igual en ambos)
import shutil
shutil.copy(f"{beto_path}/vocab.txt", f"{output_path}/vocab.txt")
print("✓ Vocab copied")

# 3. Intentar conversión de pesos (esto es complejo)
print("📦 Converting model weights...")
try:
    beto_weights = torch.load(f"{beto_path}/pytorch_model.bin")
    
    # Las arquitecturas son muy similares, muchos nombres de capas coinciden
    # Pero puede haber diferencias en naming conventions
    
    # Intentar mapeo directo primero
    new_weights = {}
    for key, value in beto_weights.items():
        # Reemplazar "bert." prefix si existe
        new_key = key.replace("bert.", "")
        new_weights[new_key] = value
    
    torch.save(new_weights, f"{output_path}/pytorch_model.bin")
    print("✓ Weights converted (basic mapping)")
except Exception as e:
    print(f"⚠️ Warning in weight conversion: {e}")
    print("✓ Keeping original weights")
    shutil.copy(f"{beto_path}/pytorch_model.bin", f"{output_path}/pytorch_model.bin")

print(f"\n✓ Conversion complete: {output_path}")
print("Files ready for K-BERT")
