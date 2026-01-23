#!/usr/bin/env python3
"""Run ablation experiments for Paper 2: KG Quality vs Data Curation"""
import json
import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path

# Configuración
BASE_DIR = Path.home() / "projects" / "K-LBERTO"
CONFIG_FILE = BASE_DIR / "experiments_ablation_kg_d4000.json"
RESULTS_DIR = BASE_DIR / "resultados_ablation_d4000"
PROGRESS_FILE = BASE_DIR / "ablation_progress_d4000.json"
LOG_FILE = BASE_DIR / "ablation_log_d4000.txt"

# Rutas fijas
PRETRAINED_MODEL = BASE_DIR / "models" / "beto_uer_model" / "pytorch_model.bin"
VOCAB_PATH = BASE_DIR / "models" / "beto_uer_model" / "vocab.txt"
CONFIG_PATH = BASE_DIR / "models" / "beto_uer_model" / "config.json"
DATA_DIR = BASE_DIR / "data" / "wikiann_tsv"
KG_DIR = BASE_DIR / "brain" / "kgs"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def run_experiment(exp, fixed_params):
    exp_id = exp["id"]
    output_dir = RESULTS_DIR / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python3", str(BASE_DIR / "run_klberto_ner.py"),
        "--pretrained_model_path", str(PRETRAINED_MODEL),
        "--output_model_path", str(output_dir / "model.bin"),
        "--vocab_path", str(VOCAB_PATH),
        "--config_path", str(CONFIG_PATH),
        "--train_path", str(DATA_DIR / exp["train"]),
        "--dev_path", str(DATA_DIR / fixed_params["dev_file"].split("/")[-1]),
        "--test_path", str(DATA_DIR / fixed_params["test_file"].split("/")[-1]),
        "--kg_name", str(KG_DIR / exp["kg"]),
        "--batch_size", str(fixed_params["batch_size"]),
        "--seq_length", str(fixed_params["max_seq_length"]),
        "--learning_rate", str(fixed_params["learning_rate"]),
        "--warmup", str(fixed_params["warmup_ratio"]),
        "--dropout", str(fixed_params["dropout"]),
        "--epochs_num", str(fixed_params["epochs"]),
        "--seed", str(exp["seed"]),
        "--experiment_name", exp_id,
        "--output_dir", str(output_dir)
    ]
    
    log(f"Ejecutando: {exp_id}")
    log(f"  Train: {exp['train']} | KG: {exp['kg']} | Seed: {exp['seed']}")
    
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    if result.returncode == 0:
        log(f"  ✅ Completado en {elapsed:.1f} min")
        return True
    else:
        log(f"  ❌ Error después de {elapsed:.1f} min")
        log(f"  STDERR: {result.stderr[-500:]}")
        return False

def main():
    log("=" * 60)
    log("INICIO: Ablation Study - KG Quality vs Data Curation")
    log("=" * 60)
    
    # Cargar configuración
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    
    fixed_params = config["fixed_params"]
    experiments = config["experiments"]
    progress = load_progress()
    
    log(f"Total experimentos: {len(experiments)}")
    log(f"Ya completados: {len(progress['completed'])}")
    log(f"Fallidos previos: {len(progress['failed'])}")
    
    # Crear directorio de resultados
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ejecutar experimentos pendientes
    for exp in experiments:
        exp_id = exp["id"]
        
        if exp_id in progress["completed"]:
            log(f"Saltando {exp_id} (ya completado)")
            continue
        
        success = run_experiment(exp, fixed_params)
        
        if success:
            progress["completed"].append(exp_id)
            if exp_id in progress["failed"]:
                progress["failed"].remove(exp_id)
        else:
            if exp_id not in progress["failed"]:
                progress["failed"].append(exp_id)
        
        save_progress(progress)
    
    # Resumen final
    log("=" * 60)
    log("RESUMEN FINAL")
    log(f"  Completados: {len(progress['completed'])}/{len(experiments)}")
    log(f"  Fallidos: {len(progress['failed'])}")
    log("=" * 60)

if __name__ == "__main__":
    main()
