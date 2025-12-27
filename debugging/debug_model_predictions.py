#!/usr/bin/env python3
"""
Debug: ¿Por qué el modelo predice solo clase 0?
Analiza logits, softmax, y predicciones reales
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from uer.utils.vocab import Vocab
from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
import brain.config as config
from brain import KnowledgeGraph
import argparse

def load_dataset(path, kg, vocab, args, max_samples=50):
    """Cargar primeros N ejemplos de test"""
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            if i > max_samples:
                break
            
            line_data = line.strip().split('\t')
            label = int(line_data[0])
            text = '[CLS]' + line_data[1]
            
            tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
            tokens = tokens[0]
            pos = pos[0]
            vm = vm[0].astype("bool")
            
            token_ids = [vocab.get(t) for t in tokens]
            mask = [0 if t != '[PAD]' else 0 for t in tokens]
            
            dataset.append((token_ids, label, mask, pos, vm, text, tokens))
    
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./outputs/kbert_tass_sentiment_88_SPANISH_v5.bin", type=str)
    parser.add_argument("--config_path", default="./models/beto_uer_model/config.json", type=str)
    parser.add_argument("--vocab_path", default="./models/beto_uer_model/vocab.txt", type=str)
    parser.add_argument("--test_path", default="./datasets/tass_spanish/test.tsv", type=str)
    parser.add_argument("--kg_name", default="./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo", type=str)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    # Setup
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab
    
    # Load config
    import json
    with open(args.config_path) as f:
        config_dict = json.load(f)
    for key, value in config_dict.items():
        setattr(args, key, value)
    args.target = "bert"
    args.labels_num = 4
    args.pooling = "first"
    
    # Build model
    model = build_model(args)
    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load KG
    kg = KnowledgeGraph(spo_files=[args.kg_name], predicate=True, word_level=True)
    
    # Load first 50 test samples
    dataset = load_dataset(args.test_path, kg, vocab, args, max_samples=50)
    
    print("=" * 80)
    print("DEBUG: Analizando primeros 50 ejemplos de test")
    print("=" * 80)
    
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx, (token_ids, label, mask, pos, vm, text, tokens) in enumerate(dataset):
            # Preparar batch de 1
            input_ids = torch.LongTensor([token_ids]).to(device)
            mask_ids = torch.LongTensor([mask]).to(device)
            pos_ids = torch.LongTensor([pos]).to(device)
            vm_ids = torch.LongTensor([vm]).to(device)
            labels = torch.LongTensor([label]).to(device)
            
            # Forward pass
            loss, logits = model(input_ids, labels, mask_ids, pos=pos_ids, vm=vm_ids)
            
            # Get probabilities and predictions
            probs = nn.Softmax(dim=1)(logits)
            pred = torch.argmax(probs, dim=1).item()
            
            logits_np = logits[0].cpu().numpy()
            probs_np = probs[0].cpu().numpy()
            
            all_logits.append(logits_np)
            all_probs.append(probs_np)
            all_preds.append(pred)
            all_labels.append(label)
            
            # Print detalles para primeros 10
            if idx < 10:
                print(f"\nEjemplo {idx}:")
                print(f"  Texto: {text[:60]}...")
                print(f"  Tokens: {tokens[:10]}...")
                print(f"  Label real: {label}")
                print(f"  Logits: {logits_np}")
                print(f"  Probs:  {probs_np}")
                print(f"  Predicción: {pred}")
                print(f"  ¿Correcto?: {pred == label}")
    
    # Análisis estadístico
    all_logits = np.array(all_logits)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("\n" + "=" * 80)
    print("ANÁLISIS ESTADÍSTICO")
    print("=" * 80)
    
    print("\nDistribución de PREDICCIONES:")
    for c in range(4):
        count = (all_preds == c).sum()
        print(f"  Clase {c}: {count}/{len(all_preds)} ({100*count/len(all_preds):.1f}%)")
    
    print("\nDistribución de LABELS REALES:")
    for c in range(4):
        count = (all_labels == c).sum()
        print(f"  Clase {c}: {count}/{len(all_labels)} ({100*count/len(all_labels):.1f}%)")
    
    print("\nANÁLISIS DE LOGITS:")
    for c in range(4):
        print(f"  Clase {c} - Media: {all_logits[:, c].mean():.4f}, Std: {all_logits[:, c].std():.4f}, Min: {all_logits[:, c].min():.4f}, Max: {all_logits[:, c].max():.4f}")
    
    print("\nANÁLISIS DE PROBABILIDADES:")
    for c in range(4):
        print(f"  Clase {c} - Media: {all_probs[:, c].mean():.4f}, Std: {all_probs[:, c].std():.4f}")
    
    # Matriz de confusión
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    print("\nMATRIZ DE CONFUSIÓN (primeros 50):")
    print(cm)
    
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds))
    
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nACCURACY: {accuracy:.4f}")
    
    # Diagnóstico
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO")
    print("=" * 80)
    
    if (all_preds == 0).sum() == len(all_preds):
        print("❌ PROBLEMA CRÍTICO: Modelo predice SOLO clase 0 para TODO")
        print("   Esto indica colapso total del modelo")
        print("   Posibles causas:")
        print("     1. Logits están muy sesgados hacia clase 0")
        print("     2. Loss function no está balanceando clases")
        print("     3. Learning rate demasiado alto/bajo")
        print("     4. Modelo no está aprendiendo")
        
        # Verificar logits
        logit_0 = all_logits[:, 0].mean()
        other_logits = all_logits[:, 1:].mean()
        print(f"\n   Logit promedio clase 0: {logit_0:.4f}")
        print(f"   Logit promedio otras: {other_logits:.4f}")
        print(f"   Diferencia: {logit_0 - other_logits:.4f}")
        
        if logit_0 - other_logits > 5:
            print("\n   → Logits EXTREMADAMENTE sesgados hacia clase 0")
            print("   → El modelo no está aprendiendo discriminación")
    else:
        print("✓ Modelo está haciendo predicciones variadas (bien)")

if __name__ == "__main__":
    main()
