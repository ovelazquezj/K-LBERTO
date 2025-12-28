#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from uer.utils.vocab import Vocab
from uer.model_builder import build_model
from brain import KnowledgeGraph
import json

def load_dataset(path, kg, vocab, args, max_samples=50):
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
    model_path = "./outputs/kbert_tass_sentiment_88_SPANISH_v5_FINAL.bin"
    config_path = "./models/beto_uer_model/config.json"
    vocab_path = "./models/beto_uer_model/vocab.txt"
    test_path = "./datasets/tass_spanish/test.tsv"
    kg_name = "./brain/kgs/TASS_sentiment_88_SPANISH_CLEAN.spo"
    
    vocab = Vocab()
    vocab.load(vocab_path)
    
    with open(config_path) as f:
        config_dict = json.load(f)
    
    class Args:
        pass
    
    args = Args()
    
    for key, value in config_dict.items():
        setattr(args, key, value)
    
    args.target = "bert"
    args.labels_num = 4
    args.pooling = "first"
    args.vocab = vocab
    args.seq_length = 128
    args.subword_type = "none"
    args.sub_vocab_path = "models/sub_vocab.txt"
    args.subencoder = "avg"
    args.sub_layers_num = 2
    args.tokenizer = "bert"
    args.no_vm = False
    args.encoder = "bert"
    args.bidirectional = True
    
    model = build_model(args)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    kg = KnowledgeGraph(spo_files=[kg_name], predicate=True, word_level=True)
    
    dataset = load_dataset(test_path, kg, vocab, args, max_samples=50)
    
    print("=" * 80)
    print("DEBUG: Analizando primeros 50 ejemplos")
    print("=" * 80)
    
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx, (token_ids, label, mask, pos, vm, text, tokens) in enumerate(dataset):
            input_ids = torch.LongTensor([token_ids]).to(device)
            mask_ids = torch.LongTensor([mask]).to(device)
            pos_ids = torch.LongTensor([pos]).to(device)
            vm_ids = torch.BoolTensor([vm]).to(device)
            
            # Solo forward pass, sin loss
            output = model.encoder(input_ids, None, mask_ids, pos=pos_ids, vm=vm_ids)
            
            # Get logits from output (first token)
            logits = model.target.mlp(output[:, 0, :])
            
            probs = nn.Softmax(dim=1)(logits)
            pred = torch.argmax(probs, dim=1).item()
            
            logits_np = logits[0].cpu().numpy()
            probs_np = probs[0].cpu().numpy()
            
            all_logits.append(logits_np)
            all_probs.append(probs_np)
            all_preds.append(pred)
            all_labels.append(label)
            
            if idx < 10:
                print(f"\nEjemplo {idx}: {text[:50]}...")
                print(f"  Real: {label}, Pred: {pred}")
                print(f"  Logits: {logits_np}")
                print(f"  Probs: {probs_np}")
    
    all_logits = np.array(all_logits)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("\n" + "=" * 80)
    print("ANÁLISIS")
    print("=" * 80)
    
    print("\nDistribución PREDICCIONES:")
    for c in range(4):
        count = (all_preds == c).sum()
        print(f"  Clase {c}: {count}/50 ({100*count/50:.1f}%)")
    
    print("\nDistribución REALES:")
    for c in range(4):
        count = (all_labels == c).sum()
        print(f"  Clase {c}: {count}/50 ({100*count/50:.1f}%)")
    
    print("\nLOGITS por clase:")
    for c in range(4):
        print(f"  Clase {c}: media={all_logits[:, c].mean():.4f}, std={all_logits[:, c].std():.4f}, min={all_logits[:, c].min():.4f}, max={all_logits[:, c].max():.4f}")
    
    print("\nPROBABILIDADES por clase:")
    for c in range(4):
        print(f"  Clase {c}: media={all_probs[:, c].mean():.4f}")
    
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    print("\nMATRIZ DE CONFUSIÓN:")
    print(cm)
    
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds))
    
    acc = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nACCURACY: {acc:.4f}")
    
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO")
    print("=" * 80)
    
    if (all_preds == 0).sum() == 50:
        print("❌ COLAPSO TOTAL: Modelo predice SOLO clase 0")
        logit_0 = all_logits[:, 0].mean()
        logit_1 = all_logits[:, 1].mean()
        logit_2 = all_logits[:, 2].mean()
        logit_3 = all_logits[:, 3].mean()
        print(f"   Logit clase 0: {logit_0:.4f}")
        print(f"   Logit clase 1: {logit_1:.4f}")
        print(f"   Logit clase 2: {logit_2:.4f}")
        print(f"   Logit clase 3: {logit_3:.4f}")
        print(f"   MAX DIFF: {logit_0 - min(logit_1, logit_2, logit_3):.4f}")
    else:
        print("✓ Modelo hace predicciones variadas")

if __name__ == "__main__":
    main()
