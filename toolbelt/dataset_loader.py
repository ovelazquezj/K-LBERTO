#!/usr/bin/env python3
"""
Dataset Loader para WikiANN Spanish NER
Compatible con K-LBERTO
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class WikiANNNERDataset(Dataset):
    """Dataset loader para WikiANN Spanish en formato K-BERT"""
    
    def __init__(self, data_path, tokenizer, max_length=128, kg_path=None):
        """
        Args:
            data_path: Path al archivo JSON del subset
            tokenizer: BETO tokenizer
            max_length: Longitud máxima de secuencia
            kg_path: Path al knowledge graph (opcional para inferencia)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Cargar datos
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.tokens = data['tokens']
        self.ner_tags = data['ner_tags']
        
        # Mapeo de tags IOB2
        self.tag_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
        self.tag2id = {tag: idx for idx, tag in enumerate(self.tag_names)}
        self.id2tag = {idx: tag for idx, tag in enumerate(self.tag_names)}
        
        # Cargar KG si se proporciona (para K-BERT injection)
        self.kg = None
        if kg_path:
            self.kg = self._load_kg(kg_path)
    
    def _load_kg(self, kg_path):
        """Cargar knowledge graph en memoria"""
        kg_dict = {}
        with open(kg_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    subj, rel, obj = parts
                    if subj not in kg_dict:
                        kg_dict[subj] = []
                    kg_dict[subj].append((rel, obj))
        return kg_dict
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        """
        Retorna un sample procesado
        """
        tokens = self.tokens[idx]
        tags = self.ner_tags[idx]
        
        # Tokenización con BETO (puede generar subtokens)
        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Alinear tags con subtokens
        word_ids = tokenized.word_ids(batch_index=0)
        aligned_tags = []
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_tags.append(-100)  # Ignore padding/special tokens
            else:
                aligned_tags.append(tags[word_idx])
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_tags, dtype=torch.long),
            'tokens': tokens,  # Original tokens para referencia
        }
    
    def get_tag_name(self, tag_id):
        """Convertir tag_id a nombre"""
        return self.id2tag.get(tag_id, 'O')

def load_wikiann_splits(base_path, dataset_size, tokenizer, kg_path=None):
    """
    Carga train/val/test splits para un tamaño específico de dataset
    
    Args:
        base_path: data/curated_datasets/
        dataset_size: 500, 750, 1000, etc.
        tokenizer: BETO tokenizer
        kg_path: Path al KG
    
    Returns:
        train_dataset, val_dataset (usamos subset completo como train,
        validation/test de WikiANN raw para evaluación)
    """
    import os
    from datasets import load_from_disk
    
    # Cargar subset específico como training
    subset_path = os.path.join(base_path, f'subset_{dataset_size}.json')
    train_dataset = WikiANNNERDataset(subset_path, tokenizer, kg_path=kg_path)
    
    # Cargar validation set original de WikiANN
    wikiann_raw = load_from_disk('data/wikiann_spanish/raw')
    
    # Convertir validation split a nuestro formato
    val_tokens = wikiann_raw['validation']['tokens'][:1000]  # Primeros 1000
    val_tags = wikiann_raw['validation']['ner_tags'][:1000]
    
    # Guardar temporalmente
    val_data = {
        'tokens': val_tokens,
        'ner_tags': val_tags,
        'langs': [['es'] * len(t) for t in val_tokens]
    }
    
    val_path = os.path.join(base_path, 'validation_temp.json')
    with open(val_path, 'w') as f:
        json.dump(val_data, f)
    
    val_dataset = WikiANNNERDataset(val_path, tokenizer, kg_path=kg_path)
    
    return train_dataset, val_dataset

# Test del loader
if __name__ == "__main__":
    from transformers import BertTokenizer
    
    print("="*60)
    print("PROBANDO DATASET LOADER")
    print("="*60)
    
    # Cargar tokenizer BETO
    print("\n[1/3] Cargando BETO tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
    print("  ✓ Tokenizer cargado")
    
    # Cargar dataset pequeño
    print("\n[2/3] Cargando dataset subset_500...")
    dataset = WikiANNNERDataset(
        'data/curated_datasets/subset_500.json',
        tokenizer,
        kg_path='data/knowledge_graph/knowledge_graph.txt'
    )
    print(f"  ✓ Dataset cargado: {len(dataset)} samples")
    
    # Probar un sample
    print("\n[3/3] Probando sample...")
    sample = dataset[0]
    print(f"  - Input IDs shape: {sample['input_ids'].shape}")
    print(f"  - Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  - Labels shape: {sample['labels'].shape}")
    print(f"  - Original tokens: {sample['tokens'][:10]}...")
    print(f"  - KG loaded: {dataset.kg is not None}")
    
    print("\n" + "="*60)
    print("DATASET LOADER FUNCIONANDO CORRECTAMENTE")
    print("="*60)
