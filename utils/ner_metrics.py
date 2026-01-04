#!/usr/bin/env python3
"""
Métricas NER por entity type
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

def calculate_ner_metrics(predictions, labels, labels_map, begin_ids):
    """
    Calcula métricas NER overall y por entity type
    
    Args:
        predictions: tensor de predicciones
        labels: tensor de labels gold
        labels_map: dict de {label_name: id}
        begin_ids: list de IDs que son B- tags
    
    Returns:
        dict con estructura:
        {
            'overall': {'precision': float, 'recall': float, 'f1': float},
            'PER': {'precision': float, 'recall': float, 'f1': float},
            'LOC': {...},
            'ORG': {...},
            'counts': {'correct': int, 'pred_entities': int, 'gold_entities': int}
        }
    """
    
    # Invertir labels_map para obtener nombres
    id2label = {v: k for k, v in labels_map.items()}
    
    # Contadores por tipo
    stats = {
        'overall': {'tp': 0, 'fp': 0, 'fn': 0},
        'PER': {'tp': 0, 'fp': 0, 'fn': 0},
        'LOC': {'tp': 0, 'fp': 0, 'fn': 0},
        'ORG': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    
    # Extraer entidades predichas y gold
    pred_entities = extract_entities(predictions, id2label, labels_map, begin_ids)
    gold_entities = extract_entities(labels, id2label, labels_map, begin_ids)
    
    # Calcular TP, FP, FN por tipo
    for entity_type in ['PER', 'LOC', 'ORG']:
        pred_type = [e for e in pred_entities if e['type'] == entity_type]
        gold_type = [e for e in gold_entities if e['type'] == entity_type]
        
        # TP: entidades en ambos (mismo span)
        pred_spans = set((e['start'], e['end']) for e in pred_type)
        gold_spans = set((e['start'], e['end']) for e in gold_type)
        
        tp = len(pred_spans & gold_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        
        stats[entity_type]['tp'] = tp
        stats[entity_type]['fp'] = fp
        stats[entity_type]['fn'] = fn
        
        stats['overall']['tp'] += tp
        stats['overall']['fp'] += fp
        stats['overall']['fn'] += fn
    
    # Calcular métricas
    metrics = {}
    for key in stats:
        tp = stats[key]['tp']
        fp = stats[key]['fp']
        fn = stats[key]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[key] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Agregar counts
    metrics['counts'] = {
        'correct': stats['overall']['tp'],
        'pred_entities': stats['overall']['tp'] + stats['overall']['fp'],
        'gold_entities': stats['overall']['tp'] + stats['overall']['fn']
    }
    
    return metrics


def extract_entities(sequence, id2label, labels_map, begin_ids):
    """
    Extrae entidades de una secuencia de labels
    
    Returns:
        list de dicts: [{'type': 'PER', 'start': int, 'end': int}, ...]
    """
    entities = []
    current_entity = None
    
    for i, label_id in enumerate(sequence):
        # Saltar padding y tokens especiales
        if label_id <= 0 or label_id == labels_map.get('[PAD]', 0) or label_id == labels_map.get('[ENT]', 1):
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        label = id2label.get(label_id, 'O')
        
        # B- tag: iniciar nueva entidad
        if label_id in begin_ids:
            # Guardar entity anterior si existe
            if current_entity:
                entities.append(current_entity)
            
            # Extraer tipo (B-PER → PER)
            entity_type = label.split('-')[1] if '-' in label else 'UNK'
            current_entity = {
                'type': entity_type,
                'start': i,
                'end': i
            }
        
        # I- tag: continuar entity actual
        elif label.startswith('I-') and current_entity:
            current_entity['end'] = i
        
        # O tag: terminar entity
        elif label == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Agregar última entity si existe
    if current_entity:
        entities.append(current_entity)
    
    return entities


# Test
if __name__ == "__main__":
    import torch
    
    print("="*60)
    print("TESTING NER Metrics")
    print("="*60)
    
    # Simular labels_map
    labels_map = {
        '[PAD]': 0, '[ENT]': 1, 'O': 2,
        'B-PER': 3, 'I-PER': 4,
        'B-ORG': 5, 'I-ORG': 6,
        'B-LOC': 7, 'I-LOC': 8
    }
    begin_ids = [3, 5, 7]  # B-PER, B-ORG, B-LOC
    
    # Simular predicciones y gold
    # Secuencia: O B-PER I-PER O B-LOC O B-ORG I-ORG O
    gold = torch.tensor([2, 3, 4, 2, 7, 2, 5, 6, 2])
    pred = torch.tensor([2, 3, 4, 2, 7, 2, 2, 2, 2])  # Predice bien PER y LOC, falla ORG
    
    metrics = calculate_ner_metrics(pred, gold, labels_map, begin_ids)
    
    print("\nMétricas calculadas:")
    for entity_type in ['overall', 'PER', 'LOC', 'ORG']:
        m = metrics[entity_type]
        print(f"{entity_type:8s}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")
    
    print(f"\nCounts: {metrics['counts']}")
    print("\n" + "="*60)
    print("TEST COMPLETADO")
    print("="*60)
