#!/usr/bin/env python3
"""
Convert entities_v1.csv to .spo format for K-BERT Spanish Knowledge Base
Creates Subject-Predicate-Object triples for geostatistical entities
"""

import csv
import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_kb_logs.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_spo_from_csv(csv_path, output_path):
    """Convert CSV entities to SPO (Subject-Predicate-Object) triples"""
    
    logger.info(f"Reading CSV: {csv_path}")
    
    spo_triples = []
    entity_count = 0
    triple_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            entity_name = row.get('entity_name', '').strip()
            entity_type = row.get('entity_type', '').strip()
            latitude = row.get('latitude', '').strip()
            longitude = row.get('longitude', '').strip()
            properties_json = row.get('properties', '{}').strip()
            
            if not entity_name:
                continue
            
            entity_count += 1
            
            # Add basic entity properties
            if entity_type:
                spo_triples.append((entity_name, 'tipo_entidad', entity_type))
                triple_count += 1
            
            if latitude and longitude:
                spo_triples.append((entity_name, 'ubicacion', f"{latitude},{longitude}"))
                triple_count += 1
                spo_triples.append((entity_name, 'latitud', latitude))
                triple_count += 1
                spo_triples.append((entity_name, 'longitud', longitude))
                triple_count += 1
            
            # Parse JSON properties
            try:
                if properties_json and properties_json != '{}':
                    props = json.loads(properties_json)
                    
                    for key, value in props.items():
                        if value and str(value).strip():
                            # Convert key to Spanish predicates
                            predicate_map = {
                                'codigo': 'codigo_oficial',
                                'tipo': 'tipo_via',
                                'highway': 'clasificacion_carretera',
                                'carriles': 'numero_carriles',
                                'velocidad_max': 'velocidad_maxima',
                            }
                            
                            predicate = predicate_map.get(key, key)
                            spo_triples.append((entity_name, predicate, str(value)))
                            triple_count += 1
            
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON for {entity_name}: {e}")
                continue
    
    # Write SPO file
    logger.info(f"\nWriting SPO file: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for subject, predicate, obj in spo_triples:
            f.write(f"{subject}\t{predicate}\t{obj}\n")
    
    logger.info(f"  ✓ Created {triple_count} triples from {entity_count} entities")
    
    return entity_count, triple_count

def generate_statistics(spo_path):
    """Generate statistics about the knowledge base"""
    
    predicates = defaultdict(int)
    subjects = set()
    objects = defaultdict(int)
    
    with open(spo_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                subject, predicate, obj = parts[0], parts[1], parts[2]
                subjects.add(subject)
                predicates[predicate] += 1
                objects[obj] += 1
    
    return {
        'total_triples': sum(predicates.values()),
        'unique_subjects': len(subjects),
        'unique_predicates': len(predicates),
        'unique_objects': len(objects),
        'predicates': dict(sorted(predicates.items(), key=lambda x: x[1], reverse=True)),
    }

def main():
    logger.info("=" * 80)
    logger.info("Convert Geostatistical Entities to K-BERT Spanish Knowledge Base")
    logger.info("=" * 80)
    
    # Define paths
    csv_path = Path('./entities_v1.csv')
    kb_dir = Path('./brain/kgs')
    kb_dir.mkdir(parents=True, exist_ok=True)
    output_path = kb_dir / 'GeoSpanish.spo'
    
    # Check if CSV exists
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        logger.info(f"Please ensure entities_v1.csv is in the current directory")
        return False
    
    logger.info(f"Input CSV: {csv_path.absolute()}")
    logger.info(f"Output KB: {output_path.absolute()}\n")
    
    # Convert CSV to SPO
    entity_count, triple_count = create_spo_from_csv(csv_path, output_path)
    
    # Generate statistics
    logger.info("\n[STEP 2] Generating statistics...")
    stats = generate_statistics(output_path)
    
    logger.info(f"\nKnowledge Base Statistics:")
    logger.info(f"  Total triples: {stats['total_triples']}")
    logger.info(f"  Unique subjects (entities): {stats['unique_subjects']}")
    logger.info(f"  Unique predicates (relations): {stats['unique_predicates']}")
    logger.info(f"  Unique objects (values): {stats['unique_objects']}")
    
    logger.info(f"\nMost common predicates:")
    for pred, count in list(stats['predicates'].items())[:10]:
        logger.info(f"    {pred}: {count}")
    
    logger.info("\n" + "=" * 80)
    logger.info("KNOWLEDGE BASE CREATED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\nFile location: {output_path.absolute()}")
    logger.info(f"\nTo use with K-BERT NER training:")
    logger.info(f"  --kg_name ./brain/kgs/GeoSpanish.spo")
    logger.info(f"\nOr as named KG:")
    logger.info(f"  --kg_name GeoSpanish")
    logger.info("\n" + "=" * 80 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
