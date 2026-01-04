#!/usr/bin/env python3
"""
Script 6: Limpieza Final de Relaciones
Convertir todas las URLs restantes a nombres legibles
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

import json
import re

# Mapeo completo de propiedades Wikidata
PROPERTY_MAPPING = {
    'P17': 'country',
    'P27': 'citizenship',
    'P31': 'instance_of',
    'P36': 'capital',
    'P54': 'member_of_sports_team',
    'P69': 'educated_at',
    'P101': 'field_of_work',
    'P106': 'occupation',
    'P108': 'employer',
    'P112': 'founded_by',
    'P131': 'located_in',
    'P136': 'genre',
    'P159': 'headquarters',
    'P166': 'award_received',
    'P19': 'place_of_birth',
    'P264': 'record_label',
    'P452': 'industry',
    'P463': 'member_of',
    'P569': 'date_of_birth',
    'P571': 'inception',
    'P625': 'coordinates',
    'P641': 'sport',
    'P734': 'family_name',
    'P735': 'given_name',
    'P800': 'notable_work',
    'P937': 'work_location',
    'P1082': 'population',
}

def clean_relation_url(relation):
    """Convierte cualquier formato URL a nombre legible"""
    # Si ya está limpio, retornar
    if not relation.startswith('http'):
        return relation
    
    # Intentar extraer P#### de varios formatos de URL
    patterns = [
        r'/([Pp]\d+)/?$',           # /P131/ o /P131
        r'/direct/([Pp]\d+)',       # /direct/P131
        r'prop/([Pp]\d+)',          # prop/P131
    ]
    
    for pattern in patterns:
        match = re.search(pattern, relation)
        if match:
            prop_id = match.group(1).upper()
            clean_name = PROPERTY_MAPPING.get(prop_id, f"property_{prop_id.lower()}")
            return clean_name
    
    # Si no se pudo parsear, usar fallback
    return "unknown_relation"

def cleanup_kg():
    """Limpieza final de todas las relaciones"""
    
    print("="*60)
    print("LIMPIEZA FINAL DE KNOWLEDGE GRAPH")
    print("="*60)
    
    # Cargar KG expandido
    print("\n[1/3] Cargando KG expandido...")
    with open("data/knowledge_graph/wikidata_kg_expanded.json", "r") as f:
        kg_data = json.load(f)
    
    print(f"  Total triplets cargados: {len(kg_data['triplets'])}")
    
    # Limpiar todas las relaciones
    print("\n[2/3] Limpiando relaciones...")
    cleaned_triplets = []
    url_relations_found = 0
    
    for triplet in kg_data['triplets']:
        subj = triplet['subject']
        rel_original = triplet['relation']
        obj = triplet['object']
        
        # Verificar si es URL
        if rel_original.startswith('http'):
            url_relations_found += 1
        
        # Limpiar relación
        rel_clean = clean_relation_url(rel_original)
        cleaned_triplets.append((subj, rel_clean, obj))
    
    print(f"  URLs encontradas y limpiadas: {url_relations_found}")
    
    # Deduplicar (por si alguna limpieza creó duplicados)
    unique_triplets = list(set(cleaned_triplets))
    print(f"  Triplets únicos después de limpieza: {len(unique_triplets)}")
    
    # Estadísticas finales
    from collections import Counter
    relations = Counter()
    entities = set()
    
    for subj, rel, obj in unique_triplets:
        entities.add(subj)
        entities.add(obj)
        relations[rel] += 1
    
    print("\n" + "="*60)
    print("ESTADÍSTICAS FINALES")
    print("="*60)
    print(f"\n  Total triplets: {len(unique_triplets)}")
    print(f"  Unique entities: {len(entities)}")
    print(f"  Unique relations: {len(relations)}")
    print(f"\n  Top 10 relaciones:")
    for rel, count in relations.most_common(10):
        print(f"    - {rel}: {count}")
    
    # Verificar que no quedan URLs
    url_check = sum(1 for rel in relations.keys() if rel.startswith('http'))
    if url_check > 0:
        print(f"\n  ⚠️ ADVERTENCIA: {url_check} relaciones todavía en formato URL")
    else:
        print(f"\n  ✅ ÉXITO: Todas las relaciones están limpias")
    
    # Guardar KG final
    print("\n[3/3] Guardando KG final...")
    
    # JSON
    kg_final = {
        "triplets": [{"subject": s, "relation": r, "object": o} 
                     for s, r, o in unique_triplets],
        "statistics": {
            "total_triplets": len(unique_triplets),
            "unique_entities": len(entities),
            "unique_relations": len(relations),
            "top_relations": dict(relations.most_common(10))
        },
        "version": "final_cleaned"
    }
    
    with open("data/knowledge_graph/wikidata_kg_final.json", "w") as f:
        json.dump(kg_final, f, ensure_ascii=False, indent=2)
    print("  ✓ Guardado JSON: data/knowledge_graph/wikidata_kg_final.json")
    
    # K-BERT format (este es el que usaremos)
    with open("data/knowledge_graph/knowledge_graph.txt", "w") as f:
        for subj, rel, obj in unique_triplets:
            f.write(f"{subj}\t{rel}\t{obj}\n")
    print("  ✓ Guardado K-BERT: data/knowledge_graph/knowledge_graph.txt")
    
    # Vocabularies
    entities_sorted = sorted(list(entities))
    with open("data/knowledge_graph/entity_vocab.txt", "w") as f:
        for entity in entities_sorted:
            f.write(f"{entity}\n")
    print("  ✓ Guardado entities: data/knowledge_graph/entity_vocab.txt")
    
    relations_sorted = sorted(list(relations.keys()))
    with open("data/knowledge_graph/relation_vocab.txt", "w") as f:
        for relation in relations_sorted:
            f.write(f"{relation}\n")
    print("  ✓ Guardado relations: data/knowledge_graph/relation_vocab.txt")
    
    print("\n" + "="*60)
    if 20000 <= len(unique_triplets) <= 50000:
        print("✅ TARGET ALCANZADO")
    elif len(unique_triplets) >= 18000:
        print("✅ CASI TARGET (>90%)")
    else:
        print("⚠️ POR DEBAJO DEL TARGET")
    print(f"   {len(unique_triplets):,} triplets")
    print("="*60)
    
    print("\n🎉 KNOWLEDGE GRAPH LISTO PARA USAR")
    print("   Archivo principal: data/knowledge_graph/knowledge_graph.txt")

if __name__ == "__main__":
    cleanup_kg()
