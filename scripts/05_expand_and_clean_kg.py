#!/usr/bin/env python3
"""
Script 5: Ampliar y Limpiar Knowledge Graph
- Agregar más países y queries
- Limpiar relaciones URL → nombres
- Combinar con KG existente
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

import json
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import Counter
import time
import re

# Mapeo de propiedades Wikidata a nombres legibles
PROPERTY_MAPPING = {
    'P106': 'occupation',
    'P131': 'located_in',
    'P17': 'country',
    'P625': 'coordinates',
    'P1082': 'population',
    'P735': 'given_name',
    'P734': 'family_name',
    'P569': 'date_of_birth',
    'P19': 'place_of_birth',
    'P69': 'educated_at',
    'P101': 'field_of_work',
    'P36': 'capital',
    'P452': 'industry',
    'P112': 'founded_by',
    'P571': 'inception',
    'P159': 'headquarters',
    'P31': 'instance_of',
    'P27': 'citizenship',
    'P54': 'member_of_sports_team',
    'P641': 'sport',
    'P463': 'member_of',
    'P108': 'employer',
    'P136': 'genre',
    'P264': 'record_label',
    'P166': 'award_received',
    'P800': 'notable_work',
    'P937': 'work_location',
}

def clean_relation(relation_url):
    """Convierte URL de propiedad Wikidata a nombre legible"""
    # Extraer P#### de la URL
    match = re.search(r'/([Pp]\d+)/?$', relation_url)
    if match:
        prop_id = match.group(1).upper()
        return PROPERTY_MAPPING.get(prop_id, f"property_{prop_id.lower()}")
    # Si ya es texto limpio, retornar as-is
    if not relation_url.startswith('http'):
        return relation_url.lower().replace(' ', '_')
    return "unknown_relation"

def clean_entity_name(name):
    """Limpia nombres de entities para formato K-BERT"""
    name = re.sub(r'\([^)]*\)', '', name)
    name = name.strip().replace(' ', '_')
    name = re.sub(r'[^\w\s-]', '', name)
    return name

def query_wikidata(sparql_query, max_retries=3):
    """Ejecuta query SPARQL en Wikidata con reintentos"""
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    
    for attempt in range(max_retries):
        try:
            results = sparql.query().convert()
            return results
        except Exception as e:
            print(f"    Intento {attempt + 1} falló: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    ERROR: Query falló después de {max_retries} intentos")
                return None

def extract_organizations_simple():
    """Query simplificada para organizaciones - evitar timeouts"""
    print("\n[1/5] Extrayendo organizaciones (query simplificada)...")
    
    query = """
    SELECT DISTINCT ?org ?orgLabel ?property ?propertyLabel ?value ?valueLabel
    WHERE {
      # Solo empresas/compañías grandes conocidas
      ?org wdt:P31 wd:Q783794 .  # instance of company
      
      # Filtrar a empresas con Wikipedia en español
      ?org wdt:P17 ?country .
      VALUES ?country { wd:Q29 wd:Q96 wd:Q717 }  # España, México, Venezuela
      
      VALUES ?property {
        wdt:P452  # industry
        wdt:P159  # headquarters
        wdt:P31   # instance of
      }
      
      ?org ?property ?value .
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
    }
    LIMIT 2000
    """
    
    results = query_wikidata(query)
    if not results:
        return []
    
    triplets = []
    for result in results["results"]["bindings"]:
        org = clean_entity_name(result["orgLabel"]["value"])
        prop_label = result["propertyLabel"]["value"]
        value = clean_entity_name(result["valueLabel"]["value"])
        
        triplets.append((org, prop_label, value))
    
    print(f"    ✓ Extraídos {len(triplets)} triplets de organizaciones")
    return triplets

def extract_athletes():
    """Extraer deportistas hispanohablantes"""
    print("\n[2/5] Extrayendo deportistas...")
    
    query = """
    SELECT DISTINCT ?athlete ?athleteLabel ?property ?propertyLabel ?value ?valueLabel
    WHERE {
      ?athlete wdt:P31 wd:Q5 ;           # human
               wdt:P106 wd:Q2066131 .    # occupation: athlete
      
      ?athlete wdt:P27 ?country .
      VALUES ?country {
        wd:Q29 wd:Q96 wd:Q717 wd:Q739 wd:Q750 wd:Q77 wd:Q298 wd:Q414 wd:Q241
      }
      
      VALUES ?property {
        wdt:P641  # sport
        wdt:P54   # member of sports team
        wdt:P27   # citizenship
        wdt:P569  # date of birth
      }
      
      ?athlete ?property ?value .
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
    }
    LIMIT 3000
    """
    
    results = query_wikidata(query)
    if not results:
        return []
    
    triplets = []
    for result in results["results"]["bindings"]:
        athlete = clean_entity_name(result["athleteLabel"]["value"])
        prop_label = result["propertyLabel"]["value"]
        value = clean_entity_name(result["valueLabel"]["value"])
        
        triplets.append((athlete, prop_label, value))
    
    print(f"    ✓ Extraídos {len(triplets)} triplets de deportistas")
    return triplets

def extract_artists():
    """Extraer artistas (músicos, pintores, escritores)"""
    print("\n[3/5] Extrayendo artistas y escritores...")
    
    query = """
    SELECT DISTINCT ?artist ?artistLabel ?property ?propertyLabel ?value ?valueLabel
    WHERE {
      ?artist wdt:P31 wd:Q5 .
      
      ?artist wdt:P106 ?occupation .
      VALUES ?occupation {
        wd:Q483501   # artist
        wd:Q1028181  # painter  
        wd:Q36834    # composer
        wd:Q639669   # musician
        wd:Q482980   # author
      }
      
      ?artist wdt:P27 ?country .
      VALUES ?country {
        wd:Q29 wd:Q96 wd:Q717 wd:Q739 wd:Q750 wd:Q77 wd:Q298
      }
      
      VALUES ?property {
        wdt:P106  # occupation
        wdt:P136  # genre
        wdt:P800  # notable work
        wdt:P27   # citizenship
      }
      
      ?artist ?property ?value .
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
    }
    LIMIT 3000
    """
    
    results = query_wikidata(query)
    if not results:
        return []
    
    triplets = []
    for result in results["results"]["bindings"]:
        artist = clean_entity_name(result["artistLabel"]["value"])
        prop_label = result["propertyLabel"]["value"]
        value = clean_entity_name(result["valueLabel"]["value"])
        
        triplets.append((artist, prop_label, value))
    
    print(f"    ✓ Extraídos {len(triplets)} triplets de artistas")
    return triplets

def extract_more_locations():
    """Extraer más ubicaciones con más países"""
    print("\n[4/5] Extrayendo más ubicaciones (países adicionales)...")
    
    query = """
    SELECT DISTINCT ?location ?locationLabel ?property ?propertyLabel ?value ?valueLabel
    WHERE {
      VALUES ?country {
        wd:Q29 wd:Q96 wd:Q717 wd:Q739 wd:Q750 wd:Q77 wd:Q298 wd:Q414 wd:Q419 wd:Q241
        wd:Q774 wd:Q790 wd:Q800 wd:Q804 wd:Q811  # Guatemala, Haiti, Honduras, El Salvador, Nicaragua
      }
      
      ?location wdt:P31/wdt:P279* wd:Q515 ;
                wdt:P17 ?country .
      
      VALUES ?property {
        wdt:P36   # capital
        wdt:P17   # country
        wdt:P131  # located in
      }
      
      ?location ?property ?value .
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
    }
    LIMIT 3000
    """
    
    results = query_wikidata(query)
    if not results:
        return []
    
    triplets = []
    for result in results["results"]["bindings"]:
        location = clean_entity_name(result["locationLabel"]["value"])
        prop_label = result["propertyLabel"]["value"]
        value = clean_entity_name(result["valueLabel"]["value"])
        
        triplets.append((location, prop_label, value))
    
    print(f"    ✓ Extraídos {len(triplets)} triplets de ubicaciones")
    return triplets

def load_and_clean_existing_kg():
    """Cargar KG existente y limpiar relaciones"""
    print("\n[5/5] Cargando y limpiando KG existente...")
    
    with open("data/knowledge_graph/wikidata_kg_raw.json", "r") as f:
        existing_kg = json.load(f)
    
    cleaned_triplets = []
    for triplet in existing_kg["triplets"]:
        subj = triplet["subject"]
        rel = clean_relation(triplet["relation"])
        obj = triplet["object"]
        cleaned_triplets.append((subj, rel, obj))
    
    print(f"    ✓ Limpiados {len(cleaned_triplets)} triplets existentes")
    return cleaned_triplets

def expand_and_clean_kg():
    """Pipeline completo"""
    
    print("="*60)
    print("AMPLIANDO Y LIMPIANDO KNOWLEDGE GRAPH")
    print("="*60)
    print("\nNOTA: Esto puede tomar 10-15 minutos")
    
    all_triplets = []
    
    # Cargar y limpiar existente
    existing_triplets = load_and_clean_existing_kg()
    all_triplets.extend(existing_triplets)
    time.sleep(1)
    
    # Nuevas extracciones
    org_triplets = extract_organizations_simple()
    all_triplets.extend(org_triplets)
    time.sleep(2)
    
    athlete_triplets = extract_athletes()
    all_triplets.extend(athlete_triplets)
    time.sleep(2)
    
    artist_triplets = extract_artists()
    all_triplets.extend(artist_triplets)
    time.sleep(2)
    
    location_triplets = extract_more_locations()
    all_triplets.extend(location_triplets)
    
    # Deduplicar
    print("\n" + "="*60)
    print("DEDUPLICANDO Y CONSOLIDANDO")
    print("="*60)
    
    print(f"\n  Triplets antes de deduplicar: {len(all_triplets)}")
    unique_triplets = list(set(all_triplets))
    print(f"  Triplets después de deduplicar: {len(unique_triplets)}")
    
    # Si sobrepasamos 50k, recortar
    if len(unique_triplets) > 50000:
        print(f"  Recortando a 50,000 triplets (máximo target)...")
        import random
        random.seed(42)
        unique_triplets = random.sample(unique_triplets, 50000)
    
    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS FINALES")
    print("="*60)
    
    relations = Counter()
    entities = set()
    
    for subj, rel, obj in unique_triplets:
        entities.add(subj)
        entities.add(obj)
        relations[rel] += 1
    
    print(f"\n  Total triplets: {len(unique_triplets)}")
    print(f"  Unique entities: {len(entities)}")
    print(f"  Unique relations: {len(relations)}")
    print(f"\n  Top 10 relaciones:")
    for rel, count in relations.most_common(10):
        print(f"    - {rel}: {count}")
    
    # Guardar
    print("\n" + "="*60)
    print("GUARDANDO KNOWLEDGE GRAPH EXPANDIDO")
    print("="*60)
    
    # JSON
    kg_data = {
        "triplets": [{"subject": s, "relation": r, "object": o} 
                     for s, r, o in unique_triplets],
        "statistics": {
            "total_triplets": len(unique_triplets),
            "unique_entities": len(entities),
            "unique_relations": len(relations),
            "top_relations": dict(relations.most_common(10))
        },
        "sources": ["existing_kg", "organizations", "athletes", "artists", "locations"]
    }
    
    with open("data/knowledge_graph/wikidata_kg_expanded.json", "w") as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)
    print("  ✓ Guardado JSON: data/knowledge_graph/wikidata_kg_expanded.json")
    
    # K-BERT format
    with open("data/knowledge_graph/wikidata_kg_clean.txt", "w") as f:
        for subj, rel, obj in unique_triplets:
            f.write(f"{subj}\t{rel}\t{obj}\n")
    print("  ✓ Guardado K-BERT: data/knowledge_graph/wikidata_kg_clean.txt")
    
    # Vocabularies
    entities_sorted = sorted(list(entities))
    with open("data/knowledge_graph/entity_vocab_expanded.txt", "w") as f:
        for entity in entities_sorted:
            f.write(f"{entity}\n")
    print("  ✓ Guardado entities: data/knowledge_graph/entity_vocab_expanded.txt")
    
    relations_sorted = sorted(list(relations.keys()))
    with open("data/knowledge_graph/relation_vocab_expanded.txt", "w") as f:
        for relation in relations_sorted:
            f.write(f"{relation}\n")
    print("  ✓ Guardado relations: data/knowledge_graph/relation_vocab_expanded.txt")
    
    # Verificar target
    print("\n" + "="*60)
    if 20000 <= len(unique_triplets) <= 50000:
        print("✅ TARGET ALCANZADO")
        print(f"   {len(unique_triplets)} triplets (objetivo: 20k-50k)")
    elif len(unique_triplets) < 20000:
        print("⚠️  POR DEBAJO DEL TARGET")
        print(f"   {len(unique_triplets)} triplets (objetivo: 20k mínimo)")
    else:
        print("✅ SOBRE EL TARGET")
        print(f"   {len(unique_triplets)} triplets (recortado a 50k máximo)")
    print("="*60)

if __name__ == "__main__":
    try:
        expand_and_clean_kg()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por usuario")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
