#!/usr/bin/env python3
"""
Script 4: Extraer Knowledge Graph de Wikidata
Extraer triplets relevantes para entities españolas (PER, LOC, ORG)
Target: 20,000-50,000 triplets
Autor: Omar Velázquez
Fecha: 2026-01-03
"""

import json
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import Counter, defaultdict
import time
import re

def clean_entity_name(name):
    """Limpia nombres de entities para formato K-BERT"""
    # Remover paréntesis y contenido
    name = re.sub(r'\([^)]*\)', '', name)
    # Reemplazar espacios con underscores
    name = name.strip().replace(' ', '_')
    # Remover caracteres especiales
    name = re.sub(r'[^\w\s-]', '', name)
    return name

def query_wikidata(sparql_query, endpoint="https://query.wikidata.org/sparql", max_retries=3):
    """Ejecuta query SPARQL en Wikidata con reintentos"""
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    
    for attempt in range(max_retries):
        try:
            results = sparql.query().convert()
            return results
        except Exception as e:
            print(f"    Intento {attempt + 1} falló: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"    ERROR: Query falló después de {max_retries} intentos")
                return None

def extract_spanish_persons():
    """Extrae personas españolas y latinoamericanas con sus propiedades"""
    print("\n[1/5] Extrayendo personas españolas/latinoamericanas...")
    
    query = """
    SELECT DISTINCT ?person ?personLabel ?property ?propertyLabel ?value ?valueLabel
    WHERE {
      # Personas de países hispanohablantes
      VALUES ?country {
        wd:Q29       # España
        wd:Q96       # México
        wd:Q717      # Venezuela
        wd:Q739      # Colombia
        wd:Q750      # Argentina
        wd:Q77       # Perú
        wd:Q298      # Chile
        wd:Q414      # Argentina
        wd:Q419      # Perú
        wd:Q241      # Cuba
      }
      
      ?person wdt:P31 wd:Q5 ;           # instance of human
              wdt:P27 ?country .         # country of citizenship
      
      # Propiedades relevantes
      VALUES ?property {
        wdt:P106  # occupation
        wdt:P735  # given name
        wdt:P734  # family name
        wdt:P569  # date of birth
        wdt:P19   # place of birth
        wdt:P69   # educated at
        wdt:P101  # field of work
      }
      
      ?person ?property ?value .
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
    }
    LIMIT 5000
    """
    
    results = query_wikidata(query)
    if not results:
        return []
    
    triplets = []
    for result in results["results"]["bindings"]:
        person = clean_entity_name(result["personLabel"]["value"])
        property_name = result["propertyLabel"]["value"].lower().replace(' ', '_')
        value = clean_entity_name(result["valueLabel"]["value"])
        
        triplets.append((person, property_name, value, "PER"))
    
    print(f"    ✓ Extraídos {len(triplets)} triplets de personas")
    return triplets

def extract_spanish_locations():
    """Extrae ubicaciones españolas y latinoamericanas"""
    print("\n[2/5] Extrayendo ubicaciones españolas/latinoamericanas...")
    
    query = """
    SELECT DISTINCT ?location ?locationLabel ?property ?propertyLabel ?value ?valueLabel
    WHERE {
      # Países hispanohablantes
      VALUES ?country {
        wd:Q29 wd:Q96 wd:Q717 wd:Q739 wd:Q750 wd:Q77 wd:Q298 wd:Q414 wd:Q419 wd:Q241
      }
      
      # Ciudades, regiones, etc.
      ?location wdt:P31/wdt:P279* wd:Q515 ;  # instance of city (or subclass)
                wdt:P17 ?country .            # country
      
      VALUES ?property {
        wdt:P36   # capital
        wdt:P17   # country
        wdt:P131  # located in
        wdt:P1082 # population
        wdt:P625  # coordinate location
      }
      
      ?location ?property ?value .
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
    }
    LIMIT 5000
    """
    
    results = query_wikidata(query)
    if not results:
        return []
    
    triplets = []
    for result in results["results"]["bindings"]:
        location = clean_entity_name(result["locationLabel"]["value"])
        property_name = result["propertyLabel"]["value"].lower().replace(' ', '_')
        
        # Manejar valores numéricos y strings
        if "value" in result["valueLabel"]:
            value = str(result["valueLabel"]["value"])
            if not value.replace('.', '').replace(',', '').isdigit():
                value = clean_entity_name(value)
        else:
            value = "unknown"
        
        triplets.append((location, property_name, value, "LOC"))
    
    print(f"    ✓ Extraídos {len(triplets)} triplets de ubicaciones")
    return triplets

def extract_spanish_organizations():
    """Extrae organizaciones españolas y latinoamericanas"""
    print("\n[3/5] Extrayendo organizaciones españolas/latinoamericanas...")
    
    query = """
    SELECT DISTINCT ?org ?orgLabel ?property ?propertyLabel ?value ?valueLabel
    WHERE {
      VALUES ?country {
        wd:Q29 wd:Q96 wd:Q717 wd:Q739 wd:Q750 wd:Q77 wd:Q298 wd:Q414 wd:Q419 wd:Q241
      }
      
      # Organizaciones
      VALUES ?orgType {
        wd:Q43229     # organization
        wd:Q4830453   # business
        wd:Q783794    # company
        wd:Q476028    # association
      }
      
      ?org wdt:P31/wdt:P279* ?orgType ;
           wdt:P17 ?country .
      
      VALUES ?property {
        wdt:P452  # industry
        wdt:P112  # founded by
        wdt:P571  # inception
        wdt:P159  # headquarters location
        wdt:P31   # instance of
      }
      
      ?org ?property ?value .
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
    }
    LIMIT 5000
    """
    
    results = query_wikidata(query)
    if not results:
        return []
    
    triplets = []
    for result in results["results"]["bindings"]:
        org = clean_entity_name(result["orgLabel"]["value"])
        property_name = result["propertyLabel"]["value"].lower().replace(' ', '_')
        value = clean_entity_name(result["valueLabel"]["value"])
        
        triplets.append((org, property_name, value, "ORG"))
    
    print(f"    ✓ Extraídos {len(triplets)} triplets de organizaciones")
    return triplets

def extract_common_relations():
    """Extrae relaciones comunes entre entities conocidas"""
    print("\n[4/5] Extrayendo relaciones comunes...")
    
    # Algunas entities muy conocidas para poblar más el grafo
    common_entities = {
        "PER": ["Pablo_Picasso", "Gabriel_García_Márquez", "Frida_Kahlo", 
                "Diego_Maradona", "Lionel_Messi", "Penélope_Cruz"],
        "LOC": ["Madrid", "Barcelona", "Buenos_Aires", "Ciudad_de_México", 
                "Lima", "Bogotá"],
        "ORG": ["Real_Madrid", "FC_Barcelona", "Banco_Santander", "Telefónica"]
    }
    
    query = """
    SELECT DISTINCT ?subject ?subjectLabel ?property ?propertyLabel ?object ?objectLabel
    WHERE {
      VALUES ?subject {
        wd:Q5593 wd:Q5878 wd:Q5589 wd:Q1410 wd:Q615 wd:Q2807 wd:Q170238 
        wd:Q1492 wd:Q1486 wd:Q1489 wd:Q8686 wd:Q1519
      }
      
      ?subject ?prop ?object .
      ?property wikibase:directClaim ?prop .
      
      FILTER(STRSTARTS(STR(?object), "http://www.wikidata.org/entity/Q"))
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
    }
    LIMIT 3000
    """
    
    results = query_wikidata(query)
    if not results:
        return []
    
    triplets = []
    for result in results["results"]["bindings"]:
        subject = clean_entity_name(result["subjectLabel"]["value"])
        property_name = result["propertyLabel"]["value"].lower().replace(' ', '_')
        obj = clean_entity_name(result["objectLabel"]["value"])
        
        triplets.append((subject, property_name, obj, "MISC"))
    
    print(f"    ✓ Extraídos {len(triplets)} triplets de relaciones comunes")
    return triplets

def filter_and_deduplicate(all_triplets, target_min=20000, target_max=50000):
    """Filtra y deduplicar triplets"""
    print("\n[5/5] Filtrando y deduplicando triplets...")
    
    # Deduplicar
    unique_triplets = list(set([(t[0], t[1], t[2]) for t in all_triplets]))
    
    print(f"    - Triplets antes de deduplicar: {len(all_triplets)}")
    print(f"    - Triplets después de deduplicar: {len(unique_triplets)}")
    
    # Si tenemos demasiados, tomar muestra estratificada
    if len(unique_triplets) > target_max:
        print(f"    - Reduciendo a {target_max} triplets...")
        # Mantener balance entre tipos
        import random
        random.seed(42)
        unique_triplets = random.sample(unique_triplets, target_max)
    
    # Si tenemos muy pocos, advertir
    if len(unique_triplets) < target_min:
        print(f"    ⚠ ADVERTENCIA: Solo {len(unique_triplets)} triplets (target: {target_min}-{target_max})")
        print(f"      Considerar ampliar queries o agregar más countries")
    else:
        print(f"    ✓ {len(unique_triplets)} triplets en rango objetivo")
    
    return unique_triplets

def extract_knowledge_graph():
    """Pipeline completo de extracción"""
    
    print("="*60)
    print("EXTRAYENDO KNOWLEDGE GRAPH DE WIKIDATA")
    print("="*60)
    print("\nNOTA: Esto puede tomar 5-10 minutos dependiendo de la conexión")
    
    all_triplets = []
    
    # Extraer por tipo de entity
    person_triplets = extract_spanish_persons()
    all_triplets.extend(person_triplets)
    time.sleep(1)  # Rate limiting
    
    location_triplets = extract_spanish_locations()
    all_triplets.extend(location_triplets)
    time.sleep(1)
    
    org_triplets = extract_spanish_organizations()
    all_triplets.extend(org_triplets)
    time.sleep(1)
    
    common_triplets = extract_common_relations()
    all_triplets.extend(common_triplets)
    
    # Filtrar y deduplicar
    final_triplets = filter_and_deduplicate(all_triplets)
    
    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS FINALES")
    print("="*60)
    
    entity_types = Counter()
    relations = Counter()
    entities = set()
    
    for subj, rel, obj in final_triplets:
        entities.add(subj)
        entities.add(obj)
        relations[rel] += 1
    
    print(f"\n  Total triplets: {len(final_triplets)}")
    print(f"  Unique entities: {len(entities)}")
    print(f"  Unique relations: {len(relations)}")
    print(f"\n  Top 10 relaciones:")
    for rel, count in relations.most_common(10):
        print(f"    - {rel}: {count}")
    
    # Guardar
    print("\n" + "="*60)
    print("GUARDANDO KNOWLEDGE GRAPH")
    print("="*60)
    
    # Formato JSON (para análisis)
    kg_data = {
        "triplets": [{"subject": s, "relation": r, "object": o} 
                     for s, r, o in final_triplets],
        "statistics": {
            "total_triplets": len(final_triplets),
            "unique_entities": len(entities),
            "unique_relations": len(relations),
            "top_relations": dict(relations.most_common(10))
        }
    }
    
    with open("data/knowledge_graph/wikidata_kg_raw.json", "w") as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)
    print("  ✓ Guardado JSON: data/knowledge_graph/wikidata_kg_raw.json")
    
    # Formato K-BERT (texto plano)
    with open("data/knowledge_graph/wikidata_kg.txt", "w") as f:
        for subj, rel, obj in final_triplets:
            f.write(f"{subj}\t{rel}\t{obj}\n")
    print("  ✓ Guardado K-BERT format: data/knowledge_graph/wikidata_kg.txt")
    
    # Entity vocabulary
    entities_sorted = sorted(list(entities))
    with open("data/knowledge_graph/entity_vocab.txt", "w") as f:
        for entity in entities_sorted:
            f.write(f"{entity}\n")
    print("  ✓ Guardado vocabulary: data/knowledge_graph/entity_vocab.txt")
    
    # Relation vocabulary
    relations_sorted = sorted(list(relations.keys()))
    with open("data/knowledge_graph/relation_vocab.txt", "w") as f:
        for relation in relations_sorted:
            f.write(f"{relation}\n")
    print("  ✓ Guardado relations: data/knowledge_graph/relation_vocab.txt")
    
    print("\n" + "="*60)
    print("EXTRACCIÓN COMPLETADA")
    print("="*60)
    
    return final_triplets

if __name__ == "__main__":
    try:
        extract_knowledge_graph()
    except KeyboardInterrupt:
        print("\n\nExtracción interrumpida por usuario")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
