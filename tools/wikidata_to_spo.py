#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikidata Spanish Knowledge Graph Extractor
==========================================

Extrae 10K entidades de Wikidata con etiquetas en español y las convierte
al formato SPO (Subject-Predicate-Object) compatible con K-BERT.

Requisitos:
    pip install SPARQLWrapper requests

Uso:
    python3 wikidata_to_spo.py --output ./WikidataSpanish.spo --limit 10000
"""

import argparse
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import defaultdict
import time
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WikidataExtractor:
    """Extractor de entidades Wikidata a formato SPO"""
    
    def __init__(self, language='es', batch_size=1000):
        """
        Args:
            language: Código de idioma (default: 'es' para español)
            batch_size: Tamaño de batch para consultas SPARQL
        """
        self.endpoint = "https://query.wikidata.org/sparql"
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)
        self.language = language
        self.batch_size = batch_size
        self.triples = defaultdict(list)
        
    def _execute_query(self, query, retries=3):
        """Ejecuta query SPARQL con manejo robusto de errores y reintentos"""
        for attempt in range(retries):
            try:
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                return results.get('results', {}).get('bindings', [])
            except Exception as e:
                error_msg = str(e)
                # Si es error de caracteres de control, no reintentar
                if 'Invalid control character' in error_msg or 'JSON' in error_msg:
                    logger.debug(f"Error JSON (intento {attempt+1}/{retries}): {error_msg[:100]}")
                    if attempt < retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        logger.warning(f"Fallando query después de {retries} intentos")
                        return []
                else:
                    logger.error(f"Error ejecutando query: {e}")
                    time.sleep(2)
        return []
    
    def extract_entities_batch(self, offset=0, limit=10000):
        """
        Extrae entidades en español de Wikidata
        Query simplificado y robusto para evitar caracteres de control
        
        Args:
            offset: Número de entidades a saltar
            limit: Número máximo de entidades a extraer
        """
        logger.info(f"Extrayendo {limit} entidades de Wikidata (offset: {offset})...")
        
        # Query más simple que evita caracteres problemáticos
        query = f"""
        SELECT ?item ?itemLabel ?instance_ofLabel
        WHERE {{
            ?item rdfs:label ?itemLabel ;
                  wdt:P31 ?instance_of .
            ?instance_of rdfs:label ?instance_ofLabel .
            
            FILTER (lang(?itemLabel) = "{self.language}")
            FILTER (lang(?instance_ofLabel) = "{self.language}")
            FILTER (STRLEN(?itemLabel) > 1)
        }}
        LIMIT {limit}
        OFFSET {offset}
        """
        
        results = self._execute_query(query)
        logger.info(f"Se obtuvieron {len(results)} entidades")
        return results
    
    def extract_properties(self, qid):
        """
        Extrae propiedades básicas de una entidad específica
        
        Args:
            qid: ID de Wikidata (ej: Q123)
            
        Returns:
            dict con propiedades: {predicado: [valores]}
        """
        query = f"""
        SELECT ?predicate ?predicateLabel ?object ?objectLabel ?value
        WHERE {{
            wd:{qid} ?p ?statement .
            ?statement ?ps ?object .
            
            ?predicate wikibase:claim ?p ;
                       wikibase:statementProperty ?ps ;
                       rdfs:label ?predicateLabel .
            
            OPTIONAL {{ ?object rdfs:label ?objectLabel . }}
            OPTIONAL {{ 
                ?statement psv:P625 [ wikibase:geoLatitude ?lat ; wikibase:geoLongitude ?lon ] .
                BIND(CONCAT(STR(?lat), ",", STR(?lon)) AS ?value)
            }}
            
            FILTER (lang(?predicateLabel) = "{self.language}")
            FILTER (!BOUND(?objectLabel) || lang(?objectLabel) = "{self.language}")
        }}
        LIMIT 50
        """
        
        try:
            results = self._execute_query(query)
            properties = defaultdict(list)
            
            for result in results:
                predicate = result.get('predicateLabel', {}).get('value', '')
                obj = result.get('objectLabel', result.get('object', {})).get('value', '')
                
                if predicate and obj:
                    properties[predicate].append(obj)
            
            return dict(properties)
        except Exception as e:
            logger.warning(f"Error extrayendo propiedades de {qid}: {e}")
            return {}
    
    def extract_basic_properties(self, qid, item_label, instance_label):
        """
        Extrae propiedades básicas recomendadas para K-BERT
        Versión simplificada y más robusta
        
        Propiedades básicas:
        - tipo_entidad (instance_of)
        - ocupacion (P106)
        - ubicacion (P625)
        - lugar_nacimiento (P19)
        - pais (P17)
        """
        logger.debug(f"Extrayendo propiedades básicas de {qid}...")
        
        properties = defaultdict(list)
        
        # Agregar tipo de entidad (ya se agregó en process_entities)
        
        # Ocupación (P106)
        try:
            query = f"""
            SELECT DISTINCT ?valueLabel WHERE {{
                wd:{qid} wdt:P106 ?value .
                ?value rdfs:label ?valueLabel .
                FILTER (lang(?valueLabel) = "{self.language}")
            }}
            LIMIT 3
            """
            results = self._execute_query(query)
            for result in results:
                if 'valueLabel' in result:
                    val = result['valueLabel']['value']
                    if val:
                        properties['ocupacion'].append(val)
        except:
            pass
        
        # País (P17)
        try:
            query = f"""
            SELECT DISTINCT ?valueLabel WHERE {{
                wd:{qid} wdt:P17 ?value .
                ?value rdfs:label ?valueLabel .
                FILTER (lang(?valueLabel) = "{self.language}")
            }}
            LIMIT 1
            """
            results = self._execute_query(query)
            for result in results:
                if 'valueLabel' in result:
                    val = result['valueLabel']['value']
                    if val:
                        properties['pais'].append(val)
        except:
            pass
        
        # Lugar de nacimiento (P19)
        try:
            query = f"""
            SELECT DISTINCT ?valueLabel WHERE {{
                wd:{qid} wdt:P19 ?value .
                ?value rdfs:label ?valueLabel .
                FILTER (lang(?valueLabel) = "{self.language}")
            }}
            LIMIT 1
            """
            results = self._execute_query(query)
            for result in results:
                if 'valueLabel' in result:
                    val = result['valueLabel']['value']
                    if val:
                        properties['lugar_nacimiento'].append(val)
        except:
            pass
        
        time.sleep(0.05)
        return dict(properties)
    
    def process_entities(self, entities, limit=10000):
        """
        Procesa lista de entidades y extrae propiedades
        Versión simplificada que evita queries complejas
        
        Args:
            entities: Lista de resultados SPARQL
            limit: Máximo número de entidades a procesar
        """
        processed = 0
        
        for i, entity in enumerate(entities):
            if processed >= limit:
                break
            
            try:
                qid = entity['item']['value'].split('/')[-1]  # Extraer ID
                label = entity['itemLabel']['value']
                instance_label = entity.get('instance_ofLabel', {}).get('value', '')
                
                if not label or len(label) < 2:
                    continue
                
                logger.info(f"[{processed+1}/{limit}] Procesando: {label} ({qid})")
                
                # Agregar tipo de entidad básico
                if instance_label:
                    self.triples[label].append(('tipo_entidad', instance_label))
                
                # Intentar extraer propiedades básicas (con manejo de errores)
                properties = self.extract_basic_properties(qid, label, instance_label)
                
                for predicate, values in properties.items():
                    for value in values:
                        if value and len(str(value)) > 0:
                            self.triples[label].append((predicate, value))
                
                processed += 1
                
                # Rate limiting
                if processed % 10 == 0:
                    time.sleep(0.5)
                    logger.info(f"Progreso: {processed}/{limit} entidades procesadas")
            
            except Exception as e:
                logger.warning(f"Error procesando entidad {i}: {e}")
                continue
        
        logger.info(f"Total de entidades procesadas: {processed}")
    
    def save_spo(self, output_path):
        """
        Guarda triples en formato SPO (Subject-Predicate-Object)
        
        Formato:
        Entidad[TAB]Predicado[TAB]Valor
        
        Args:
            output_path: Ruta del archivo de salida
        """
        logger.info(f"Guardando {len(self.triples)} entidades en {output_path}...")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                triple_count = 0
                
                for entity, properties in sorted(self.triples.items()):
                    for predicate, value in properties:
                        # Limpiar caracteres especiales
                        entity_clean = str(entity).replace('\n', ' ').replace('\t', ' ')
                        predicate_clean = str(predicate).replace('\n', ' ').replace('\t', ' ')
                        value_clean = str(value).replace('\n', ' ').replace('\t', ' ')
                        
                        # Escribir triple
                        f.write(f"{entity_clean}\t{predicate_clean}\t{value_clean}\n")
                        triple_count += 1
            
            logger.info(f"✓ Guardado exitoso: {triple_count} triples en {output_path}")
            return triple_count
        
        except Exception as e:
            logger.error(f"Error guardando archivo: {e}")
            return 0
    
    def get_statistics(self):
        """Retorna estadísticas de los triples extraídos"""
        total_entities = len(self.triples)
        total_triples = sum(len(props) for props in self.triples.values())
        predicates = set()
        
        for props in self.triples.values():
            for pred, _ in props:
                predicates.add(pred)
        
        return {
            'total_entities': total_entities,
            'total_triples': total_triples,
            'unique_predicates': len(predicates),
            'predicates': sorted(predicates)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Extrae entidades de Wikidata y convierte a formato SPO para K-BERT'
    )
    parser.add_argument(
        '--output',
        default='./WikidataSpanish.spo',
        help='Ruta del archivo de salida (default: ./WikidataSpanish.spo)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10000,
        help='Número máximo de entidades a extraer (default: 10000)'
    )
    parser.add_argument(
        '--language',
        default='es',
        help='Código de idioma (default: es)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Wikidata Spanish KG Extractor para K-BERT")
    logger.info("=" * 70)
    logger.info(f"Parámetros:")
    logger.info(f"  - Idioma: {args.language}")
    logger.info(f"  - Límite: {args.limit} entidades")
    logger.info(f"  - Salida: {args.output}")
    logger.info("=" * 70)
    
    # Inicializar extractor
    extractor = WikidataExtractor(language=args.language)
    
    # Extraer entidades en batches
    logger.info("\n[PASO 1] Extrayendo entidades de Wikidata en batches...")
    all_entities = []
    batch_size = 500  # Batches pequeños para evitar errores
    num_batches = (args.limit + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        offset = batch_num * batch_size
        current_limit = min(batch_size, args.limit - len(all_entities))
        
        logger.info(f"\nBatch {batch_num + 1}/{num_batches} (offset: {offset}, limit: {current_limit})")
        entities = extractor.extract_entities_batch(offset=offset, limit=current_limit)
        
        if not entities:
            logger.warning(f"Batch {batch_num + 1} no retornó resultados, intentando siguiente...")
            if batch_num == 0:
                logger.error("Primer batch vacío, abortando")
                sys.exit(1)
            time.sleep(2)
            continue
        
        all_entities.extend(entities)
        time.sleep(1)  # Esperar entre batches
    
    if not all_entities:
        logger.error("No se extrajeron entidades. Verifica conexión a Wikidata.")
        sys.exit(1)
    
    # Procesar entidades
    logger.info(f"\n[PASO 2] Procesando {len(all_entities)} entidades...")
    extractor.process_entities(all_entities, limit=args.limit)
    
    # Guardar triples
    logger.info(f"\n[PASO 3] Guardando triples en formato SPO...")
    triple_count = extractor.save_spo(args.output)
    
    # Mostrar estadísticas
    logger.info(f"\n[PASO 4] Generando estadísticas...")
    stats = extractor.get_statistics()
    
    logger.info("=" * 70)
    logger.info("ESTADÍSTICAS FINALES:")
    logger.info("=" * 70)
    logger.info(f"Total de entidades: {stats['total_entities']}")
    logger.info(f"Total de triples: {stats['total_triples']}")
    logger.info(f"Predicados únicos: {stats['unique_predicates']}")
    logger.info(f"\nPredicados extraídos:")
    for pred in stats['predicates']:
        logger.info(f"  - {pred}")
    logger.info("=" * 70)
    logger.info("✓ Proceso completado exitosamente")


if __name__ == '__main__':
    main()
