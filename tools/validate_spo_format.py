#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validador de Formato SPO para K-BERT
====================================

Verifica que el archivo WikidataSpanish.spo sea compatible con el formato
esperado por K-BERT, comparándolo con GeoSpanish.spo.

Uso:
    python3 validate_spo_format.py \
        --spo_file ./WikidataSpanish.spo \
        --reference_file ./brain/kgs/GeoSpanish.spo \
        --sample_size 30
"""

import argparse
import sys
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SPOValidator:
    """Validador de archivos en formato SPO"""
    
    def __init__(self, spo_file, reference_file=None):
        self.spo_file = spo_file
        self.reference_file = reference_file
        self.triples = []
        self.statistics = {}
    
    def load_spo(self, filepath):
        """
        Carga archivo SPO
        
        Returns:
            list de tuples (entity, predicate, value)
        """
        try:
            triples = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    
                    if len(parts) != 3:
                        logger.warning(f"Línea {line_num}: esperaba 3 campos, obtuve {len(parts)}")
                        logger.warning(f"  Contenido: {line[:100]}")
                        continue
                    
                    entity, predicate, value = parts
                    triples.append((entity, predicate, value))
            
            logger.info(f"✓ Cargadas {len(triples)} triples de {filepath}")
            return triples
        
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {filepath}")
            return []
        except Exception as e:
            logger.error(f"Error cargando {filepath}: {e}")
            return []
    
    def validate_structure(self, triples):
        """
        Valida la estructura de los triples
        
        Returns:
            dict con resultados de validación
        """
        logger.info("\n[VALIDACIÓN 1] Verificando estructura...")
        
        results = {
            'valid_triples': 0,
            'invalid_triples': 0,
            'issues': []
        }
        
        for i, (entity, predicate, value) in enumerate(triples):
            is_valid = True
            
            # Verificar que no hay tabs
            if '\t' in entity or '\t' in predicate or '\t' in value:
                results['issues'].append(f"Triple {i}: contiene TABs en campos")
                is_valid = False
            
            # Verificar que no hay newlines
            if '\n' in entity or '\n' in predicate or '\n' in value:
                results['issues'].append(f"Triple {i}: contiene newlines")
                is_valid = False
            
            # Verificar campos no vacíos
            if not entity.strip() or not predicate.strip() or not value.strip():
                results['issues'].append(f"Triple {i}: campos vacíos")
                is_valid = False
            
            # Verificar longitud mínima
            if len(entity) < 1 or len(predicate) < 1 or len(value) < 1:
                results['issues'].append(f"Triple {i}: campos muy cortos")
                is_valid = False
            
            if is_valid:
                results['valid_triples'] += 1
            else:
                results['invalid_triples'] += 1
        
        return results
    
    def analyze_statistics(self, triples):
        """
        Analiza estadísticas de los triples
        
        Returns:
            dict con estadísticas
        """
        logger.info("\n[VALIDACIÓN 2] Analizando estadísticas...")
        
        stats = {
            'total_triples': len(triples),
            'unique_entities': set(),
            'unique_predicates': set(),
            'unique_values': set(),
            'predicates_count': defaultdict(int),
            'entities_count': defaultdict(int),
            'avg_triples_per_entity': 0
        }
        
        for entity, predicate, value in triples:
            stats['unique_entities'].add(entity)
            stats['unique_predicates'].add(predicate)
            stats['unique_values'].add(value)
            stats['predicates_count'][predicate] += 1
            stats['entities_count'][entity] += 1
        
        if stats['unique_entities']:
            stats['avg_triples_per_entity'] = len(triples) / len(stats['unique_entities'])
        
        return stats
    
    def compare_with_reference(self, test_triples, ref_triples):
        """
        Compara estructura con archivo de referencia (GeoSpanish.spo)
        
        Returns:
            dict con comparación
        """
        logger.info("\n[VALIDACIÓN 3] Comparando con archivo de referencia...")
        
        test_stats = self.analyze_statistics(test_triples)
        ref_stats = self.analyze_statistics(ref_triples)
        
        comparison = {
            'test_entities': len(test_stats['unique_entities']),
            'ref_entities': len(ref_stats['unique_entities']),
            'test_triples': test_stats['total_triples'],
            'ref_triples': ref_stats['total_triples'],
            'test_predicates': len(test_stats['unique_predicates']),
            'ref_predicates': len(ref_stats['unique_predicates']),
            'common_predicates': test_stats['unique_predicates'] & ref_stats['unique_predicates'],
            'format_compatible': True
        }
        
        return comparison, test_stats, ref_stats
    
    def show_samples(self, triples, sample_size=30):
        """Muestra muestras del archivo"""
        logger.info(f"\n[MUESTRA] Primeras {min(sample_size, len(triples))} líneas:")
        logger.info("=" * 100)
        
        for i, (entity, predicate, value) in enumerate(triples[:sample_size]):
            logger.info(f"{i+1}. {entity:40} | {predicate:30} | {value:20}")
            if (i + 1) % 10 == 0:
                logger.info("-" * 100)


def main():
    parser = argparse.ArgumentParser(
        description='Validador de formato SPO para K-BERT'
    )
    parser.add_argument(
        '--spo_file',
        required=True,
        help='Ruta del archivo WikidataSpanish.spo a validar'
    )
    parser.add_argument(
        '--reference_file',
        help='Ruta del archivo GeoSpanish.spo para comparación (opcional)'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=30,
        help='Número de muestras a mostrar (default: 30)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 100)
    logger.info("VALIDADOR DE FORMATO SPO PARA K-BERT")
    logger.info("=" * 100)
    
    # Validar archivo principal
    validator = SPOValidator(args.spo_file, args.reference_file)
    
    logger.info(f"\nCargando archivo: {args.spo_file}")
    test_triples = validator.load_spo(args.spo_file)
    
    if not test_triples:
        logger.error("✗ No se pudieron cargar triples del archivo")
        sys.exit(1)
    
    # Validar estructura
    struct_results = validator.validate_structure(test_triples)
    logger.info(f"✓ Triples válidas: {struct_results['valid_triples']}")
    logger.info(f"✗ Triples inválidas: {struct_results['invalid_triples']}")
    
    if struct_results['issues']:
        logger.warning("Problemas encontrados:")
        for issue in struct_results['issues'][:10]:
            logger.warning(f"  - {issue}")
    
    # Analizar estadísticas
    test_stats = validator.analyze_statistics(test_triples)
    logger.info(f"\nEstadísticas del archivo:")
    logger.info(f"  - Total de triples: {test_stats['total_triples']}")
    logger.info(f"  - Entidades únicas: {len(test_stats['unique_entities'])}")
    logger.info(f"  - Predicados únicos: {len(test_stats['unique_predicates'])}")
    logger.info(f"  - Promedio triples/entidad: {test_stats['avg_triples_per_entity']:.2f}")
    
    logger.info(f"\nPredicados encontrados:")
    for pred, count in sorted(test_stats['predicates_count'].items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {pred}: {count}")
    
    # Comparar con referencia si se proporciona
    if args.reference_file:
        logger.info(f"\nCargando archivo de referencia: {args.reference_file}")
        ref_triples = validator.load_spo(args.reference_file)
        
        if ref_triples:
            comparison, _, ref_stats = validator.compare_with_reference(test_triples, ref_triples)
            
            logger.info(f"\n[COMPARACIÓN]")
            logger.info(f"  Archivo de prueba:")
            logger.info(f"    - Entidades: {comparison['test_entities']}")
            logger.info(f"    - Triples: {comparison['test_triples']}")
            logger.info(f"    - Predicados: {comparison['test_predicates']}")
            
            logger.info(f"\n  Archivo de referencia (GeoSpanish.spo):")
            logger.info(f"    - Entidades: {comparison['ref_entities']}")
            logger.info(f"    - Triples: {comparison['ref_triples']}")
            logger.info(f"    - Predicados: {comparison['ref_predicates']}")
            
            logger.info(f"\n  Predicados en común: {len(comparison['common_predicates'])}")
            for pred in sorted(comparison['common_predicates']):
                logger.info(f"    - {pred}")
    
    # Mostrar muestras
    validator.show_samples(test_triples, args.sample_size)
    
    logger.info("\n" + "=" * 100)
    if struct_results['invalid_triples'] == 0:
        logger.info("✓ VALIDACIÓN EXITOSA - Archivo compatible con K-BERT")
    else:
        logger.warning("⚠ VALIDACIÓN CON ADVERTENCIAS - Revisar problemas arriba")
    logger.info("=" * 100)


if __name__ == '__main__':
    main()
