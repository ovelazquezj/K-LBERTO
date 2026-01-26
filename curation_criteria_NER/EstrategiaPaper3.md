# ESTRATEGIA DE INVESTIGACIÓN - PAPER 3
## Curation Criteria Impact on NER: A Systematic Study

**Fecha de creación:** 2026-01-25
**Autor:** Omar Francisco Velázquez Juárez
**Programa:** Doctorado en Ingeniería de la Información y del Conocimiento - UAH
**Directors:** Dr. García Cabot Antonio, Dra. García López Eva

---

## 1. MOTIVACIÓN Y CONTEXTO

### 1.1 Origen del Paper 3

Paper 2 descubrió que la curación de datos **perjudica** el rendimiento en NER:
- RAW > CUR: +4.7%, p<0.0001, d=9.08
- **Causa raíz identificada:** Distribution shift por criterio ">90% entidades"

Este hallazgo abre una pregunta de investigación más amplia:
> ¿Cuáles criterios de curación benefician NER y cuáles lo perjudican?

### 1.2 Conexión con Tesis Doctoral

Este paper servirá como **fundamento empírico** para la propuesta de arquitectura de la tesis:

```
Tesis: Arquitectura de KGs Distribuidos en Edge para NLP Español

Paper 1: K-LBERTO base + Sentiment (validación de KG curado)
    ↓
Paper 2: KG Ablation + NER (task-dependency, curation hurts)
    ↓
Paper 3: Curation Criteria Systematic Study (fundamento metodológico)
    ↓
Propuesta de Arquitectura: Sistema distribuido que adapta
curación y KG según tarea y distribución de datos en edge
```

### 1.3 Gap en la Literatura

La literatura actual asume que "datos limpios = mejor rendimiento", pero:
- No hay estudios sistemáticos de criterios de curación para NER
- No existe framework para evaluar impacto de cada criterio
- No hay guías de curación específicas para tareas token-level

---

## 2. PREGUNTAS DE INVESTIGACIÓN

| RQ | Pregunta | Tipo |
|----|----------|------|
| **RQ1** | ¿Cuál es el impacto individual de cada criterio de curación en NER? | Ablación |
| **RQ2** | ¿Existen interacciones entre criterios de curación? | Factorial |
| **RQ3** | ¿El impacto varía según el tipo de entidad (PER/LOC/ORG)? | Granularidad |
| **RQ4** | ¿Cómo afecta la alineación train-test a cada criterio? | Distribution |
| **RQ5** | ¿Se pueden predecir criterios óptimos según características del dataset? | Framework |

---

## 3. DISEÑO EXPERIMENTAL

### 3.1 Criterios de Curación a Evaluar

| ID | Criterio | Descripción | Hipótesis |
|----|----------|-------------|-----------|
| C1 | Tokens mínimos | Remover samples con <3 tokens | Neutral/Positivo |
| C2 | Sin entidades | Remover samples sin entidades | Positivo |
| C3 | BIO inconsistente | Remover tags I- sin B- previo | Positivo |
| C4 | Duplicados | Remover samples repetidos | Positivo |
| C5 | Alta densidad (>90%) | Remover samples casi puros de entidades | **Negativo** (confirmado) |
| C6 | Alta densidad (>80%) | Umbral más conservador | Negativo (esperado) |
| C7 | Alta densidad (>70%) | Umbral aún más conservador | A determinar |
| C8 | REDIRECCIÓN | Remover samples tipo redirect | Negativo (esperado) |
| C9 | Longitud máxima | Remover samples muy largos (>50 tokens) | Neutral |
| C10 | Balance de entidades | Equilibrar distribución PER/LOC/ORG | A determinar |

### 3.2 Diseño Factorial

**Experimento Principal: Ablación Individual**

```
Para cada criterio Ci:
  - Baseline: Sin ningún criterio (RAW)
  - Condición: Solo criterio Ci aplicado
  - Comparación: F1(Baseline) vs F1(Ci)

Total: 10 criterios × 5 seeds = 50 experimentos
```

**Experimento Secundario: Combinaciones**

```
Combinaciones seleccionadas:
  - C1+C2+C3+C4 (todos excepto densidad)
  - C1+C2+C3+C4+C5 (curación actual)
  - C1+C2+C3+C4+C8 (sin REDIRECCIÓN)
  - C4 only (solo duplicados)

Total: 4 combinaciones × 5 seeds = 20 experimentos
```

**Experimento Terciario: Múltiples Datasets**

```
Datasets:
  - WikiANN Spanish (actual)
  - CoNLL 2002 Spanish
  - (Opcional) CAPITEL Spanish NER

Total: 3 datasets × mejores condiciones = ~30 experimentos adicionales
```

### 3.3 Resumen de Experimentos

| Fase | Experimentos | Tiempo Estimado |
|------|--------------|-----------------|
| Ablación individual | 50 | ~8 días |
| Combinaciones | 20 | ~3 días |
| Multi-dataset | 30 | ~5 días |
| **Total** | **100** | **~16 días** |

---

## 4. MÉTRICAS Y ANÁLISIS

### 4.1 Métricas Primarias

- F1 Overall
- F1 por entidad (PER, LOC, ORG)
- Precision/Recall por entidad

### 4.2 Métricas de Distribution Shift

```python
def distribution_gap(train, test, feature):
    """Calcula gap de distribución para una característica"""
    train_pct = calculate_percentage(train, feature)
    test_pct = calculate_percentage(test, feature)
    return abs(train_pct - test_pct)
```

Features a medir:
- % REDIRECCIÓN
- Densidad promedio de entidades
- Distribución de longitud de secuencia
- Distribución de tipos de entidad

### 4.3 Análisis Estadístico

- ANOVA factorial para interacciones
- Pairwise t-tests con corrección Bonferroni
- Effect sizes (Cohen's d)
- Correlación gap-distribución vs F1

---

## 5. CONTRIBUCIONES ESPERADAS

### Contribución 1: Taxonomy of Curation Criteria Impact
Clasificación empírica de criterios en:
- Beneficiosos (mejoran F1)
- Neutrales (sin impacto significativo)
- Perjudiciales (reducen F1)

### Contribución 2: Distribution-Aware Curation Framework
Metodología para:
1. Analizar distribución de test/producción
2. Seleccionar criterios que mantengan alineación
3. Predecir impacto de curación antes de entrenar

### Contribución 3: Task-Specific Curation Guidelines
Guías específicas para NER que contrastan con guidelines genéricas de NLP.

### Contribución 4: Foundation for Adaptive Edge Systems
Base empírica para sistemas que adaptan curación según:
- Tarea (sentence-level vs token-level)
- Distribución de datos en deployment
- Recursos disponibles en edge

---

## 6. CONEXIÓN CON ARQUITECTURA DE TESIS

### 6.1 Propuesta de Arquitectura (Preview)

```
┌─────────────────────────────────────────────────────────────┐
│                    EDGE DEVICE (Jetson)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Data      │  │  Curation   │  │    KG Selection     │  │
│  │  Analyzer   │──│  Selector   │──│    Module           │  │
│  │             │  │ (Paper 3)   │  │   (Papers 1-2)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │               │                    │               │
│         ▼               ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    K-LBERTO Model                        ││
│  │           (Adapted for task + distribution)              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

Flujo:
1. Data Analyzer: detecta distribución de datos entrantes
2. Curation Selector: aplica criterios óptimos según Paper 3
3. KG Selection: elige KG apropiado según Papers 1-2
4. K-LBERTO: procesa con configuración adaptada
```

### 6.2 Valor del Paper 3 para la Tesis

| Componente de Tesis | Aporte de Paper 3 |
|---------------------|-------------------|
| Curation Selector | Framework empírico de selección |
| Data Analyzer | Métricas de distribution shift |
| Adaptive System | Base para decisiones en runtime |
| Edge Optimization | Criterios que no requieren reentrenamiento |

---

## 7. TIMELINE PROPUESTO

```
Fase 1: Preparación (1 semana)
├── Crear scripts de curación parametrizables
├── Preparar datasets adicionales (CoNLL)
└── Configurar pipeline de experimentos

Fase 2: Experimentos Principales (2.5 semanas)
├── Ablación individual (8 días)
├── Combinaciones (3 días)
└── Análisis preliminar

Fase 3: Multi-dataset (1 semana)
├── Experimentos CoNLL
└── Comparación cross-dataset

Fase 4: Análisis y Redacción (2 semanas)
├── Análisis estadístico completo
├── Framework de curation
└── Redacción del paper

Total: ~6-7 semanas
```

---

## 8. RECURSOS NECESARIOS

### Hardware
- Jetson Orin NX (disponible) - ~16 días de cómputo

### Datasets
- WikiANN Spanish (disponible)
- CoNLL 2002 Spanish (por descargar)
- CAPITEL (opcional, requiere solicitud)

### Software
- Scripts de curación (por desarrollar)
- Pipeline de análisis (extender de Paper 2)

---

## 9. RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| Resultados no concluyentes | Baja | Paper 2 ya mostró efectos claros |
| Tiempo de cómputo excede estimación | Media | Priorizar ablación individual |
| CoNLL no disponible | Baja | Usar solo WikiANN (suficiente) |
| Overlap con literatura existente | Baja | Búsqueda exhaustiva previa |

---

## 10. POSIBLES VENUES

| Venue | Tipo | Deadline típico | Fit |
|-------|------|-----------------|-----|
| ACL | Conferencia | Enero/Febrero | Alto |
| EMNLP | Conferencia | Mayo/Junio | Alto |
| NAACL | Conferencia | Diciembre | Alto |
| TACL | Journal | Rolling | Medio |
| LREC-COLING | Conferencia | Variable | Alto |

---

## 11. CHECKLIST PRE-APROBACIÓN DIRECTOR

- [ ] Revisar RQs con director
- [ ] Confirmar timeline es compatible con tesis
- [ ] Validar que Paper 3 aporta a propuesta de arquitectura
- [ ] Discutir prioridad vs otros trabajos de tesis
- [ ] Acordar venue objetivo

---

## 12. PRÓXIMOS PASOS INMEDIATOS

1. **Enviar email a director** con propuesta
2. **Esperar feedback** y ajustar según comentarios
3. **Preparar scripts** de curación parametrizable
4. **Descargar CoNLL 2002** Spanish

---

*Documento creado: 2026-01-25*
*Estado: PROPUESTA - Pendiente aprobación de director*
