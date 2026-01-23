# ESTRATEGIA DE INVESTIGACIÓN - PAPER 2
## Ablación de Ruido de Conocimiento en K-LBERTO para NER Español

**Fecha de creación:** 2026-01-09  
**Última actualización:** 2026-01-11  
**Autor:** Omar Francisco Velázquez Juárez  
**Programa:** Doctorado en Ingeniería de la Información y del Conocimiento - UAH

---

## 1. CONTEXTO Y MOTIVACIÓN

### 1.1 Antecedentes

El Paper 1 ("Data Curation and Hyperparameter Scaling...") identificó que la curación de datos y el ajuste de parámetros son interdependientes. Sin embargo, no cuantificó:

1. Qué porcentaje de mejora se debe a la curación del dataset
2. Qué porcentaje se debe a la calidad del Knowledge Graph
3. Si existe interacción entre ambos factores

### 1.2 Propuesta del Director (Carencia #4)

> "Realizar un estudio de ablación del conocimiento, comparando el rendimiento de K-LBERTO usando un grafo genérico frente a uno especializado (curado) bajo la misma configuración de hiperparámetros. Esto aislaría el impacto de la 'calidad semántica' de la 'calidad de los datos de entrenamiento'."

### 1.3 Hallazgo Emergente (2026-01-11)

Los resultados preliminares (10/18 experimentos) muestran un patrón **inesperado**:

```
NoKG (0.6244) > KG Curado (0.6156) > KG Genérico (0.5645)
```

Esto contradice la expectativa de que KG mejora el rendimiento (basada en Paper 1 - Sentiment). La investigación ahora debe explicar **por qué** ocurre esto.

---

## 2. ESTADO ACTUAL DEL ESTUDIO

### 2.1 Progreso de Experimentos

| Condición | Completados | F1 Media | Std |
|-----------|-------------|----------|-----|
| CUR_NOKG | 3/3 ✅ | 0.6244 | 0.0127 |
| CUR_GEN | 3/3 ✅ | 0.5645 | 0.0988 |
| CUR_CUR | 3/3 ✅ | 0.6156 | 0.0114 |
| RAW_NOKG | 1/3 🔄 | 0.5458 | - |
| RAW_GEN | 0/3 ⏳ | - | - |
| RAW_CUR | 0/3 ⏳ | - | - |

**Total:** 10/18 (56%)

### 2.2 Resultados Individuales Completos

```
ABL_CUR_NOKG_s42:   F1=0.6236
ABL_CUR_NOKG_s123:  F1=0.6122
ABL_CUR_NOKG_s456:  F1=0.6375

ABL_CUR_GEN_s42:    F1=0.6290
ABL_CUR_GEN_s123:   F1=0.6127
ABL_CUR_GEN_s456:   F1=0.4518  ⚠️ OUTLIER

ABL_CUR_CUR_s42:    F1=0.6290
ABL_CUR_CUR_s123:   F1=0.6077
ABL_CUR_CUR_s456:   F1=0.6101

ABL_RAW_NOKG_s42:   F1=0.5458
```

### 2.3 Análisis Estadístico Preliminar

**ANOVA Factorial 2×3:**

| Source | SS | df | MS | F | p | η² |
|--------|-----|----|----|---|---|-----|
| Data | 0.0028 | 1 | 0.0028 | 0.56 | 0.768 | 0.097 |
| KG | 0.0044 | 2 | 0.0022 | 0.45 | 0.667 | **0.154** |
| Data×KG | 0.0018 | 2 | 0.0009 | 0.19 | 0.837 | 0.064 |

**Comparaciones Pareadas:**

| Comparación | Diff | t | p | Cohen's d |
|-------------|------|---|---|-----------|
| NoKG vs Generic | +0.060 | 1.05 | 0.293 | 0.86 |
| NoKG vs Curated | +0.009 | 0.89 | 0.374 | 0.73 |
| Generic vs Curated | -0.051 | -0.90 | 0.370 | -0.73 |

**Interpretación:** Ningún efecto significativo (p > 0.05), pero η² de KG = 0.154 (grande) y Cohen's d > 0.7 sugieren efectos sustanciales que podrían detectarse con más datos.

---

## 3. PREGUNTAS DE INVESTIGACIÓN

### 3.1 RQs Originales y Estado

| RQ | Pregunta | Comparación | Estado |
|----|----------|-------------|--------|
| RQ1 | ¿El KG genérico mejora sobre baseline? | NoKG vs Generic | ⚠️ Generic PEOR |
| RQ2 | ¿El KG curado mejora sobre baseline? | NoKG vs Curated | ⚠️ Curated ≈ NoKG (ligeramente peor) |
| RQ3 | ¿Calidad > Cantidad? | Generic vs Curated | ✅ Curated > Generic |
| RQ4 | ¿Cuánto aporta curación de datos? | CUR vs RAW | 🔄 Parcial |
| RQ5 | ¿Interacción Data×KG? | ANOVA | 🔄 Parcial |
| RQ6 | ¿KG compensa datos malos? | RAW+KG | ⏳ Pendiente |

### 3.2 RQ Emergente

| RQ7 | ¿Por qué KG beneficia Sentiment pero perjudica NER? | Cross-task | 🆕 Nueva |

---

## 4. REVISIÓN DE LITERATURA RELEVANTE

### 4.1 K-BERT Original (Liu et al. 2020)

**Hallazgo crítico - Knowledge Noise (KN):**

> "Too much knowledge incorporation may divert the sentence from its correct meaning, which is called knowledge noise (KN) issue."

> "K-BERT without visible matrix is worse than BERT, which confirms the existence of KN, i.e., improperly adding knowledge can lead to performance degradation."

**Resultados NER en K-BERT:**
- BERT baseline: 92.5%
- K-BERT + MedicalKG: 94.2% (+1.7%)

**Observación:** Mejoras modestas (1-2%), con KG específico de dominio, en datasets grandes.

### 4.2 KR-BERT (AAAI 2023)

**Hallazgo directo:**

> "We also test our triplet sourcing method within ERNIE and K-BERT and observe that even with triplet sourcing, the performance does not improve. Therefore an additional mechanism needs to filter relevant triplets."

**Implicación:** K-BERT **sin filtrado de relevancia** no mejora y puede degradar.

### 4.3 Frontiers in LLM-KG Fusion (2025)

**Sobre ruido en KG:**

> "High cost and scalability issues in KG construction... introduce noisy or redundant triples and suffer from low precision."

> "These issues not only degrade the reliability of the KG itself but also reduce the effectiveness of downstream KG-enhanced LLMs."

### 4.4 Recursos Adicionales Identificados

| Recurso | Relevancia |
|---------|------------|
| GitHub: entity-related-papers | Compilación de papers NER |
| GitHub: Low-resource-KEPapers | Papers extracción low-resource |
| Survey NER 2024 (arXiv) | Estado del arte NER |
| GLiNER (2024) | Alternativa span-based a KG injection |

---

## 5. ANÁLISIS CONCEPTUAL: TOKEN-LEVEL VS SENTENCE-LEVEL

### 5.1 Diferencias Fundamentales

| Aspecto | Sentiment (Sentence) | NER (Token) |
|---------|---------------------|-------------|
| **Granularidad** | 1 label por oración | 1 label por token |
| **Representación** | CLS token agregado | Embedding individual |
| **Objetivo** | Polaridad global | Boundaries exactos |
| **Dependencias** | Contexto global | Vecinos inmediatos |
| **Tolerancia a ruido** | Alta | Baja |

### 5.2 Mecanismo Propuesto de Degradación

**Entrada original:**
```
Tokens:  [Madrid]  [es]  [la]  [capital]  [de]  [España]
Tags:    B-LOC     O     O     O          O     B-LOC
```

**Con KG injection:**
```
Tokens:  [Madrid]  [capital]  [España]  [ciudad]  [es]  [la]  [capital]  [de]  [España]
                   ↑ injected  ↑ injected ↑ injected
```

**Problema:** Tokens inyectados interrumpen:
1. Posiciones relativas (soft-position puede no compensar completamente)
2. Dependencias CRF entre tokens consecutivos
3. Patterns aprendidos de boundaries

### 5.3 Por qué Sentiment NO Sufre Este Problema

En Sentiment:
- Se usa **CLS token** que agrega toda la secuencia
- Tokens adicionales **enriquecen** el contexto global
- El modelo no necesita preservar boundaries exactos
- Más información → mejor representación semántica

---

## 6. HIPÓTESIS A VALIDAR

### H1: Knowledge Noise Overwhelming Signal

**Predicción:** El efecto negativo de KG será mayor en datasets más pequeños.

**Test:** Comparar resultados con diferentes tamaños de training set (si tiempo permite).

### H2: Task-Level Mismatch

**Predicción:** La mejora en Sentiment no predice mejora en NER.

**Evidencia existente:** Paper 1 (Sentiment mejora) vs Paper 2 (NER no mejora).

### H3: KG Content Irrelevance

**Predicción:** El tipo de relaciones en el KG (relaciones semánticas entre entidades) no es útil para identificar token boundaries.

**Test:** Analizar qué entidades matchean y si las relaciones inyectadas son relevantes para boundary detection.

### H4: Outlier Artifact

**Predicción:** Sin el outlier s456, los resultados de CUR_GEN serían similares a CUR_NOKG.

**Test:** Re-calcular sin s456 y verificar convergencia en logs.

---

## 7. RECURSOS Y CONFIGURACIÓN

### 7.1 Knowledge Graphs

| KG | Triplets | Contenido | Ejemplo |
|----|----------|-----------|---------|
| empty.spo | 1 | Dummy | `DUMMY_ENTITY_NEVER_MATCH` |
| WikidataES_CLEAN | 3.4M | Multidominio | Relaciones diversas de Wikidata |
| wikidata_ner_spanish | 15k | NER-relevante | `(Pablo_Picasso, tiene_obras_en_la_colección, Currier_Museum_of_Art)` |

### 7.2 Datasets

| Dataset | Samples | Características |
|---------|---------|-----------------|
| train_2000.tsv | ~2000 | Curado (sin duplicados, sin >90% entities) |
| train_raw_2000.tsv | ~2000 | Raw (con ruido de WikiANN) |
| dev.tsv | - | Validación |
| test.tsv | - | Evaluación final |

### 7.3 Parámetros Fijos

| Parámetro | Valor |
|-----------|-------|
| Learning Rate | 2e-5 |
| Batch Size | 8 |
| Epochs | 10 |
| Seeds | 42, 123, 456 |

---

## 8. PRÓXIMOS PASOS

### 8.1 Inmediatos (Completar Experimentos)

1. [ ] Ejecutar ABL_RAW_NOKG_s123, s456
2. [ ] Ejecutar ABL_RAW_GEN_s42, s123, s456
3. [ ] Ejecutar ABL_RAW_CUR_s42, s123, s456
4. [ ] Re-ejecutar análisis con 18/18

### 8.2 Análisis Post-Experimentos

5. [ ] Investigar logs de ABL_CUR_GEN_s456 (outlier)
6. [ ] Calcular análisis con y sin outlier
7. [ ] Generar gráficas (interaction plot, heatmap)
8. [ ] Analizar F1 por tipo de entidad (PER/LOC/ORG)

### 8.3 Investigación Adicional

9. [ ] Verificar implementación de visible matrix
10. [ ] Calcular match rate entre entidades y KG
11. [ ] Buscar más literatura sobre KG degradation en NER

### 8.4 Documentación

12. [ ] Generar gráficas finales en Colab
13. [ ] Redactar sección de resultados
14. [ ] Preparar discusión con director

---

## 9. POSIBLES NARRATIVAS DEL PAPER

### Narrativa A: "Knowledge Noise in Low-Resource NER"

**Angle:** Documentar condiciones donde KG perjudica.

**Contribución:** Guía práctica de cuándo NO usar KG.

### Narrativa B: "Task-Dependent KG Effects"

**Angle:** Contrastar Sentiment vs NER con mismo KG.

**Contribución:** Framework teórico de token-level vs sentence-level.

### Narrativa C: "Quality Over Quantity in KG Selection"

**Angle:** Si confirmamos Curated > Generic consistentemente.

**Contribución:** Recomendaciones de selección de KG.

### Narrativa D: "Data Quality Dominates KG Quality"

**Angle:** Si efecto de Data >> efecto de KG en ANOVA final.

**Contribución:** Priorización de esfuerzos en deployment.

---

## 10. RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| Resultados RAW cambian conclusiones | Media | Completar antes de concluir |
| Outlier no es outlier real | Baja | Verificar logs, posible re-run |
| Sin significancia estadística | Media | Reportar effect sizes, aumentar seeds |
| Contradicción con literatura | Baja | Enmarcar como "condición de borde" |

---

## 11. COMANDOS ÚTILES

### Verificar progreso
```bash
cd ~/projects/K-LBERTO
cat ablation_progress.json | python3 -m json.tool
ls -la outputs/ablation/
```

### Continuar experimentos
```bash
python3 run_ablation_batch.py --resume
```

### Análisis
```bash
python3 analyze_ablation_results.py
cat ablation_analysis/summary_stats.csv
cat ablation_analysis/pairwise_comparisons.csv
```

### Verificar outlier
```bash
cat outputs/ablation/ABL_CUR_GEN_s456/training_log.txt | grep -E "loss|F1"
```

---

## 12. ARCHIVOS CLAVE

| Archivo | Propósito |
|---------|-----------|
| `run_ablation_batch.py` | Orquestador de experimentos |
| `analyze_ablation_results.py` | Análisis estadístico |
| `ablation_progress.json` | Estado de progreso |
| `ablation_analysis/*.csv` | Datos para gráficas |
| `BITACORA_P2_ABLACION_KG.md` | Log detallado |

---

## 13. CONTACTO Y SUPERVISIÓN

- **Director:** Pendiente comunicar hallazgos preliminares
- **Hallazgo crítico:** NoKG > KG contradice Paper 1
- **Acción requerida:** Discutir reorientación de narrative

---

*Última actualización: 2026-01-11 ~04:30 UTC*

---

## 14. ACTUALIZACIÓN 2026-01-22: NUEVOS PASOS POST-ANÁLISIS DE DILUCIÓN

### 14.1 Contexto

Durante el checkpoint 29/30, se detectó pérdida de significancia estadística en varias RQs debido a:
1. **RAW_GEN_s456:** Colapso catastrófico (F1=0.3458) que baja la media de RAW_GEN
2. **RAW_CUR_s789:** Resultado parcial (epoch 5/10) que distorsiona temporalmente la media

### 14.2 Decisión Metodológica: Opción B - Sensitivity Analysis

Se decidió **reportar ambos análisis** (con y sin outliers) siguiendo mejores prácticas de literatura (Bethard 2022, Dodge 2020).

### 14.3 Pasos Inmediatos (Próximas 6 horas)

| Paso | Descripción | ETA | Prioridad |
|------|-------------|-----|-----------|
| P1 | Esperar finalización RAW_CUR_s789 | ~80 min | CRÍTICO |
| P2 | Recalcular análisis con s789 completo | +15 min | CRÍTICO |
| P3 | Verificar si RQ4c recupera significancia | +5 min | CRÍTICO |
| P4 | Esperar finalización RAW_CUR_s2024 | ~4-5h | ALTO |
| P5 | Análisis final 30/30 completo | +30 min | ALTO |

### 14.4 Pasos de Análisis (Post 30/30)

| Paso | Descripción | Tiempo estimado |
|------|-------------|-----------------|
| A1 | Generar tabla sensitivity analysis (con/sin s456) | 1h |
| A2 | Calcular estadísticos robustos (mediana, IQR) | 30 min |
| A3 | Generar gráficas de interacción actualizadas | 1h |
| A4 | Análisis F1 por tipo de entidad (PER/LOC/ORG) | 2h |
| A5 | Documentar conclusiones ajustadas | 1h |

### 14.5 Estructura del Sensitivity Analysis para Paper

```
Appendix X: Sensitivity Analysis

Table X.1: Impact of Outlier Exclusion on Statistical Significance

Comparison          | With s456 (n=5) | Without s456 (n=4) |
--------------------|-----------------|---------------------|
RAW_GEN mean        | 0.5867 ± 0.121  | 0.6469 ± 0.014      |
RQ4b (GEN CUR→RAW)  | p=0.954, ns     | p<0.01, sig***      |
Conclusion          | No difference   | RAW > CUR           |

Interpretation:
The convergence failure of seed 456 in RAW+Generic condition
(F1=0.3458, >2 SD below mean) substantially affects RQ4b.
When excluded, the pattern RAW > CUR becomes consistent
across all KG conditions, supporting the hypothesis that
aggressive data filtering removes informative patterns.
```

### 14.6 Criterios de Decisión Post-30/30

**Escenario A: RQ4c recupera significancia (p<0.05)**
→ Narrativa fuerte: "RAW > CUR en 2/3 condiciones KG (NoKG, CUR_KG)"
→ Sensitivity analysis muestra 3/3 sin outlier

**Escenario B: RQ4c no recupera significancia (p>0.05)**
→ Narrativa moderada: "RAW > CUR confirmado solo para baseline NoKG"
→ Efecto de KG quality puede modular el efecto Data quality

**Escenario C: Nuevo outlier emerge en RAW_CUR_s2024**
→ Revisar patrones de convergencia
→ Posible problema sistemático con RAW+KG combinations

### 14.7 Actualización de RQs

| RQ | Estado Original | Estado Actual | Acción |
|----|-----------------|---------------|--------|
| RQ1 | ⚠️ Generic PEOR | ✅ Confirmado p<0.0001 | Mantener |
| RQ2 | ⚠️ Curated ≈ NoKG | ✅ Confirmado p=0.98 | Mantener |
| RQ3 | ✅ Curated > Generic | ✅ Confirmado p<0.0001 | Mantener |
| RQ4 | 🔄 Parcial | ✅ Confirmado p<0.0001 (NoKG) | Mantener |
| RQ4b | N/A | ❌ No sig (p=0.95) | Sensitivity |
| RQ4c | N/A | ❌ No sig (p=0.51) | Esperar 30/30 |
| RQ5 | 🔄 Parcial | ⏳ Pendiente ANOVA final | Esperar |
| RQ6 | ⏳ Pendiente | ❌ No sig (p=0.23) | Esperar |
| RQ7 | 🆕 Nueva | ✅ Task-dependency confirmado | Desarrollar |

### 14.8 Timeline Actualizado

```
2026-01-22 10:45  ← AHORA (checkpoint 29/30)
2026-01-22 12:00  → s789 completo, recálculo parcial
2026-01-22 16:00  → s2024 completo, análisis 30/30
2026-01-22 18:00  → Sensitivity analysis completo
2026-01-23        → Iniciar R2 (análisis PER/LOC/ORG)
2026-01-24        → Iniciar R3 (baseline GLiNER)
```

### 14.9 Checklist Pre-Submission

- [ ] 30/30 experimentos completos sin errores
- [ ] ANOVA factorial 2×3 con todos los datos
- [ ] Sensitivity analysis documentado
- [ ] Análisis por tipo de entidad (PER/LOC/ORG)
- [ ] Comparación con GLiNER baseline
- [ ] Figuras de interacción actualizadas
- [ ] Abstract revisado según hallazgos finales
- [ ] Título final seleccionado

### 14.10 Notas para Discusión con Director

**Puntos a comunicar:**
1. Hallazgo RAW > CUR es robusto para NoKG (p<0.0001)
2. Para condiciones con KG, el efecto es sensible a outliers
3. Seed 456 muestra patrón consistente de inestabilidad con Generic KG
4. Decisión metodológica: Opción B (sensitivity analysis)

**Preguntas pendientes:**
1. ¿Agregar más seeds para RAW+GEN para diluir outlier?
2. ¿Investigar causa raíz del colapso s456+Generic?
3. ¿Priorizar narrative A (Curation Hurts) o B (Task-Dependency)?
