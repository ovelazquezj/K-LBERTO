# Email de Seguimiento - Director de Investigación

**Para:** Dr. García Cabot Antonio, Dra. García López Eva
**De:** Omar Francisco Velázquez Juárez
**Asunto:** Actualización Paper 2 NER + Propuesta Paper 3 sobre Criterios de Curación
**Fecha:** 2026-01-25

---

Estimados Antonio y Eva,

Espero que se encuentren bien. Les escribo para darles una actualización sobre el avance del Paper 2 y presentarles una propuesta para un Paper 3 que surgió de los hallazgos recientes.

## 1. Estado del Paper 2: "When Curation Hurts"

### Experimentos Completados
- **30/30 experimentos** ejecutados exitosamente en Jetson Orin NX (~115 horas)
- Diseño factorial 2×3: (Data: Curado/Raw) × (KG: NoKG/Generic/Curated)
- 5 seeds por condición para robustez estadística

### Hallazgos Principales

| Hallazgo | Evidencia |
|----------|-----------|
| **KG no mejora NER** | NoKG ≥ Curated KG > Generic KG |
| **Raw > Curated** | +4.7%, p<0.0001, Cohen's d=9.08 |
| **Task-dependency** | KG ayuda en Sentiment pero no en NER |
| **Generic KG perjudica** | -2.7% vs baseline |

### Análisis de Causa Raíz (Nuevo)

Realizamos un análisis post-hoc para entender **por qué** los datos crudos superan a los curados. Descubrimos un **distribution shift**:

```
                    TEST     CUR(train)   RAW(train)
REDIRECCIÓN         36.9%    0.8%         27.7%
Alta densidad       50.8%    23.1%        62.1%
```

El criterio de curación ">90% entidades" eliminó patrones que constituyen el 37% del test set. El modelo CUR nunca aprendió estos patrones, mientras que RAW sí.

### Estado Actual
- **Análisis completado** - Estadístico, por entidad, y de errores
- **En redacción** - Estructurando secciones de resultados y discusión
- **Pendiente** - Generación de figuras (requiere Colab por incompatibilidad matplotlib en Jetson)

---

## 2. Propuesta: Paper 3 - "Curation Criteria Impact on NER"

El hallazgo del distribution shift abre una línea de investigación más amplia que me gustaría proponer:

### Motivación
Paper 2 identificó que **un criterio específico** de curación perjudicó el rendimiento. Sin embargo, no sabemos:
- ¿Cuáles otros criterios son beneficiosos o perjudiciales?
- ¿Existen interacciones entre criterios?
- ¿Las conclusiones generalizan a otros datasets?

### Preguntas de Investigación Propuestas
1. ¿Cuál es el impacto individual de cada criterio de curación en NER?
2. ¿Existen interacciones entre criterios de curación?
3. ¿El impacto varía según tipo de entidad (PER/LOC/ORG)?
4. ¿Se pueden predecir criterios óptimos según distribución del dataset?

### Diseño Experimental
- **10 criterios** de curación evaluados individualmente
- **4 combinaciones** de criterios
- **2-3 datasets** (WikiANN, CoNLL 2002, posiblemente CAPITEL)
- **~100 experimentos** totales (~16 días de cómputo)

### Contribuciones Esperadas
1. **Taxonomía empírica** de criterios (beneficioso/neutral/perjudicial)
2. **Framework de curación** distribution-aware
3. **Guías específicas** para NER vs otras tareas NLP

### Conexión con la Tesis

Este paper serviría como **fundamento metodológico** para mi propuesta de arquitectura de KGs distribuidos en edge:

```
Paper 1 (Sentiment) → Paper 2 (NER) → Paper 3 (Curation)
                                           ↓
                    Propuesta de Arquitectura Adaptativa
                    (Sistema que selecciona curación + KG
                     según tarea y distribución de datos)
```

El sistema adaptativo necesita saber **qué criterios aplicar según el contexto**, y Paper 3 proporcionaría esa base empírica.

### Timeline Estimado
- Preparación: 1 semana
- Experimentos: 2.5 semanas
- Multi-dataset: 1 semana
- Análisis y redacción: 2 semanas
- **Total: ~6-7 semanas**

---

## 3. Preguntas para Discusión

1. ¿Consideran viable esta línea de investigación para Paper 3?
2. ¿Hay prioridades en la tesis que debería considerar antes de iniciar?
3. ¿Tienen sugerencias sobre datasets adicionales a incluir?
4. ¿Qué venue considerarían apropiado? (ACL, EMNLP, LREC-COLING)

---

## 4. Próximos Pasos Propuestos

**Inmediatos (Paper 2):**
- Completar redacción de resultados y discusión
- Generar figuras en Colab
- Enviar borrador para revisión

**Pendiente aprobación (Paper 3):**
- Desarrollar scripts de curación parametrizable
- Descargar y preparar CoNLL 2002 Spanish
- Configurar pipeline de experimentos

---

Quedo atento a sus comentarios y disponible para una reunión si lo consideran necesario.

Saludos cordiales,

**Omar Francisco Velázquez Juárez**
Doctorando en Ingeniería de la Información y del Conocimiento
Universidad de Alcalá de Henares

---

*Adjuntos sugeridos:*
- `NER_RESEARCH/RESEARCH.md` - Documentación completa Paper 2
- `curation_criteria_NER/EstrategiaPaper3.md` - Propuesta detallada Paper 3
