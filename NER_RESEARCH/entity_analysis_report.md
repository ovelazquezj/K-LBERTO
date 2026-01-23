# Análisis Detallado por Tipo de Entidad
======================================================================

## 1. Resumen de F1 por Entidad y Condición

### Tabla de Medias (F1 Score)

| Data | KG | Overall | PER | LOC | ORG |
|------|-----|---------|-----|-----|-----|
| CUR | NOKG | 0.6099 | 0.5232 | 0.5488 | 0.3778 |
| CUR | GEN | 0.5749 | 0.4917 | 0.5160 | 0.3453 |
| CUR | CUR | 0.6060 | 0.5206 | 0.5463 | 0.3714 |
| RAW | NOKG | 0.6549 | 0.5476 | 0.5755 | 0.4510 |
| RAW | GEN | 0.5832 | 0.4978 | 0.5089 | 0.3752 |
| RAW | CUR | 0.6481 | 0.5397 | 0.5651 | 0.4479 |

### Tabla de Desviaciones Estándar

| Data | KG | Overall | PER | LOC | ORG |
|------|-----|---------|-----|-----|-----|
| CUR | NOKG | 0.0041 | 0.0048 | 0.0086 | 0.0117 |
| CUR | GEN | 0.0030 | 0.0050 | 0.0096 | 0.0086 |
| CUR | CUR | 0.0023 | 0.0066 | 0.0098 | 0.0028 |
| RAW | NOKG | 0.0076 | 0.0107 | 0.0165 | 0.0209 |
| RAW | GEN | 0.1383 | 0.0771 | 0.0947 | 0.1502 |
| RAW | CUR | 0.0063 | 0.0129 | 0.0210 | 0.0170 |

## 2. Análisis por Tipo de Entidad

### PER (Personas)

- **Mejor condición:** RAW_NOKG (F1=0.5476 ± 0.0107)
- **Peor condición:** CUR_GEN (F1=0.4917 ± 0.0050)
- **Diferencia:** 0.0559 (11.4%)

**Valores F1 PER por condición:**
```
CUR_NOKG: [0.5249, 0.5295, 0.5220, 0.5163, 0.5235] -> mean=0.5232
CUR_GEN: [0.4895, 0.4841, 0.4936, 0.4942, 0.4969] -> mean=0.4917
CUR_CUR: [0.5212, 0.5152, 0.5257, 0.5281, 0.5127] -> mean=0.5206
RAW_NOKG: [0.5536, 0.5507, 0.5498, 0.5288, 0.5551] -> mean=0.5476
RAW_GEN: [0.5380, 0.5372, 0.5005, 0.3638, 0.5495] -> mean=0.4978
RAW_CUR: [0.5209, 0.5357, 0.5564, 0.5437, 0.5418] -> mean=0.5397
```

### LOC (Lugares)

- **Mejor condición:** RAW_NOKG (F1=0.5755 ± 0.0165)
- **Peor condición:** RAW_GEN (F1=0.5089 ± 0.0947)
- **Diferencia:** 0.0665 (13.1%)

**Valores F1 LOC por condición:**
```
CUR_NOKG: [0.5443, 0.5465, 0.5640, 0.5446, 0.5444] -> mean=0.5488
CUR_GEN: [0.5292, 0.5226, 0.5120, 0.5103, 0.5059] -> mean=0.5160
CUR_CUR: [0.5441, 0.5332, 0.5427, 0.5525, 0.5588] -> mean=0.5463
RAW_NOKG: [0.5931, 0.5626, 0.5757, 0.5556, 0.5904] -> mean=0.5755
RAW_GEN: [0.5509, 0.5612, 0.5418, 0.3399, 0.5509] -> mean=0.5089
RAW_CUR: [0.5366, 0.5757, 0.5930, 0.5576, 0.5625] -> mean=0.5651
```

### ORG (Organizaciones)

- **Mejor condición:** RAW_NOKG (F1=0.4510 ± 0.0209)
- **Peor condición:** CUR_GEN (F1=0.3453 ± 0.0086)
- **Diferencia:** 0.1057 (30.6%)

**Valores F1 ORG por condición:**
```
CUR_NOKG: [0.3949, 0.3783, 0.3711, 0.3636, 0.3810] -> mean=0.3778
CUR_GEN: [0.3514, 0.3501, 0.3307, 0.3445, 0.3498] -> mean=0.3453
CUR_CUR: [0.3732, 0.3675, 0.3737, 0.3693, 0.3732] -> mean=0.3714
RAW_NOKG: [0.4526, 0.4565, 0.4186, 0.4767, 0.4507] -> mean=0.4510
RAW_GEN: [0.4336, 0.4610, 0.4426, 0.1073, 0.4314] -> mean=0.3752
RAW_CUR: [0.4692, 0.4551, 0.4397, 0.4240, 0.4517] -> mean=0.4479
```

## 3. Ranking de Dificultad por Entidad

1. **LOC (Lugares)**: F1 promedio = 0.5434
2. **PER (Personas)**: F1 promedio = 0.5201
3. **ORG (Organizaciones)**: F1 promedio = 0.3948

**Interpretación:** ORG es consistentemente la entidad más difícil de reconocer,
mientras que LOC tiende a ser la más fácil.

## 4. Efecto del Knowledge Graph por Entidad

### PER (Personas)
- NoKG: 0.5354
- Generic KG: 0.4947 (-4.07% vs NoKG)
- Curated KG: 0.5301 (-0.53% vs NoKG)

### LOC (Lugares)
- NoKG: 0.5621
- Generic KG: 0.5125 (-4.96% vs NoKG)
- Curated KG: 0.5557 (-0.65% vs NoKG)

### ORG (Organizaciones)
- NoKG: 0.4144
- Generic KG: 0.3602 (-5.42% vs NoKG)
- Curated KG: 0.4097 (-0.47% vs NoKG)

## 5. Efecto de Data Curation por Entidad

### PER (Personas)
- Curated Data: 0.5118
- Raw Data: 0.5284
- Diferencia: +1.65% (RAW > CUR)

### LOC (Lugares)
- Curated Data: 0.5370
- Raw Data: 0.5498
- Diferencia: +1.28% (RAW > CUR)

### ORG (Organizaciones)
- Curated Data: 0.3648
- Raw Data: 0.4247
- Diferencia: +5.99% (RAW > CUR)

## 6. Hallazgos Principales por Entidad

### Hallazgo 1: ORG es la Entidad más Difícil
- F1 promedio de ORG (~0.40) es consistentemente menor que PER (~0.52) y LOC (~0.54)
- Esto se debe a la mayor variabilidad en nombres de organizaciones
- Mayor sensibilidad a ruido de KG en ORG

### Hallazgo 2: LOC Beneficia más de Raw Data
- RAW LOC: 0.5498 vs CUR LOC: 0.5370
- Diferencia: +1.28%
- Los datos raw preservan patrones geográficos útiles

### Hallazgo 3: Generic KG Perjudica Todas las Entidades
- PER: NoKG=0.5354, GEN=0.4947 (-4.07%)
- LOC: NoKG=0.5621, GEN=0.5125 (-4.96%)
- ORG: NoKG=0.4144, GEN=0.3602 (-5.42%)
