# Validación técnica

## Ejecutado para preparar esta contribución

- Python3.12/macOS ARM64; TensorFlow2.18.1, Keras3.8.0, tensorflow-metal1.2.0, PyTorch2.9.1, NumPy2.0.2.
- Pruebas sintéticas de comparación pareada; fórmula LDL TensorFlow frente a NumPy; gradientes; serialización de modelo/optimizador; ETL → dos vistas → entrenamiento → reanudación → evaluación → oclusión; caso de abstención total; rechazo de test no reconocido como exploratorio.
- Integración en CPU con DINOv2 público real y una imagen sintética:1536 características, paso TensorFlow con gradientes y diagnóstico de oclusión.
- Comprobación de sintaxis Python/bash, ayuda CLI y exclusión de datos privados del commit.

## Límites

Las pruebas automatizadas del entorno de contribución no disponen de una GPU utilizable. El instalador exige comprobar MPS y Metal en el equipo destino. Esas comprobaciones operativas no sustituyen una evaluación independiente, un benchmark ni validación anatómica.

No se incorporan datos, etiquetas, identidades, predicciones por animal ni pesos entrenados. Los CSV de comparación se proporcionan externamente. La receta cambia backend e inicialización; si se entrena otro detector también cambia el pipeline, por lo que no se atribuye automáticamente una diferencia al clasificador.
