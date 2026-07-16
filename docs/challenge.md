Análisis Logístico de la predicción de consumo al pedido de reabastecimiento.

1. El problema no es predecir consumo, es decidir cuándo y cuánto pedir

El modelo implementado predice cantidad: las unidades que se consumirán (salidas) de un producto en una fecha dada. Ese es un insumo necesario, pero no es la decisión que le importa a CENABAST. La decisión real es de naturaleza logística: ¿cuándo se debe emitir el próximo pedido de un producto, y por cuánta cantidad?

Para pasar de una cosa a la otra hace falta combinar la predicción de consumo con tres elementos que hoy no están en el output del modelo:


Stock actual disponible del producto (stock.csv).
Lead time de reposición: cuánto tiempo transcurre entre que se emite un pedido y que la mercadería llega a bodega.
Incertidumbre de la demanda: el modelo predice un valor puntual, pero el consumo real varía alrededor de esa predicción, y esa variabilidad es la que obliga a mantener stock de seguridad.

2. Enfoque propuesto: política de reorden basada en punto de pedido (ROP)

Uso un modelo clásico de gestión de inventario punto de pedido (Reorder Point) con stock de seguridad  pero alimentado por las predicciones del modelo en vez de un promedio histórico fijo, que es como típicamente se calcula en sistemas de inventario tradicionales.

2.1 Demanda esperada durante el lead time

El modelo actual predice consumo para una fecha puntual. Para dimensionar un pedido hace falta el consumo esperado durante todo el lead time, no solo un día. Esto se resuelve extendiendo la predicción a un horizonte de L días (el lead time) mediante forecasting recursivo multi-step: se predice el día t+1, ese valor se usa como prev_cantidad / se incorpora a los rolling features para predecir t+2, y así sucesivamente hasta cubrir L días. El modelo ya está diseñado para esto — el modo "serving" de preprocess() reconstruye lags a partir del historial cacheado, por lo que basta con ir alimentando las predicciones generadas como si fueran nuevas observaciones.

La demanda esperada durante el lead time es:

D_L = Σ (predicción de cantidad para cada día dentro del lead time)

2.2 Stock de seguridad

El error del modelo (MAE de validación, ~3.28 unidades) da una estimación de la variabilidad no explicada. Formalizo esto con la desviación estándar del error de predicción (σ) y un nivel de servicio objetivo (z, por ejemplo z=1.65 para 95% de nivel de servicio):

Stock_seguridad = z · σ · √L

Esto absorbe tanto la variabilidad del consumo como el hecho de que el lead time mismo puede fluctuar (si se dispone de un histórico de lead times reales, se puede sumar también la varianza del lead time con la fórmula combinada de demanda y lead time variables).

2.3 Punto de pedido (Reorder Point)

ROP = D_L + Stock_seguridad

Regla de decisión: cuando el stock actual de un producto (de stock.csv, o idealmente un feed en tiempo real) cae por debajo de su ROP, se dispara la recomendación de generar un pedido.

2.4 Cantidad a pedir

Dos alternativas, dependiendo de la política operativa de CENABAST:

Order-up-to level (política s,S): se pide hasta llegar a un nivel objetivo S = ROP + Q_ciclo (donde Q_ciclo cubre el consumo esperado hasta el próximo ciclo de revisión). Cantidad a pedir = S − stock_actual.

Lote económico de pedido (EOQ): si hay costos de ordenar y de mantener inventario razonablemente estables, se puede fijar una cantidad estándar de pedido en vez de recalcularla cada vez, minimizando costo total.

Para un contexto hospitalario donde el desabastecimiento tiene costo altísimo (riesgo clínico) frente al costo de sobre-stock, me inclinaría por s,S con un nivel de servicio alto, dejando el EOQ como refinamiento posterior una vez que el proceso esté validado.

3. Estimación del lead time

El dataset no trae explícitamente el lead time por producto o proveedor, pero se puede aproximar con los datos disponibles: movimientos.csv tiene movimientos tipo_movimiento == 'E' (entradas), que corresponden a la llegada de pedidos. Si se dispusiera además del registro de cuándo se emitió cada pedido (no solo cuándo llegó), el lead time por producto sería directamente la diferencia de fechas entre emisión y entrada. Sin ese dato de emisión, una aproximación razonable es usar el intervalo típico entre entradas consecutivas como proxy del ciclo de reposición, reconociendo que es una simplificación que debería reemplazarse por el dato real de lead time del proveedor en cuanto esté disponible.

4. Consideraciones adicionales para producción

canasta_vigente: productos que salieron de la canasta farmacéutica vigente no deberían generar recomendaciones de reposición aunque su ROP lo sugiera; esta bandera actúa como filtro previo a cualquier recomendación.

Cantidades mínimas de compra y presentación del proveedor: la cantidad calculada debe redondearse hacia arriba al múltiplo de la unidad de compra o factor de empaque (cajas, blisters, etc.), no se puede pedir una cantidad arbitraria.

Vida útil / vencimiento: para fármacos con vida útil corta, el nivel objetivo S debe acotarse por la tasa de consumo para evitar mermas por vencimiento, incluso si el cálculo de ROP puro sugiriera un stock mayor.

Validación: antes de operacionalizar la política, se puede hacer backtesting corriendo la lógica de ROP sobre el histórico de stock.csv y movimientos.csv, comparando cuántos quiebres de stock reales habría evitado (o generado) la política propuesta, y ajustar z según el apetito de riesgo real de CENABAST.

5. Resumen del flujo

Predicción de consumo (modelo actual, por día)
        │
        ▼
Forecast recursivo a L días (horizonte = lead time)
        │
        ▼
Demanda esperada en lead time (D_L) + Stock de seguridad (σ, z)
        │
        ▼
Punto de pedido (ROP) por producto
        │
        ▼
¿stock_actual < ROP?  ──No──▶ no se genera acción
        │
       Sí
        ▼
Recomendación de pedido: cantidad = S − stock_actual
(ajustada por múltiplos de compra, vida útil y estado de canasta_vigente)

Este enfoque reutiliza directamente el modelo de consumo ya construido no requiere entrenar un modelo distinto para "predecir pedidos" y traduce su output a una decisión operativa mediante reglas de inventario estándar, que es exactamente el tipo de capa que falta entre un modelo de forecasting y un sistema de reabastecimiento real.