# Decisiones pendientes

Se conservan los IDs de la segunda fuente. Todas están **pendientes**: documentar una recomendación no la convierte en decisión. Las correcciones neutrales pueden empezar sin resolverlas. No se requiere una autorización adicional para completar esta reconciliación documental.

<a id="dec-01"></a>

## DEC-01 — Oferta y despliegue inicial

**Opciones:** piloto dedicado operado con el cliente, self-host soportado o SaaS compartido. **Recomendación provisional:** validar un trabajo EDA recurrente en una instalación dedicada y supervisada; elegir self-host soportado si comprador, operación y economía lo justifican. MMM permanece experimental hasta su evaluación específica.

**Evidencia para decidir:** SR-035: comprador, datos permitidos, quién instala/opera, frecuencia, resultado esperado y disposición a pagar. Comparar costo de soporte/distribución frente a servicio gestionado. La exigencia de 2–3 clientes de pago antes de SaaS es un criterio propuesto, no una regla técnica ni un resultado conseguido.

**Bloquea:** comprometer oferta, topología de producción y precio; condiciona SR-036/038–041/056. **No bloquea:** integridad, aislamiento, UI y medición básicas. Antes de compartir una instalación, SR-037 aplica cualquiera que sea la ruta.

<a id="dec-02"></a>

## DEC-02 — Derechos y licencia por componente

**Opciones a evaluar:** licencia abierta del conjunto, componentes abiertos con otros comerciales o distribución comercial con derechos verificados. No se adopta automáticamente la propuesta open-core de la otra revisión.

**Evidencia para decidir:** inventario SR-051, titularidad/contribuciones, distribución prevista y revisión de los textos aplicables. `pyproject.toml` y el manifiesto del plugin declaran MIT; esto exige coherencia documental, no permite concluir aquí qué derechos se concedieron ni que vender requiera cambiar de licencia.

**Bloquea:** modificar declaraciones de licencia, publicar textos definitivos y distribuir una oferta cuyas condiciones no estén resueltas. **No bloquea:** inventario, correcciones o preparación de paquetes privados. Registrar decisión con alcance por componente y responsable; no reescribir historia ni asumir derechos sobre aportes ajenos.

<a id="dec-03"></a>

## DEC-03 — Solución DS o plataforma general

**Recomendación provisional:** presentar un resultado DS concreto. La separación del harness es una capacidad de arquitectura; la UI actual y el único dominio desarrollado no demuestran una plataforma general comercial.

**Evidencia para decidir:** demanda por otro dominio y SR-032 con segundo cartridge, namespaces, herramientas, modelo/configuración y UI derivados de su contrato; SR-028 para mensajes y marca. Un cartridge de prueba puede verificar el contrato sin financiar un segundo producto completo.

**Bloquea:** promesas de plataforma general, white-label y marketplace. **No bloquea:** piloto EDA. Registro remoto, firmas o requisitos enterprise se desglosan en SR-055/057 si hay demanda.

<a id="dec-04"></a>

## DEC-04 — Límite de ejecución y runtime soportado

**Entregable:** ADR de SR-009 con modelo de amenazas y experimentos de aceptación. Comparar contenedor endurecido y alternativas de aislamiento según usuarios, sistema operativo, GPU, red y credenciales. No tratar Docker, gVisor, Kata y microVMs como mecanismos idénticos.

**Criterios:** datos y secretos accesibles, plano de control protegido, Python persistente, límites, costos, cancelación, recuperación y operación. Un protocolo llamado sandbox, un cwd virtual o una VM por cliente no prueba aislamiento del código generado frente al servidor.

**Bloquea:** cierre de SR-010/048 y despliegue soportado. **No bloquea:** SR-046 como mitigación parcial, ni corregir HTTP, archivos o persistencia. Proveedor y tecnología quedan por decidir con evidencia, sin contratación implícita.

<a id="dec-05"></a>

## DEC-05 — Términos de proveedores y distribución externa

**Entregable:** registro de SR-051 de componentes/proveedores efectivamente usados, versión, uso propuesto, fuente contractual y cuestiones pendientes. Incluir modelos, BYOK o cuenta gestionada, Meridian, imágenes y redistribución de dependencias.

**Criterio:** resolver condiciones para la oferta elegida con responsables adecuados. BYOK cambia quién aporta la clave; no se acepta como demostración de ausencia de obligaciones. El informe de licencias previo no sustituye inventario ni verificación de textos.

**Bloquea:** afirmaciones contractuales y la oferta/distribución afectada. **No bloquea:** trabajo técnico neutral. No se ha emitido una conclusión jurídica en esta reconciliación.

## Registro

| ID | Estado | Responsable | Fecha | Decisión y evidencia |
|---|---|---|---|---|
| DEC-01 | Pendiente | Por asignar | — | — |
| DEC-02 | Pendiente | Por asignar | — | — |
| DEC-03 | Pendiente | Por asignar | — | — |
| DEC-04 | Pendiente | Por asignar | — | — |
| DEC-05 | Pendiente | Por asignar | — | — |
