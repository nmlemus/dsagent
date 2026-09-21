# Fuentes y límites de evidencia

Fecha de reconciliación: 2026-09-19. Base común declarada por las revisiones: commit `8ac2758`. El nombre del informe inicial lleva 2026-09-18; se conserva sin renombrar y sin inferir una ejecución adicional. Esta reconciliación compara documentos y comprueba puntos concretos del código; no repite toda la auditoría, pruebas de navegador, sonda de sandbox ni benchmarks.

- **Reproducido en la evaluación inicial:** pytest 369 passed/12 skipped mediante python -m pytest, build UI correcto, siete hallazgos Ruff; replay EDA y defectos de archivos, exportación, identidad, sesiones y validadores detallados en el informe. Son resultados históricos, no pruebas ejecutadas de nuevo al reconciliar.
- **Reportado por la segunda fuente:** once revisiones y una sonda de acceso del backend local. Se conserva la atribución; no se presenta su experimento como ejecutado por esta reconciliación.
- **Confirmado por lectura al reconciliar:** make_kernel_env usa inherit_env=True; Jupyter se crea de forma diferida en run_python; metadata MIT en pyproject.toml y manifiesto del plugin. Esto no resuelve permisos legales ni determina exposición de un despliegue concreto.
- **Pendiente:** carga/restart multi-réplica, sandbox endurecido, precisión MMM/EDA fuera de lo evaluado, UX con usuarios, seguridad de dependencias, derechos/términos y demanda comercial. No se ha realizado estudio de mercado ni validación jurídica.

Se preservan los originales. Los hashes permiten detectar si cambian después; no certifican sus afirmaciones. No se copiaron secretos, datos de cliente ni archivos temporales de pruebas.

| Fuente | SHA-256 al reconciliar |
|---|---|
| [docs/startup-readiness-review-2026-09-18.md](../startup-readiness-review-2026-09-18.md) | `81e16bf743053e54ff13de54b0a94bef4dfe9b389d60959374b0715106883bfa` |
| [docs/STARTUP-BACKLOG.md](../STARTUP-BACKLOG.md) | `86f69be12eda94b09bba18e02dd413c2f77e72407b772df988ffcc2804f99f47` |
| [docs/reviews/2026-09-19-product-readiness/00-consolidated-plan.md](../reviews/2026-09-19-product-readiness/00-consolidated-plan.md) | `f55902986bfcf55bb834c61e16d5edaa18c43dc330ff2b19b9ba1fc799d21dbd` |
| [docs/reviews/2026-09-19-product-readiness/BACKLOG.md](../reviews/2026-09-19-product-readiness/BACKLOG.md) | `349a291904a0cb88cba3025bbaa979419fc657e46af6c14dd41fb3af39d8e383` |
| [docs/reviews/2026-09-19-product-readiness/README.md](../reviews/2026-09-19-product-readiness/README.md) | `46a53249292d3945e540b59790358e8e467f2868cea11b7532c46e406a2af290` |
| [docs/reviews/2026-09-19-product-readiness/round1/00-synthesis-P0-P3.md](../reviews/2026-09-19-product-readiness/round1/00-synthesis-P0-P3.md) | `5d79c21fb091fc2bf7095918b1bb31a594362cb9b82651816088501c5766b958` |
| [docs/reviews/2026-09-19-product-readiness/round1/01-architecture.md](../reviews/2026-09-19-product-readiness/round1/01-architecture.md) | `14f0604a49904ba8cb706406eeb0874c0750855e2582610da3f80adde873863c` |
| [docs/reviews/2026-09-19-product-readiness/round1/02-scalability.md](../reviews/2026-09-19-product-readiness/round1/02-scalability.md) | `eadf3fc63dc2ad35471d100a0c70204664fd93bc6a46e0ebfc73222158927339` |
| [docs/reviews/2026-09-19-product-readiness/round1/03-ui.md](../reviews/2026-09-19-product-readiness/round1/03-ui.md) | `afeb85e47653bda1fc0d12b585ac0a192ed8f3a6933c62557eee71b2eed63842` |
| [docs/reviews/2026-09-19-product-readiness/round1/04-security.md](../reviews/2026-09-19-product-readiness/round1/04-security.md) | `c244f320774f8bc6a4101ece0125e0e3d5610e33dead3b648d2f49b2257ca968` |
| [docs/reviews/2026-09-19-product-readiness/round1/05-premortem.md](../reviews/2026-09-19-product-readiness/round1/05-premortem.md) | `699bddeb6687619c8cc5cb0722561ae2138e836ffa4ff0e8ee9a932471142409` |
| [docs/reviews/2026-09-19-product-readiness/round2/01-cloud-scale-ops.md](../reviews/2026-09-19-product-readiness/round2/01-cloud-scale-ops.md) | `07e2d95f31bfcb0d0c0e86b5c5da944d824901f8668a7a6d579ae1363dd175bb` |
| [docs/reviews/2026-09-19-product-readiness/round2/02-saas-tenancy.md](../reviews/2026-09-19-product-readiness/round2/02-saas-tenancy.md) | `763c10c0b68aaf62562ef871da610a69a04bf5535055cd01a7743991e2375f34` |
| [docs/reviews/2026-09-19-product-readiness/round2/03-saas-security.md](../reviews/2026-09-19-product-readiness/round2/03-saas-security.md) | `6fb0ae0d32b888b42502cd5db94c2ae57e49af13897c88ea6be4104de20f6add` |
| [docs/reviews/2026-09-19-product-readiness/round2/04-data-durability.md](../reviews/2026-09-19-product-readiness/round2/04-data-durability.md) | `94e4ac5c8adfefc04cb90dc74b5a79b8518f2f0ea5608f9eb4b5115ca6667cec` |
| [docs/reviews/2026-09-19-product-readiness/round2/05-product-commercial.md](../reviews/2026-09-19-product-readiness/round2/05-product-commercial.md) | `47e16d0039bac9d26cc0918861f358333ae604b89485e02d922ad6663aa1f32e` |
| [docs/reviews/2026-09-19-product-readiness/round2/06-premortem-product.md](../reviews/2026-09-19-product-readiness/round2/06-premortem-product.md) | `337174e3ea9c583435d616ee28758492803fae9139c343b7e5ac7cf33eba5811` |
