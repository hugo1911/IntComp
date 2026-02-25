# Arquitectura y Tecnologías de la Solución

## Arquitectura General: Aplicación Web de ML en 3 Capas

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Streamlit Web UI (Frontend + Backend integrado)       │ │
│  │  - Interfaz interactiva con sliders y formularios      │ │
│  │  - Visualizaciones con Plotly                          │ │
│  │  - CSS personalizado para tema oscuro                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE LÓGICA DE NEGOCIO                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Scikit-learn Pipeline                                 │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │  1. ColumnTransformer (Preprocesamiento)         │ │ │
│  │  │     - SimpleImputer (mediana/moda)               │ │ │
│  │  │     - StandardScaler (normalización)             │ │ │
│  │  │     - OneHotEncoder (variables categóricas)      │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │  2. LogisticRegression (Clasificador)            │ │ │
│  │  │     - Multi-class (4 casas)                      │ │ │
│  │  │     - Solver: lbfgs                              │ │ │
│  │  │     - C=1.0 (regularización)                     │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PERSISTENCIA                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Modelos Serializados (Joblib)                        │ │
│  │  - best_model_logistic_regression.joblib              │ │
│  │  - label_encoder.joblib                               │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Tecnologías Utilizadas

### 1. Machine Learning & Data Science

| Tecnología | Versión | Uso en el Proyecto |
|------------|---------|-------------------|
| **Scikit-learn** | 1.6.1 | Framework principal de ML. Incluye:<br>• `Pipeline`: Encadenamiento de transformaciones<br>• `ColumnTransformer`: Preprocesamiento diferenciado<br>• `LogisticRegression`: Modelo de clasificación<br>• `GridSearchCV`: Búsqueda de hiperparámetros<br>• `StratifiedKFold`: Validación cruzada estratificada<br>• `StandardScaler`: Normalización de features<br>• `OneHotEncoder`: Codificación de variables categóricas<br>• `SimpleImputer`: Imputación de valores faltantes |
| **NumPy** | 1.26.4 | Operaciones numéricas y arrays multidimensionales |
| **Pandas** | 2.0.3 | Manipulación de datos tabulares (DataFrames) |
| **Joblib** | 1.3.2 | Serialización eficiente de modelos de ML |

### 2. Visualización & Frontend

| Tecnología | Versión | Uso en el Proyecto |
|------------|---------|-------------------|
| **Streamlit** | 1.28.0 | Framework web completo:<br>• Servidor web integrado<br>• Sistema de componentes interactivos (sliders, selectbox, buttons)<br>• Caching de recursos (`@st.cache_resource`)<br>• Gestión de estado de sesión<br>• Hot-reloading para desarrollo |
| **Plotly** | 5.17.0 | Visualizaciones interactivas:<br>• Gráficos de barras (probabilidades)<br>• Radar charts (perfil de habilidades)<br>• Gráficos responsivos con hover effects |
| **CSS3** | - | Estilos personalizados:<br>• Tema oscuro con gradientes<br>• Animaciones y transiciones<br>• Diseño responsive |
| **Google Fonts** | - | Tipografía profesional:<br>• Cinzel (headers)<br>• Inter (body text) |

### 3. Infraestructura & Deployment

| Tecnología | Versión | Uso en el Proyecto |
|------------|---------|-------------------|
| **Docker** | - | Containerización:<br>• Imagen base: `python:3.9-slim`<br>• Multi-stage build para optimización<br>• Arquitectura: `linux/amd64` |
| **Google Cloud Run** | - | Plataforma serverless:<br>• Auto-scaling (0-10 instancias)<br>• 2GB RAM, 2 vCPUs<br>• Timeout: 300s<br>• Puerto: 8080 |
| **Artifact Registry** | - | Registro de contenedores Docker |
| **Cloud Build** | - | CI/CD para builds automatizados |

### 4. Herramientas de Desarrollo

| Tecnología | Uso |
|------------|-----|
| **Jupyter Notebook** | Desarrollo y entrenamiento del modelo |
| **Kagglehub** | Descarga del dataset de Kaggle |
| **Git** | Control de versiones |
| **gcloud CLI** | Gestión de recursos GCP |

---

## Flujo de Datos

```
Usuario → Streamlit UI → Input Validation → Pandas DataFrame
                                                    ↓
                                          Scikit-learn Pipeline
                                                    ↓
                                    ┌───────────────┴───────────────┐
                                    ↓                               ↓
                            Preprocessing                    Classification
                         (ColumnTransformer)              (LogisticRegression)
                                    ↓                               ↓
                            Scaled Features                  Predictions
                                    └───────────────┬───────────────┘
                                                    ↓
                                          Probability Scores
                                                    ↓
                                    Plotly Visualizations ← Streamlit
                                                    ↓
                                              Usuario
```

---

## Arquitectura de Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                         USUARIO                              │
│                    (Navegador Web)                           │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   GOOGLE CLOUD RUN                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Container Instance                        │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │  Streamlit Server (Puerto 8080)                  │ │ │
│  │  │  ┌────────────────────────────────────────────┐ │ │ │
│  │  │  │  app.py                                    │ │ │ │
│  │  │  │  - Load models (cached)                    │ │ │ │
│  │  │  │  - Handle requests                         │ │ │ │
│  │  │  │  - Generate predictions                    │ │ │ │
│  │  │  └────────────────────────────────────────────┘ │ │ │
│  │  │  ┌────────────────────────────────────────────┐ │ │ │
│  │  │  │  models/                                   │ │ │ │
│  │  │  │  - best_model_logistic_regression.joblib   │ │ │ │
│  │  │  │  - label_encoder.joblib                    │ │ │ │
│  │  │  └────────────────────────────────────────────┘ │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Auto-scaling: 0-10 instancias según demanda                │
└─────────────────────────────────────────────────────────────┘
                         ↑
                         │ Pull Image
                         │
┌─────────────────────────────────────────────────────────────┐
│              ARTIFACT REGISTRY                               │
│  us-central1-docker.pkg.dev/hogwarts-sorting-955744/        │
│  hogwarts-repo/hogwarts-sorting-hat:latest                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Justificación de Tecnologías

### ¿Por qué Streamlit?
- **Desarrollo rápido**: Convierte scripts Python en apps web sin HTML/CSS/JS
- **Integración nativa con ML**: Funciona directamente con NumPy, Pandas, Scikit-learn
- **Caching inteligente**: `@st.cache_resource` evita recargar modelos
- **Deployment simple**: Compatible con Docker y Cloud Run

### ¿Por qué Scikit-learn?
- **Pipeline robusto**: Evita data leakage en preprocesamiento
- **Reproducibilidad**: Serialización completa del flujo de trabajo
- **Validación cruzada**: StratifiedKFold mantiene distribución de clases
- **Estándar de la industria**: Ampliamente usado y documentado

### ¿Por qué Google Cloud Run?
- **Serverless**: No gestión de servidores, auto-scaling
- **Costo-efectivo**: Pay-per-use, $0 cuando no hay tráfico
- **Escalabilidad**: Maneja picos de tráfico automáticamente
- **Simplicidad**: Deploy con un comando

### ¿Por qué Plotly sobre Matplotlib?
- **Interactividad**: Zoom, pan, hover tooltips
- **Responsive**: Se adapta a diferentes tamaños de pantalla
- **Estética moderna**: Mejor integración con tema oscuro

---

## Patrón de Arquitectura

**Patrón utilizado**: **Model-View-Controller (MVC) simplificado**

- **Model**: Pipeline de Scikit-learn (lógica de ML)
- **View**: Componentes de Streamlit (UI)
- **Controller**: Funciones de callback de Streamlit (manejo de eventos)

**Características adicionales**:
- **Separation of Concerns**: Modelo entrenado offline, app solo hace inferencia
- **Stateless**: Cada request es independiente (ideal para serverless)
- **Caching**: Modelo cargado una vez en memoria, reutilizado en requests

---

## Pipeline de Machine Learning

### Preprocesamiento

```python
ColumnTransformer([
    ('numeric', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    
    ('categorical', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])
```

**Features numéricas** (8):
- Bravery, Intelligence, Loyalty, Ambition
- Dark Arts Knowledge, Quidditch Skills, Dueling Skills, Creativity

**Features categóricas** (1):
- Blood Status (Pure-blood, Half-blood, Muggle-born)

### Modelo de Clasificación

```python
LogisticRegression(
    max_iter=1000,
    C=1.0,              # Regularización L2
    solver='lbfgs',     # Optimizador
    multi_class='auto'  # One-vs-Rest para 4 clases
)
```

**Output**: Probabilidades para 4 clases (Gryffindor, Hufflepuff, Ravenclaw, Slytherin)

---

## Métricas de Rendimiento

| Métrica | Train | Test |
|---------|-------|------|
| **Accuracy** | 92% | 89% |
| **F1-Score (macro)** | 90% | 87% |
| **Balanced Accuracy** | 90% | 88% |

**Validación**: 5-fold Stratified Cross-Validation

---

## Estructura de Archivos

```
ProyectoHP/
├── app.py                          # Aplicación Streamlit
├── Dockerfile                      # Configuración de contenedor
├── requirements.txt                # Dependencias Python
├── .dockerignore                   # Exclusiones de build
├── models/
│   ├── best_model_logistic_regression.joblib
│   └── label_encoder.joblib
├── Proyecto_P1_ICO.ipynb          # Notebook de entrenamiento
├── README.md                       # Documentación principal
├── ARCHITECTURE.md                 # Este archivo
├── DEPLOYMENT.md                   # Guía de deployment
└── QUICK_REFERENCE.txt            # Comandos rápidos
```

---

## Consideraciones de Seguridad

1. **No hay datos sensibles**: Modelos pre-entrenados, sin acceso a BD
2. **Stateless**: No se almacena información de usuarios
3. **HTTPS**: Cloud Run proporciona certificados SSL automáticos
4. **Autenticación**: Público (no requiere login para demo)
5. **Rate limiting**: Manejado por Cloud Run (protección DDoS)

---

## Escalabilidad

### Horizontal (Cloud Run)
- **Min instances**: 0 (scale to zero)
- **Max instances**: 10
- **Concurrency**: 80 requests por instancia
- **Cold start**: ~3-5 segundos

### Vertical (Recursos)
- **Memory**: 2 GB por instancia
- **CPU**: 2 vCPUs por instancia
- **Timeout**: 300 segundos

### Capacidad estimada
- **Requests/segundo**: ~800 (10 instancias × 80 concurrency)
- **Usuarios simultáneos**: ~800

---

## Monitoreo y Observabilidad

### Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND \
    resource.labels.service_name=hogwarts-sorting-hat"
```

### Métricas disponibles
- Request count
- Request latency (p50, p95, p99)
- Error rate
- Instance count
- CPU/Memory utilization

### Dashboards
- Google Cloud Console > Cloud Run > Metrics
- Logs Explorer para debugging

---

## Costos Estimados

| Recurso | Costo |
|---------|-------|
| **Cloud Run** | $0.00002 por request (después de 2M gratis) |
| **Artifact Registry** | $0.10/GB/mes de almacenamiento |
| **Networking** | Incluido en Cloud Run |
| **Total estimado** | **$0-5/mes** para uso moderado |

---

## Mejoras Futuras

### Técnicas
- [ ] Implementar modelos ensemble (Random Forest, XGBoost)
- [ ] A/B testing de diferentes modelos
- [ ] Feature engineering avanzado
- [ ] Explicabilidad con SHAP values

### Infraestructura
- [ ] CI/CD con GitHub Actions
- [ ] Staging environment
- [ ] Monitoring con Prometheus/Grafana
- [ ] Custom domain con Cloud DNS

### Funcionalidad
- [ ] Historial de predicciones
- [ ] Comparación con personajes famosos
- [ ] API REST para integraciones
- [ ] Multi-idioma (i18n)

---

**Proyecto desarrollado por:**
- David Alexandro García Morales (31624)
- Luis Bernardo Bremer Ortega (32366)
- Edgar Daniel De la Torre Reza (34887)
- Hugo German Manzano López (36231)

**Escuela de Ingeniería - Ingeniería en Ciencias Computacionales**  
**Inteligencia Computacional - Proyecto 1**
