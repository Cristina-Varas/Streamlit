import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
#import shap

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report


###############################################################################


# Título de la página
st.title('Diseño y Programación de Herramientas Analíticas')
st.write('## **Actividad 2: Desarrollo de Herramienta Analítica**')


######## CARGA DEL DATASET ####################################################

# Carga del DataSet
dataset = pd.read_csv("C:\\Users\\crist\\.spyder-py3\\Actividad2\\Data\\penguins_size.csv")
dataset = dataset.dropna()

# Codificar las variables no numéricas
ord_enc = OrdinalEncoder()

dataset_encoded = dataset.copy(deep=True)
dataset_encoded["island"] = ord_enc.fit_transform(dataset_encoded[["island"]])
dataset_encoded["sex"] = ord_enc.fit_transform(dataset_encoded[["sex"]])
dataset_encoded["species"] = ord_enc.fit_transform(dataset_encoded[["species"]])

features = dataset_encoded.columns.drop(['species'])
X = dataset_encoded[features]
y = dataset_encoded['species']

SEED = 1234
LABELS = ['Adelie - 1', 'Chinstrap - 2', 'Gentoo - 3']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, shuffle = True, test_size = 0.2, random_state=SEED)

#data_dataset_encoded = dataset_encoded.values
#X = data_dataset_encoded[:,0:6]
#y = data_dataset_encoded[:, 0]
#print(X)
#print(y)

# Lista de características
# species, island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex
st.write('#### 1. Descripción del Dataset: Palmer Archipielago (Antartica) penguin data')
st.write('Los datos con los que se va a trabajar fueron recopilados y puestos a disposición por la doctora kristen Gorman y la Estación Palmer, miembro de la Red De Investigación Ecológica a Largo Plazo.')
st.write('Están compuestos por las siguientes características:')
st.write('**species** - Variable a predecir, indica la especie a la que pertenece el pingüino (Dream, Biscoe y Torgersen). ')
st.write('**island** - Isla a la que pertenece el pingüino (Dream, Biscoe y Torgersen).')
st.write('**culmen_length_mm** - Longitud de la cresta del pingüino en mm.')
st.write('**culmen_depth_mm** - Profundidad de la cresta del pingüino en mm. ')
st.write('**flipper_length_mm** - Longitud de la aleta del pingüino en mm.')
st.write('**body_mass_g** - Peso del pingüino en gramos.')
st.write('**sex** - Sexo del pingüino.')

st.write('##### Dataset original con el que se va a trabajar')
st.write(dataset)

st.write('##### Dataset codificado')
st.write('Es necesario codificar las características no numéricas a números para poder llevar a cabo el entrenamiento del modelo.')
st.write(dataset_encoded)

################# EDA #########################################################
st.markdown('#### 2. Análisis Exploratorio de los Datos (EDA)')
st.write('##### Matriz de Correlación')
corr_df = dataset_encoded.drop(['species'], axis=1).corr(method='pearson')
corr_df.style.background_gradient(cmap='coolwarm')
st.write(corr_df)

st.write('##### Dimensiones del dataset')
st.write(dataset_encoded.shape)
st.write("El dataset cuenta con 334 filas y 7 características")

st.write('##### Algunos datos estadísticos de las características del dataset ya codificado')
st.write(dataset_encoded.describe().T)

st.write('##### Número de valores nulos de cada característica')
st.write(dataset_encoded.isnull().sum())

st.write('##### Número total de valores duplicados')
st.write(dataset_encoded.duplicated().sum())


st.write('---')

############### GRÁFICOS INTERACTIVOS ########################################
st.write('##### Gráficos')
chart_select = st.selectbox(
    label = "Selecciona el tipo de gráfico",
    options=['Diagrama de puntos', 'Diagrama de líneas', 'Histograma', 'Boxplot']
    )

numeric_columns = list(dataset_encoded.select_dtypes(['float', 'int']).columns)

if chart_select == 'Diagrama de puntos':
    st.write('##### Ajustes del diagrama de puntos')
    try:
        x_values = st.selectbox('Eje X', options=numeric_columns)
        y_values = st.selectbox('Eje Y', options=numeric_columns)
        plot_puntos = px.scatter(data_frame = dataset, x = x_values, y = y_values)
        st.write(plot_puntos)
    except Exception as e:
        print(e)
        
if chart_select == "Histograma":
    st.write('##### Ajustes del histograma')
    try:
        x_values = st.selectbox('Eje X', options=numeric_columns)
        plot_histograma = px.histogram(data_frame=dataset, x=x_values)
        st.write(plot_histograma)
    except Exception as e:
        print(e)
        
if chart_select == "Diagrama de líneas":
    st.write('##### Ajustes del Diagrama de Líneas')
    try:
        x_values = st.selectbox('Eje X', options = numeric_columns)
        y_values = st.selectbox('Eje Y', options = numeric_columns)
        plot_lineas = px.line(dataset, x = x_values, y = y_values)
        st.write(plot_lineas)
    except Exception as e:
        print(e)

if chart_select == "Boxplot":
    st.write('##### del Boxplot')
    try:
        x_values = st.selectbox('Eje X', options = numeric_columns)
        y_values = st.selectbox('Eje Y', options = numeric_columns)
        plot_box = px.box(dataset, x = x_values, y = y_values)
        st.write(plot_box)
    except Exception as e:
        print(e)
        
##### SIDEBAR
# Cabecera para especificar los parámetros de entrada
st.sidebar.header('Especifica los parámetros de entrada para entrenar el modelo')

def user_input_features():

    try:
        islas = list((0, 1, 2))
        ISLAND = st.sidebar.selectbox('Island (0 - Dream, 1 - Biscoe, 2 - Torgersen)', options=islas)
        #st.write(plot)
    except Exception as e:
        print(e)
    
    CULMEN_LENGTH_MM = st.sidebar.slider('CULMEN_LENGTH_MM', float(X['culmen_length_mm'].min()), float(X['culmen_length_mm'].max()), float(X['culmen_length_mm'].mean()))
    CULMEN_DEPTH_MM = st.sidebar.slider('CULMEN_DEPTH_MM ', float(X.culmen_depth_mm.min()), float(X.culmen_depth_mm.max()), float(X.culmen_depth_mm.mean()))
    FLIPPER_LENGTH_MM = st.sidebar.slider('FLIPPER_LENGTH_MM', float(X.flipper_length_mm.min()), float(X.flipper_length_mm.max()), float(X.flipper_length_mm.mean()))
    BODY_MASS_G = st.sidebar.slider('BODY_MASS_G', float(X.body_mass_g.min()), float(X.body_mass_g.max()), float(X.body_mass_g.mean()))
   
    try:
        sexo = list((0, 1))
        SEX = st.sidebar.selectbox('Sexo(0 - Masculino, 1 - Femenino)', options=sexo)
        #st.write(plot)
    except Exception as e:
        print(e)
    input_data = {'ISLAND': ISLAND,
            'CULMEN_LENGTH_MM': CULMEN_LENGTH_MM,
            'CULMEN_DEPTH_MM': CULMEN_DEPTH_MM,
            'FLIPPER_LENGTH_MM': FLIPPER_LENGTH_MM,
            'BODY_MASS_G': BODY_MASS_G,
            'SEX': SEX}
 
    features = pd.DataFrame(input_data, index=[0])
    return features
    
df = user_input_features()


############### ENTRENAMIENTO DEL MODELO ######################################
# Construir el modelo de clasificación

st.write('#### 3. Entrenamiento del modelo y testeo')
st.write('##### 3.1. Entrenamiento del modelo')
chart_select = st.selectbox(
    label = "Selecciona el tipo de algoritmo que quieres utilizar para entrenar el modelo de clasificación",
    options=['Árbol de decisión', 'Random Forest']
    )

# Entrenamiento del modelo si se escoge un árbol de decisión
def train_tree():
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    tree_score_train = tree.score(X_train, y_train)
    rtree_score_test = tree.score(X_test, y_test)
    print("Train: ", tree.score(X_train, y_train))
    print("Test: ", tree.score(X_test, y_test))
    st.write("Accuracy de Train: " + str(tree_score_train))
    st.write("Accuracy de Test: " + str(rtree_score_test))
    return tree

# Entrenamiento del modelo si se escoge un random forest
def train_random_forest():
    random_forest = RandomForestClassifier(max_depth=5, random_state=42)
    random_forest.fit(X_train, y_train)
    random_forest_score_train = random_forest.score(X_train, y_train)
    random_forest_score_test = random_forest.score(X_test, y_test)
    print("Train: ", random_forest_score_train)
    print("Test: ", random_forest_score_test)
    st.write("Accuracy de Train: " + str(random_forest_score_train))
    st.write("Accuracy de Test: " + str(random_forest_score_test))
    return random_forest

# Predicción del modelo
def predict(model):
    pred_y = model.predict(X_test)
    return pred_y
    
# Mostrar resultados
def mostrar_resultados(model, y_test, y_pred):
    plt.figure(figsize=(12, 12))
    plt.title("Matriz de confusión")
    plt.ylabel('Clase verdadera')
    plt.xlabel('Clase predicha')
    plt.show()

    report = classification_report(y_test, pred_y, output_dict = True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    st.write(report_df)
    
# Prediccion del modelo en base a los parametros introducidos
def predecir(model):
    st.write('Escoja los parámetros en la barra lateral para predecir la clase de pingüino en base al modelo elegido entrenado.')
    st.write('**Datos introducidos**')
    st.dataframe(df)
    # Aplicar el modelo para hacer la predicción
    prediction = model.predict(df)
    st.write('**Predicción en base a los datos introducidos**')
    st.write(prediction)
    prediction_numeric = int(prediction)
    print("PREDICTION: " + str(prediction))
    print(type(prediction))
    print(type(prediction_numeric))
    if prediction_numeric == 0:
        st.write("Un pingüino con las características seleccionadas es de la especie Adelie")
    if prediction_numeric == 1:
        st.write("Un pingüino con las características seleccionadas es de la especie Chinstrap")
    if prediction_numeric == 2:
        st.write("Un pingüino con las características seleccionadas es de la especie Gentoo")
        
    st.write('---')
    

# Selección del modelo    
if chart_select == 'Árbol de decisión':
    tree = train_tree()
    pred_y = predict(tree)
    mostrar_resultados(tree, y_test, pred_y)
    st.write('##### 3.2. Predición sobre el modelo Árbol de Decisión')
    predecir(tree)
    
    
if chart_select == 'Random Forest':
    random_forest = train_random_forest()
    pred_y = predict(random_forest)
    mostrar_resultados(random_forest, y_test, pred_y)
    st.write('##### 3.2. Predición sobre el modelo Random Forest')
    predecir(random_forest)


     
