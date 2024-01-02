import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


####################################################################################################################
#Dataframe Original
df_original = pd.read_parquet('final.parquet.gzip')

#Dataframe Steam_Games
df_steam_games=pd.read_json('data/steam_games.json',lines=True)
df_steam_games=df_steam_games.drop(columns=['Unnamed: 0'])

#Dataframe Review
df_review = df_original.loc[:, ['recomended_item_id','sentiment_analysis','recommend']]
df_review=df_review.dropna()
df_review['recomended_item_id'] = df_review['recomended_item_id'].apply(lambda x: [int(i) for i in x])
    # Primero, eliminamos los duplicados en df_steam_games
df_steam_games = df_steam_games.drop_duplicates(subset='id')

    # Convertimos df_steam_games a un diccionario para facilitar la búsqueda
dict_df2 = df_steam_games.set_index('id').to_dict('index')

    # Creamos una nueva columna en df1 que contendrá los datos correspondientes de df2
df_review['games_info'] = df_review['recomended_item_id'].apply(lambda x: [dict_df2.get(i, {}) for i in x])


#Dataframe Usuarios
df_user_info=df_original.copy()

#Dataframe para funcion 1
df_genre_year_with_most_playtime=pd.read_json('data_optimizada/genre_year_with_most_playtime.json',lines=True)

#Dataframe para funcion 2
df_user_with_most_playtime=pd.read_json('data_optimizada/user_with_most_playtime.json',lines=True)
####################################################################################################################






################################################ FUNCION 1 ################################################
def PlayTimeGenre_Funct(genre):
    """
    ---1---
    Devuelve el año con más horas jugadas para dicho género.
    OPTIMIZAR
    """
    global df_review, df_steam_games, df_original
    # Crear un nuevo DataFrame para evitar modificar el original
    df_filtrado = df_genre_year_with_most_playtime[df_genre_year_with_most_playtime['Genre'] == genre]
    anio_con_mas_horas = int(df_filtrado['Year with Most Playtime'].max())
    return {f"Año de lanzamiento con más horas jugadas para {genre}": anio_con_mas_horas}



################################################ FUNCION 2 ################################################
def UserForGenre_Funct(genre):
    """
    ---2---
    Devuelve el usuario que acumula más horas jugadas para el género dado
    y una lista de la acumulación de horas jugadas por año.
    OPTIMIZAR
    """
    # Filtrar el DataFrame por el género dado
    df_genre = df_user_with_most_playtime[df_user_with_most_playtime['Género'] == genre]

    # Si no hay datos para el género dado, devolver un mensaje indicando esto
    if df_genre.empty:
        return f"No hay datos disponibles para el género {genre}."

    # Obtener el usuario con más horas jugadas para el género dado
    user = df_genre['Usuario con más horas jugadas'].iloc[0]

    # Obtener la lista de la acumulación de horas jugadas por año
    playtime_by_year = df_genre['Años y Horas'].iloc[0]

    # Devolver un diccionario con el usuario con más horas jugadas y la lista de la acumulación de horas jugadas por año
    return {"Usuario con más horas jugadas para " + genre: user, "Horas jugadas": playtime_by_year}


################################################ FUNCION 3 ################################################
def UsersRecommend_Funct(year):
    """
    ---3---
    Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
    """
    
    global df_review, df_steam_games, df_original

    def extract_year(release_date):
        # Función auxiliar para extraer el año de 'release_date'
        try:
            return datetime.strptime(release_date, '%Y-%m-%d').year
        except (TypeError, ValueError):
            return None

    # Filtramos los juegos que fueron lanzados en el año especificado
    df_year = df_review[df_review['games_info'].apply(lambda x: any((
        isinstance(d, dict) and 'release_date' in d and 
        (('release_date' in d and isinstance(d['release_date'], datetime) and d['release_date'].year == year) or
         ('release_date' in d and isinstance(d['release_date'], str) and extract_year(d['release_date']) == year))
    ) for d in x))]

    # Aplanamos la lista de 'recomended_item_id' y contamos las recomendaciones para cada ID
    id_list = [id for sublist in df_year['recomended_item_id'].tolist() for id in sublist]
    
    # Usamos un diccionario en lugar de Counter para contar las recomendaciones
    recommend_count = {}
    for id in id_list:
        if id in recommend_count:
            recommend_count[id] += 1
        else:
            recommend_count[id] = 1

    # Obtenemos los tres juegos más recomendados
    top_games = sorted(recommend_count.items(), key=lambda item: item[1], reverse=True)[:3]
    
    # Creamos un diccionario con los nombres de los juegos
    game_names = {game_id: df_steam_games.loc[df_steam_games['id'] == game_id, 'app_name'].values[0] for game_id, _ in top_games}
    
    # Devolvemos una lista de diccionarios con los puestos y los nombres de los juegos
    return [{"Puesto {}".format(i+1): game_names[id]} for i, (id, count) in enumerate(top_games)]



################################################ FUNCION 4 ################################################
def UsersNotRecommend_Funct(year):
    """
    ---4---
    Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.
    """
    
    global df_review, df_steam_games

    def extract_year(release_date):
        # Función auxiliar para extraer el año de 'release_date'
        try:
            return datetime.strptime(release_date, '%Y-%m-%d').year
        except (TypeError, ValueError):
            return None

    # Filtramos los juegos que fueron lanzados en el año especificado y no son recomendados
    df_year = df_review[(df_review['games_info'].apply(lambda x: any(
        isinstance(d, dict) and 'release_date' in d and 
        (('release_date' in d and isinstance(d['release_date'], datetime) and d['release_date'].year == year) or
         ('release_date' in d and isinstance(d['release_date'], str) and extract_year(d['release_date']) == year))
    for d in x))) & (df_review['recommend'].apply(lambda x: all(not r for r in x)))]

    # Aplanamos la lista de 'recomended_item_id' y contamos las recomendaciones para cada ID
    id_list = [id for sublist in df_year['recomended_item_id'].tolist() for id in sublist]
    
    # Usamos un diccionario en lugar de Counter para contar las recomendaciones
    recommend_count = {}
    for id in id_list:
        if id in recommend_count:
            recommend_count[id] += 1
        else:
            recommend_count[id] = 1

    # Obtenemos los tres juegos menos recomendados
    top_games = sorted(recommend_count.items(), key=lambda item: item[1])[:3]
    
    # Creamos un diccionario con los nombres de los juegos
    game_names = {game_id: df_steam_games.loc[df_steam_games['id'] == game_id, 'app_name'].values[0] for game_id, _ in top_games}
    
    # Devolvemos una lista de diccionarios con los puestos y los nombres de los juegos
    return [{"Puesto {}".format(i+1): game_names[id]} for i, (id, count) in enumerate(top_games)]



################################################ FUNCION 5 ################################################
def Sentiment_Analysis_Funct(year):
    """
    ---5---
    Según el año de lanzamiento, se devuelve una lista con la cantidad de registros
    de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
    """
    df=df_review.copy()
    # Primero, vamos a "normalizar" la columna games_info para extraer el año de lanzamiento
    df['release_year'] = df['games_info'].apply(lambda x: [d.get('release_date')[:4] for d in x if 'release_date' in d])

    # Ahora, vamos a "explotar" las listas en las columnas para que cada elemento tenga su propia fila
    df = df.explode('release_year').explode('sentiment_analysis')

    # Filtramos el dataframe por el año dado
    df_year = df[df['release_year'] == str(year)]

    # Contamos los valores de análisis de sentimiento
    sentiment_counts = df_year['sentiment_analysis'].value_counts()

    # Convertimos los enteros de análisis de sentimiento a cadenas
    sentiment_dict = {"Negative": int(sentiment_counts.get(0, 0)), 
                      "Neutral": int(sentiment_counts.get(1, 0)), 
                      "Positive": int(sentiment_counts.get(2, 0))}
    
    return sentiment_dict


################################################ FUNCION 6 ################################################
def Items_Recommend_Funct( id_producto ):
    """
    ---6---
    Ingresando el id de producto, 
    deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.
    """
    df=df_steam_games
    def obtener_datos_producto(df, id_producto):
        # Esta función debería devolver los datos del producto con el id dado
        producto_df = df[df['id'] == id_producto]
        if not producto_df.empty:
            producto_info = producto_df[['publisher', 'genres', 'app_name', 'title', 'tags', 'specs', 'price', 'early_access', 'developer']].values[0]
            return ' '.join(map(str, producto_info))
        else:
            return None

    def calcular_similitud_coseno(producto, todos_los_productos):
        # Esta función debería calcular la similitud del coseno entre el producto dado y todos los demás productos
        vectorizer = TfidfVectorizer()
        producto_vec = vectorizer.fit_transform([producto])
        todos_los_productos_vec = vectorizer.transform(todos_los_productos)
        
        # Luego, calculamos la similitud del coseno
        similitudes = cosine_similarity(producto_vec, todos_los_productos_vec)
        return similitudes

    def ordenar_por_similitud(similitudes):
        # Esta función debería ordenar los productos por similitud
        return similitudes.argsort()

    # Obtén los datos del producto
    producto = obtener_datos_producto(df, id_producto)
    
    # Si el producto no existe, devuelve un mensaje de error
    if producto is None:
        return {"Error": f"No se encontró ningún producto con el id {id_producto}"}
    
    # Si el producto existe, calcula la similitud del coseno entre el producto y todos los demás productos
    todos_los_productos = [' '.join(map(str, x)) for x in df[['publisher', 'genres', 'app_name', 'title', 'tags', 'specs', 'price', 'early_access', 'developer']].values]
    similitudes = calcular_similitud_coseno(producto, todos_los_productos)
    
    # Ordena los productos por similitud y toma los 5 más similares
    productos_similares = ordenar_por_similitud(similitudes)[0][-5:]
    
    # Devuelve los productos similares en formato de diccionario con id y nombre del producto
    return {id: nombre for id, nombre in zip(df.iloc[productos_similares]['id'], df.iloc[productos_similares]['app_name'])}


################################################ FUNCION 7 ################################################
def Users_Recommend_Funct( id_usuario ):
    """
    ---7---
    Ingresando el id de un usuario,
    deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.
    """
    df_user = df_user_info
    df_item = df_steam_games

    def obtener_datos_usuario(df, id_usuario):
        # Esta función debería devolver los datos del usuario con el id dado
        usuario_df = df[df['user_id'] == id_usuario]
        if not usuario_df.empty:
            usuario_info = usuario_df[['steam_id', 'user_url', 'user_item_id', 'played_item_name', 'playtime_forever', 'playtime_2weeks', 'review_content', 'recomended_item_id', 'sentiment_analysis', 'recommend']].values[0]
            return ' '.join(map(str, usuario_info))
        else:
            return None

    def calcular_similitud_coseno(usuario, todos_los_productos):
        # Esta función debería calcular la similitud del coseno entre el usuario dado y todos los productos
        vectorizer = TfidfVectorizer()
        usuario_vec = vectorizer.fit_transform([usuario])
        todos_los_productos_vec = vectorizer.transform(todos_los_productos)
        
        # Luego, calculamos la similitud del coseno
        similitudes = cosine_similarity(usuario_vec, todos_los_productos_vec)
        return similitudes

    def ordenar_por_similitud(similitudes):
        # Esta función debería ordenar los productos por similitud
        return similitudes.argsort()

    # Obtén los datos del usuario
    usuario = obtener_datos_usuario(df_user, id_usuario)
    
    # Si el usuario no existe, devuelve un mensaje de error
    if usuario is None:
        return {"Error": f"No se encontró ningún usuario con el id {id_usuario}"}
    
    # Si el usuario existe, calcula la similitud del coseno entre el usuario y todos los productos
    todos_los_productos = [' '.join(map(str, x)) for x in df_item[['publisher', 'genres', 'app_name', 'title', 'tags', 'specs', 'price', 'early_access', 'developer']].values]
    similitudes = calcular_similitud_coseno(usuario, todos_los_productos)
    
    # Ordena los productos por similitud y toma los 5 más similares
    productos_similares = ordenar_por_similitud(similitudes)[0][-5:]
    
    # Devuelve los productos similares en formato de diccionario con id y nombre del producto
    return {id: nombre for id, nombre in zip(df_item.iloc[productos_similares]['id'], df_item.iloc[productos_similares]['app_name'])}