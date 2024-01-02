from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from functions import *

app = FastAPI()



@app.get("/", response_class=HTMLResponse)

def index_html():
    
    with open("templates/index.html", "r", encoding="utf-8") as file:
        html_content = file.read()

    return HTMLResponse(content=html_content)

@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero:str):
    
    try:
        return PlayTimeGenre_Funct(genero)
    except Exception as e:
        return {"Error":str(e)}
  
@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):

    try:
        return UserForGenre_Funct(genero)
    except Exception as e:
        return {"Error":str(e)}
    
@app.get('/UsersRecommend/{anio}')
def UsersRecommend(anio: int):
    
    try:
        return UsersRecommend_Funct(anio)
    except Exception as e:
        return {"Error":str(e)}
    
@app.get('/UsersNotRecommend/{anio}')   
def UsersNotRecommend(anio: int):
    
    try:
        return UsersNotRecommend_Funct(anio)
    except Exception as e:
        return {"Error":str(e)}
    
@app.get('/sentiment_analysis/{anio}') 
def sentiment_analysis(anio: int):
    
    try:
        return Sentiment_Analysis_Funct(anio)
    except Exception as e:
        return {"Error":str(e)}

@app.get('/Items_Recommend/{id}') 
def Items_Recommend(id: int):
    
    try:
        return Items_Recommend_Funct(id)
    except Exception as e:
        return {"Error":str(e)}

@app.get('/Users_Recommend/{id}') 
def Items_Recommend(id: str):
    
    try:
        return Users_Recommend_Funct(id)
    except Exception as e:
        return {"Error":str(e)}
