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
        return UsersRecommend_func(anio)
    except Exception as e:
        return {"Error":str(e)}
    
@app.get('/UsersNotRecommend/{anio}')   
def UsersNotRecommend(anio: int):
    
    try:
        return UsersNotRecommend_funct(anio)
    except Exception as e:
        return {"Error":str(e)}
    
@app.get('/sentiment_analysis/{anio}') 
def sentiment_analysis(anio: int):
    
    try:
        return sentiment_analysis_funct(anio)
    except Exception as e:
        return {"Error":str(e)}
