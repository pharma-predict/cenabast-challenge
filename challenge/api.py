import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd

from challenge.model import ReplenishmentModel


app = fastapi.FastAPI()


model = ReplenishmentModel()


try:
    movimientos = pd.read_csv(
        "dataset/movimientos.csv"
    )

    movimientos["gtin"] = movimientos["gtin"].astype(str)

except Exception:
    movimientos = pd.DataFrame()



class ProductRequest(BaseModel):
    gtin: str
    fecha: str



class PredictionRequest(BaseModel):
    products: list[ProductRequest]



@app.get("/health", status_code=200)
async def get_health() -> dict:

    return {
        "status": "OK"
    }



@app.post("/predict", status_code=200)
async def post_predict(
    request: PredictionRequest
) -> dict:


    predictions = []


    for product in request.products:


        # Validar fecha

        try:

            datetime.strptime(
                product.fecha,
                "%Y-%m-%d"
            )

        except ValueError:

            raise HTTPException(
                status_code=400,
                detail="Fecha inválida"
            )


        # Validar producto

        if product.gtin not in movimientos["gtin"].values:

            raise HTTPException(
                status_code=400,
                detail="Producto desconocido"
            )


        data = movimientos[
            movimientos["gtin"] == product.gtin
        ].copy()


        features = model.preprocess(
            data=data
        )


        result = model.predict(
            features=features
        )


        cantidad = result[0]["cantidad"]


        predictions.append(
            {
                "gtin": product.gtin,
                "fecha": product.fecha,
                "cantidad": cantidad
            }
        )


    return {
        "predict": predictions
    }