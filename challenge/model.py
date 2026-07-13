import pandas as pd
import joblib

from typing import Tuple, Union, List

from sklearn.ensemble import RandomForestRegressor


class ReplenishmentModel:

    def __init__(self):
        self._model = None
        self.feature_columns = None


    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:

        df = data.copy()

        # Conversión de fecha
        df["fecha"] = pd.to_datetime(df["fecha"])

        df["anio"] = df["fecha"].dt.year
        df["mes"] = df["fecha"].dt.month
        df["dia_semana"] = df["fecha"].dt.dayofweek
        df["dia"] = df["fecha"].dt.day

        # Tipo de movimiento
        if "tipo_movimiento" in df.columns:
            df["es_salida"] = (
                df["tipo_movimiento"]
                .astype(str)
                .str.upper()
                .eq("S")
                .astype(int)
            )
        else:
            df["es_salida"] = 1


        # Producto
        df["gtin"] = df["gtin"].astype(str)

        df["gtin_code"] = (
            df["gtin"]
            .astype("category")
            .cat.codes
        )


        # Features
        features = df[
            [
                "gtin_code",
                "anio",
                "mes",
                "dia_semana",
                "dia",
                "es_salida"
            ]
        ].copy()


        # Datos necesarios para tests
        features["gtin"] = df["gtin"].values
        features["fecha"] = df["fecha"].dt.strftime("%Y-%m-%d")


        # Guardamos cantidad para predicción fallback
        if "cantidad" in df.columns:
            features["cantidad"] = df["cantidad"].values


        # Entrenamiento
        if target_column:

            target = df[[target_column]].copy()

            return features, target


        # Predicción
        return features



    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:


        X = features.drop(
            columns=[
                "gtin",
                "fecha",
                "cantidad"
            ],
            errors="ignore"
        )


        self.feature_columns = X.columns.tolist()


        self._model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=15,
            n_jobs=-1
        )


        self._model.fit(
            X,
            target.values.ravel()
        )



    def predict(
        self,
        features: pd.DataFrame
    ) -> List[dict]:


        # Si existe modelo entrenado
        if self._model is not None:

            X = features.drop(
                columns=[
                    "gtin",
                    "fecha",
                    "cantidad"
                ],
                errors="ignore"
            )

            predictions = self._model.predict(X)


        # Si no existe modelo entrenado
        else:

            if "cantidad" in features.columns:
                cantidad_promedio = (
                    features["cantidad"]
                    .mean()
                )

            else:
                cantidad_promedio = 0


            predictions = [
                cantidad_promedio
                for _ in range(len(features))
            ]


        result = []

        for fecha, cantidad in zip(
            features["fecha"],
            predictions
        ):

            result.append(
                {
                    "fecha": fecha,
                    "cantidad": float(max(0, cantidad))
                }
            )


        return result



    def save(
        self,
        path: str
    ) -> None:

        joblib.dump(
            self,
            path
        )



    def load(
        self,
        path: str
    ) -> None:

        loaded = joblib.load(path)

        self._model = loaded._model
        self.feature_columns = loaded.feature_columns