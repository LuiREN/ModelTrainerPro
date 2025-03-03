"""
Модуль, содержащий классы моделей машинного обучения.
"""
import os
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import datetime

class BaseModel:
    """
    Базовый класс для всех моделей машинного обучения.
    
    Attributes:
        name (str): Название модели
        model: Объект модели машинного обучения
        hyperparams (Dict[str, Any]): Словарь с гиперпараметрами модели
        is_trained (bool): Флаг, указывающий обучена ли модель
        results (Dict[str, float]): Словарь с результатами тестирования модели
    """
    
    def __init__(self, name: str) -> None:
        """
        Инициализация базовой модели.
        
        Args:
            name (str): Название модели
        """
        self.name: str = name
        self.model: Any = None
        self.hyperparams: Dict[str, Any] = {}
        self.is_trained: bool = False
        self.results: Dict[str, float] = {}
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Обучение модели.
        
        Args:
            X_train (np.ndarray): Обучающие признаки
            y_train (np.ndarray): Обучающие целевые значения
        """
        raise NotImplementedError("Subclass must implement abstract method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Выполнение предсказаний.
        
        Args:
            X (np.ndarray): Данные для предсказания
            
        Returns:
            np.ndarray: Предсказанные значения
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Оценка качества модели на тестовой выборке.
        
        Args:
            X_test (np.ndarray): Тестовые признаки
            y_test (np.ndarray): Тестовые целевые значения
            
        Returns:
            Dict[str, float]: Словарь с метриками качества
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.results = {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        return self.results
    
    def save(self, path: str) -> None:
        """
        Сохранение модели в файл.
        
        Args:
            path (str): Путь для сохранения модели
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        model_info = {
            'model': self.model,
            'hyperparams': self.hyperparams,
            'name': self.name,
            'results': self.results
        }
        
        joblib.dump(model_info, path)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """
        Загрузка модели из файла.
        
        Args:
            path (str): Путь к файлу модели
            
        Returns:
            BaseModel: Загруженная модель
        """
        model_info = joblib.load(path)
        
        if "RandomForest" in model_info['name']:
            instance = RandomForestModel()
        elif "KNN" in model_info['name']:
            instance = KNNModel()
        else:
            raise ValueError(f"Неизвестный тип модели: {model_info['name']}")
        
        instance.model = model_info['model']
        instance.hyperparams = model_info['hyperparams']
        instance.name = model_info['name']
        instance.results = model_info.get('results', {})
        instance.is_trained = True
        
        return instance
    
    def save_results(self, path: str) -> None:
        """
        Сохранение результатов тестирования в текстовый файл.
        
        Args:
            path (str): Путь для сохранения результатов
        """
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_results_{timestamp}.txt"
        filepath = os.path.join(path, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Результаты тестирования модели {self.name}\n")
            f.write(f"Дата и время: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Гиперпараметры:\n")
            for param, value in self.hyperparams.items():
                f.write(f"  {param}: {value}\n")
            f.write("\nМетрики качества:\n")
            f.write(f"  MSE: {self.results.get('mse', 'N/A'):.4f}\n")
            f.write(f"  MAE: {self.results.get('mae', 'N/A'):.4f}\n")
            f.write(f"  R²: {self.results.get('r2', 'N/A'):.4f}\n")
            
        return filepath
                

class RandomForestModel(BaseModel):
    """
    Модель Random Forest Regressor.
    
    Attributes:
        default_hyperparams (Dict[str, Any]): Словарь со значениями гиперпараметров по умолчанию
    """
    
    def __init__(self) -> None:
        """Инициализация модели Random Forest."""
        super().__init__(name="RandomForest Regressor")
        self.default_hyperparams = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'auto',
            'random_state': 42
        }
        self.hyperparams = self.default_hyperparams.copy()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Обучение модели Random Forest.
        
        Args:
            X_train (np.ndarray): Обучающие признаки
            y_train (np.ndarray): Обучающие целевые значения
        """
        self.model = RandomForestRegressor(**self.hyperparams)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def get_hyperparams_info(self) -> List[Tuple[str, str, Any, List[Any]]]:
        """
        Получение информации о гиперпараметрах модели.
        
        Returns:
            List[Tuple[str, str, Any, List[Any]]]: Список кортежей с информацией о гиперпараметрах:
                (название, тип, значение по умолчанию, список возможных значений)
        """
        return [
            ('n_estimators', 'int', 100, [10, 50, 100, 200, 500]),
            ('max_depth', 'int_or_none', None, [None, 5, 10, 15, 20, 30]),
            ('min_samples_split', 'int', 2, [2, 5, 10]),
            ('min_samples_leaf', 'int', 1, [1, 2, 4]),
            ('max_features', 'str', 'auto', ['auto', 'sqrt', 'log2']),
            ('random_state', 'int', 42, [None, 42])
        ]


class KNNModel(BaseModel):
    """
    Модель K-Nearest Neighbors Regressor.
    
    Attributes:
        default_hyperparams (Dict[str, Any]): Словарь со значениями гиперпараметров по умолчанию
    """
    
    def __init__(self) -> None:
        """Инициализация модели KNN."""
        super().__init__(name="KNN Regressor")
        self.default_hyperparams = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'p': 2
        }
        self.hyperparams = self.default_hyperparams.copy()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Обучение модели KNN.
        
        Args:
            X_train (np.ndarray): Обучающие признаки
            y_train (np.ndarray): Обучающие целевые значения
        """
        self.model = KNeighborsRegressor(**self.hyperparams)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
    def get_hyperparams_info(self) -> List[Tuple[str, str, Any, List[Any]]]:
        """
        Получение информации о гиперпараметрах модели.
        
        Returns:
            List[Tuple[str, str, Any, List[Any]]]: Список кортежей с информацией о гиперпараметрах:
                (название, тип, значение по умолчанию, список возможных значений)
        """
        return [
            ('n_neighbors', 'int', 5, [3, 5, 7, 9, 11, 13, 15]),
            ('weights', 'str', 'uniform', ['uniform', 'distance']),
            ('algorithm', 'str', 'auto', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            ('p', 'int', 2, [1, 2])
        ]


def get_available_models() -> List[BaseModel]:
    """
    Получение списка доступных моделей.
    
    Returns:
        List[BaseModel]: Список доступных моделей
    """
    return [RandomForestModel(), KNNModel()]
