import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    """Класс для загрузки и обработки данных."""
    
    def __init__(self):
        """Инициализация загрузчика данных."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_column = None
        self.features = None
        self.scaler = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Загружает данные из CSV файла.
        
        Args:
            file_path (str): Путь к CSV файлу
            
        Returns:
            pd.DataFrame: Загруженные данные
            
        Raises:
            ValueError: Если файл не существует или имеет неподдерживаемый формат
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Неподдерживаемый формат файла. Используйте CSV или Excel.")
            
            # Замена пропущенных значений на NaN
            self.data = self.data.replace('-', np.nan)
            
            # Преобразование всех столбцов с числовыми данными
            for col in self.data.columns:
                if col != 'Дата':  # Пропускаем столбец с датой
                    try:
                        self.data[col] = pd.to_numeric(self.data[col])
                    except:
                        pass  # Если не удалось преобразовать, оставляем как есть
            
            return self.data
        except Exception as e:
            raise ValueError(f"Ошибка при загрузке данных: {str(e)}")
    
    def get_column_names(self) -> List[str]:
        """
        Возвращает список имен столбцов.
        
        Returns:
            List[str]: Список имен столбцов
            
        Raises:
            ValueError: Если данные не загружены
        """
        if self.data is None:
            raise ValueError("Данные не загружены")
        
        return list(self.data.columns)
    
    def prepare_data(self, target_column: str, test_size: float = 0.2, 
                    random_state: int = 42, scale_data: bool = True) -> Dict[str, Any]:
        """
        Подготавливает данные для обучения модели.
        
        Args:
            target_column (str): Имя целевого столбца (зависимая переменная)
            test_size (float, optional): Доля тестовых данных. По умолчанию 0.2.
            random_state (int, optional): Начальное значение для генератора случайных чисел. По умолчанию 42.
            scale_data (bool, optional): Флаг масштабирования данных. По умолчанию True.
            
        Returns:
            Dict[str, Any]: Словарь с обучающими и тестовыми данными
            
        Raises:
            ValueError: Если данные не загружены или целевой столбец не найден
        """
        if self.data is None:
            raise ValueError("Данные не загружены")
        
        if target_column not in self.data.columns:
            raise ValueError(f"Целевой столбец '{target_column}' не найден в данных")
        
        self.target_column = target_column
        
        # Отделяем признаки от целевой переменной
        X = self.data.drop(columns=[target_column])
        
        # Если есть столбец с датой, удаляем его
        if 'Дата' in X.columns:
            X = X.drop(columns=['Дата'])
        
        # Удаляем строки с пропущенными значениями
        X = X.dropna()
        y = self.data.loc[X.index, target_column]
        
        # Сохраняем названия признаков
        self.features = list(X.columns)
        
        # Разделяем данные на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Масштабирование данных, если требуется
        if scale_data:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'features': self.features,
            'target': self.target_column,
            'scaler': self.scaler
        }
    
    def transform_new_data(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Преобразует новые данные в формат, соответствующий обученной модели.
        
        Args:
            new_data (pd.DataFrame): Новые данные для преобразования
            
        Returns:
            np.ndarray: Преобразованные данные
            
        Raises:
            ValueError: Если модель не обучена или данные имеют неверный формат
        """
        if self.scaler is None or self.features is None:
            raise ValueError("Данные для обучения не были подготовлены")
        
        # Проверяем наличие необходимых признаков
        missing_features = [f for f in self.features if f not in new_data.columns]
        if missing_features:
            raise ValueError(f"В новых данных отсутствуют следующие признаки: {', '.join(missing_features)}")
        
        # Выбираем только нужные признаки в том же порядке
        X_new = new_data[self.features]
        
        # Применяем масштабирование, если оно было использовано
        if self.scaler:
            X_new = self.scaler.transform(X_new)
        
        return X_new
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводную информацию о данных.
        
        Returns:
            Dict[str, Any]: Словарь со сводной информацией
            
        Raises:
            ValueError: Если данные не загружены
        """
        if self.data is None:
            raise ValueError("Данные не загружены")
        
        # Получаем базовую статистику
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        summary = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'numeric_columns': len(numeric_data.columns),
            'non_numeric_columns': len(self.data.columns) - len(numeric_data.columns),
            'missing_values': self.data.isna().sum().sum(),
            'column_types': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'basic_stats': numeric_data.describe().to_dict()
        }
        
        return summary
