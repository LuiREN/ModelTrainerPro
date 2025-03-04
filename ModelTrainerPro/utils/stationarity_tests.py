import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Dict, Any, List, Tuple, Optional, Union
import io
import base64

def adf_test(series: pd.Series) -> Dict[str, Any]:
    """
    Проводит тест Дики-Фуллера на стационарность временного ряда.
    
    Args:
        series (pd.Series): Временной ряд для проверки
        
    Returns:
        Dict[str, Any]: Словарь с результатами теста
    """
    result = adfuller(series.dropna())
    
    output = {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags': result[2],
        'observations': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05
    }
    
    return output

def kpss_test(series: pd.Series) -> Dict[str, Any]:
    """
    Проводит KPSS тест на стационарность временного ряда.
    
    Args:
        series (pd.Series): Временной ряд для проверки
        
    Returns:
        Dict[str, Any]: Словарь с результатами теста
    """
    result = kpss(series.dropna())
    
    output = {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags': result[2],
        'critical_values': result[3],
        'is_stationary': result[1] > 0.05
    }
    
    return output

def rolling_statistics(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет скользящее среднее и стандартное отклонение временного ряда.
    
    Args:
        series (pd.Series): Временной ряд
        
    Returns:
        Tuple[pd.Series, pd.Series]: Кортеж (скользящее среднее, скользящее стандартное отклонение)
    """
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()
    
    return rolling_mean, rolling_std

def plot_stationarity_tests(series: pd.Series, column_name: str) -> str:
    """
    Создает график для визуализации стационарности временного ряда.
    
    Args:
        series (pd.Series): Временной ряд
        column_name (str): Название столбца/переменной
    
    Returns:
        str: Закодированное в base64 изображение графика
    """
    # Создаем новую фигуру с большими размерами
    plt.figure(figsize=(10, 8))
    
    # Создаем разные подграфики
    plt.subplot(311)
    plt.plot(series, label='Исходный ряд')
    rolling_mean, rolling_std = rolling_statistics(series)
    plt.plot(rolling_mean, label='Скользящее среднее')
    plt.plot(rolling_std, label='Скользящее стандартное отклонение')
    plt.legend(loc='best')
    plt.title(f'Анализ стационарности {column_name}')
    
    # График ACF
    plt.subplot(323)
    plot_acf(series.dropna(), ax=plt.gca(), lags=40)
    plt.title('Автокорреляционная функция')
    
    # График PACF
    plt.subplot(324)
    plot_pacf(series.dropna(), ax=plt.gca(), lags=40)
    plt.title('Частичная автокорреляционная функция')
    
    # Гистограмма распределения
    plt.subplot(325)
    plt.hist(series.dropna(), bins=30)
    plt.title('Гистограмма распределения')
    
    # QQ-Plot
    from scipy import stats
    plt.subplot(326)
    stats.probplot(series.dropna(), plot=plt)
    plt.title('Q-Q Plot')
    
    # Настраиваем расположение
    plt.tight_layout()
    
    # Сохраняем график в буфер и кодируем в base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

def format_stationarity_report(series_name: str, adf_results: Dict[str, Any], kpss_results: Dict[str, Any]) -> str:
    """
    Форматирует результаты тестов стационарности в читаемый текст.
    
    Args:
        series_name (str): Название временного ряда
        adf_results (Dict[str, Any]): Результаты ADF теста
        kpss_results (Dict[str, Any]): Результаты KPSS теста
        
    Returns:
        str: Отформатированный текст с результатами
    """
    report = f"## Результаты тестов на стационарность для {series_name}\n\n"
    
    # ADF тест
    report += "### Тест Дики-Фуллера (ADF)\n\n"
    report += f"* Тестовая статистика: {adf_results['test_statistic']:.4f}\n"
    report += f"* p-значение: {adf_results['p_value']:.4f}\n"
    report += f"* Количество задействованных лагов: {adf_results['lags']}\n"
    report += f"* Количество наблюдений: {adf_results['observations']}\n"
    report += "* Критические значения:\n"
    
    for key, value in adf_results['critical_values'].items():
        report += f"  - {key}: {value:.4f}\n"
    
    if adf_results['is_stationary']:
        report += "\n**Вывод по ADF тесту:** Временной ряд стационарен (p-значение <= 0.05)\n\n"
    else:
        report += "\n**Вывод по ADF тесту:** Временной ряд нестационарен (p-значение > 0.05)\n\n"
    
    # KPSS тест
    report += "### KPSS тест\n\n"
    report += f"* Тестовая статистика: {kpss_results['test_statistic']:.4f}\n"
    report += f"* p-значение: {kpss_results['p_value']:.4f}\n"
    report += f"* Количество задействованных лагов: {kpss_results['lags']}\n"
    report += "* Критические значения:\n"
    
    for key, value in kpss_results['critical_values'].items():
        report += f"  - {key}: {value:.4f}\n"
    
    if kpss_results['is_stationary']:
        report += "\n**Вывод по KPSS тесту:** Временной ряд стационарен (p-значение > 0.05)\n\n"
    else:
        report += "\n**Вывод по KPSS тесту:** Временной ряд нестационарен (p-значение <= 0.05)\n\n"
    
    # Общий вывод
    if adf_results['is_stationary'] and kpss_results['is_stationary']:
        report += "### Общий вывод\n\n"
        report += "**Оба теста подтверждают, что временной ряд является стационарным.**\n"
        report += "Это хорошая новость для моделирования, так как многие алгоритмы требуют стационарных данных.\n"
    elif not adf_results['is_stationary'] and not kpss_results['is_stationary']:
        report += "### Общий вывод\n\n"
        report += "**Оба теста показывают, что временной ряд нестационарен.**\n"
        report += "Рекомендуется применить дифференцирование или другие преобразования для приведения ряда к стационарному виду.\n"
    else:
        report += "### Общий вывод\n\n"
        report += "**Результаты тестов противоречивы.**\n"
        report += "ADF тест: " + ("Стационарен" if adf_results['is_stationary'] else "Нестационарен") + "\n"
        report += "KPSS тест: " + ("Стационарен" if kpss_results['is_stationary'] else "Нестационарен") + "\n"
        report += "Рекомендуется дополнительный анализ или применение преобразований к временному ряду.\n"
    
    return report
