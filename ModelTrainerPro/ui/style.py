"""
Модуль стилей приложения ModelTrainerPro.
Содержит стили и цветовую палитру для UI.
"""

# Основная цветовая палитра в фиолетовых тонах
COLOR_PRIMARY = "#6A0DAD"  # Насыщенный фиолетовый для акцентов
COLOR_PRIMARY_LIGHT = "#9370DB"  # Светло-фиолетовый для кнопок и элементов
COLOR_PRIMARY_DARK = "#483D8B"  # Темно-фиолетовый для заголовков
COLOR_BACKGROUND = "#F5F0FF"  # Очень светлый фиолетовый фон
COLOR_TEXT = "#2E0854"  # Темно-фиолетовый для текста
COLOR_SECONDARY = "#E6E6FA"  # Лавандовый для второстепенных элементов
COLOR_HIGHLIGHT = "#8A2BE2"  # Яркий фиолетовый для выделений
COLOR_SUCCESS = "#4CAF50"  # Зеленый для успешных операций
COLOR_ERROR = "#F44336"  # Красный для ошибок
COLOR_WARNING = "#FFC107"  # Желтый для предупреждений

# CSS стили для приложения
# Обновим стили для принудительной светлой темы
STYLESHEET = f"""
QMainWindow, QDialog, QWidget {{
    background-color: {COLOR_BACKGROUND};
    color: {COLOR_TEXT};
}}

QLabel {{
    color: {COLOR_TEXT};
}}

QLabel#header {{
    font-size: 16pt;
    font-weight: bold;
    color: {COLOR_PRIMARY_DARK};
}}

QLabel#subheader {{
    font-size: 12pt;
    font-weight: bold;
    color: {COLOR_PRIMARY_DARK};
}}

QPushButton {{
    background-color: {COLOR_PRIMARY_LIGHT};
    color: white;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}}

QPushButton#helpButton {{
    background-color: {COLOR_PRIMARY_DARK};
    color: white;
    border-radius: 15px;
    padding: 0px;
    font-weight: bold;
    font-size: 16px;
}}

QPushButton#helpButton:hover {{
    background-color: {COLOR_PRIMARY};
}}

QPushButton#helpButton:pressed {{
    background-color: {COLOR_HIGHLIGHT};
}}

QPushButton:hover {{
    background-color: {COLOR_PRIMARY};
}}

QPushButton:pressed {{
    background-color: {COLOR_PRIMARY_DARK};
}}

QPushButton:disabled {{
    background-color: #CCCCCC;
    color: #666666;
}}

QComboBox {{
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    border-radius: 4px;
    padding: 5px;
    background-color: white;
    color: {COLOR_TEXT};
}}

QComboBox:hover {{
    border: 1px solid {COLOR_PRIMARY};
}}

QComboBox::drop-down {{
    border: 0px;
}}

QComboBox::down-arrow {{
    image: url(down_arrow.png);
    width: 12px;
    height: 12px;
}}

QComboBox QAbstractItemView {{
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    selection-background-color: {COLOR_PRIMARY_LIGHT};
    background-color: white;
    color: {COLOR_TEXT};
}}

QLineEdit, QTextEdit, QPlainTextEdit {{
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    border-radius: 4px;
    padding: 5px;
    background-color: white;
    color: {COLOR_TEXT};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {COLOR_PRIMARY};
}}

QProgressBar {{
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    border-radius: 4px;
    background-color: white;
    text-align: center;
    color: {COLOR_TEXT};
}}

QProgressBar::chunk {{
    background-color: {COLOR_PRIMARY_LIGHT};
}}

QGroupBox {{
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    border-radius: 4px;
    margin-top: 10px;
    font-weight: bold;
    color: {COLOR_PRIMARY_DARK};
    background-color: {COLOR_BACKGROUND};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
    background-color: {COLOR_BACKGROUND};
}}

QTabWidget::pane {{
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    background-color: white;
}}

QTabBar::tab {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    border-bottom-color: {COLOR_PRIMARY_LIGHT};
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 5px 10px;
}}

QTabBar::tab:selected, QTabBar::tab:hover {{
    background-color: {COLOR_PRIMARY_LIGHT};
    color: white;
}}

QCheckBox {{
    color: {COLOR_TEXT};
    background-color: transparent;
}}

QCheckBox::indicator {{
    width: 13px;
    height: 13px;
}}

QRadioButton {{
    color: {COLOR_TEXT};
    background-color: transparent;
}}

QRadioButton::indicator {{
    width: 13px;
    height: 13px;
}}

QTableView {{
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    gridline-color: {COLOR_PRIMARY_LIGHT};
    selection-background-color: {COLOR_PRIMARY_LIGHT};
    background-color: white;
    color: {COLOR_TEXT};
    alternate-background-color: #F0F0FF;  
}}

QHeaderView::section {{
    background-color: {COLOR_PRIMARY_LIGHT};
    color: white;
    padding: 4px;
    border: 1px solid {COLOR_PRIMARY_LIGHT};
}}

QStatusBar {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_TEXT};
}}

QMenuBar {{
    background-color: {COLOR_SECONDARY};
    color: {COLOR_TEXT};
}}

QMenuBar::item {{
    padding: 2px 10px;
    background-color: transparent;
}}

QMenuBar::item:selected {{
    background-color: {COLOR_PRIMARY_LIGHT};
    color: white;
}}

QMenu {{
    background-color: white;
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    color: {COLOR_TEXT};
}}

QMenu::item {{
    padding: 5px 30px 5px 10px;
}}

QMenu::item:selected {{
    background-color: {COLOR_PRIMARY_LIGHT};
    color: white;
}}

QToolTip {{
    background-color: {COLOR_PRIMARY_DARK};
    color: white;
    border: none;
}}

QScrollArea {{
    background-color: white;
    color: {COLOR_TEXT};
}}

QScrollBar:vertical {{
    border: 1px solid #C4C4C4;
    background: white;
    width: 15px;
    margin: 16px 0 16px 0;
}}

QScrollBar::handle:vertical {{
    background: {COLOR_PRIMARY_LIGHT};
    min-height: 20px;
}}

QScrollBar::add-line:vertical {{
    border: 1px solid #C4C4C4;
    background: white;
    height: 15px;
    subcontrol-position: bottom;
    subcontrol-origin: margin;
}}

QScrollBar::sub-line:vertical {{
    border: 1px solid #C4C4C4;
    background: white;
    height: 15px;
    subcontrol-position: top;
    subcontrol-origin: margin;
}}

QScrollBar:horizontal {{
    border: 1px solid #C4C4C4;
    background: white;
    height: 15px;
    margin: 0 16px 0 16px;
}}

QScrollBar::handle:horizontal {{
    background: {COLOR_PRIMARY_LIGHT};
    min-width: 20px;
}}

QScrollBar::add-line:horizontal {{
    border: 1px solid #C4C4C4;
    background: white;
    width: 15px;
    subcontrol-position: right;
    subcontrol-origin: margin;
}}

QScrollBar::sub-line:horizontal {{
    border: 1px solid #C4C4C4;
    background: white;
    width: 15px;
    subcontrol-position: left;
    subcontrol-origin: margin;
}}

QSpinBox, QDoubleSpinBox {{
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    border-radius: 4px;
    padding: 5px;
    background-color: white;
    color: {COLOR_TEXT};
}}

QDateEdit, QTimeEdit, QDateTimeEdit {{
    border: 1px solid {COLOR_PRIMARY_LIGHT};
    border-radius: 4px;
    padding: 5px;
    background-color: white;
    color: {COLOR_TEXT};
}}

/* Для всех виджетов, для которых не указаны явные стили */
* {{
    background-color: {COLOR_BACKGROUND};
    color: {COLOR_TEXT};
}}
"""

# Функция для получения стилей
def get_stylesheet() -> str:
    """
    Возвращает CSS стили для приложения.
    
    Returns:
        str: CSS стили
    """
    return STYLESHEET