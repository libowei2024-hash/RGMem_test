#l1_memory/utils/_init_.py

# 导入我们的日志设置函数
from .logger import setup_logging

# 调用设置函数，并获取返回的logger实例
# 这使得在整个应用程序的生命周期中，我们只有一个配置好的logger实例
logger = setup_logging()