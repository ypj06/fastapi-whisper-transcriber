"""FastAPI 应用包。

``app.main`` 暴露 ``create_app()`` 工厂与 ``app`` 实例；
具体业务逻辑分别放在 ``app.api``（路由层）与 ``app.services``（服务层）。
"""

__version__ = "2.0.0"
__all__ = ["__version__"]
