# 使用官方 Python 3.13 镜像作为基础镜像
FROM spotlight-registry.cn-beijing.cr.aliyuncs.com/spotlight-middleware/python:3.13-slim-build-essential

# 设置工作目录
WORKDIR /app

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 设置环境变量
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 复制 uv 配置文件
COPY pyproject.toml uv.lock ./

# 使用 uv 安装依赖
RUN uv sync --frozen --no-cache

# 复制应用代码
COPY . .

# 环境变量
ENV PYTHONPATH /app

# 暴露端口
EXPOSE 8000

# 使用 uv 运行应用
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
