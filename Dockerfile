FROM python:3.12.8-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /bin/

WORKDIR /app
COPY . .
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "--no-dev", "python", "-m", "my_ds_project.train"]
