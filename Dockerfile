FROM python:3.12.8-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /bin/

RUN apt-get update \n    && apt-get install -y --no-install-recommends git \n    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "--no-dev", "python", "-m", "my_ds_project.train"]
