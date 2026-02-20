FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY requirements.txt /app/

RUN uv venv && . /app/.venv/bin/activate
RUN uv pip install -r requirements.txt

COPY . /app
RUN uv pip install -r requirements_extra.txt


ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "train_model.py"]