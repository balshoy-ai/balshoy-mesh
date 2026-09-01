# syntax=docker/dockerfile:1

FROM        python:3.12-slim-bookworm

ARG         UID=1000
ARG         GID=1000
ENV         PROJECTPATH=/opt/balshoy-mesh \
            UV_COMPILE_BYTECODE=1 \
            UV_LINK_MODE=copy \
            UV_PYTHON_DOWNLOADS=0 \
            PYTHONFAULTHANDLER=1 \
            PYTHONUNBUFFERED=1
RUN         --mount=type=cache,target=/var/cache/apt,sharing=locked \
            --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
            apt-get update && apt-get install -y --no-install-recommends \
            libpq-dev gcc git \
            && rm -rf /var/lib/apt/lists/*
RUN         groupadd -g "${GID}" appuser \
            && useradd -u "${UID}" -g "${GID}" -m appuser \
            && mkdir -p "${PROJECTPATH}" \
            && chown -R appuser:appuser "${PROJECTPATH}"
COPY        --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
USER        appuser
WORKDIR     "${PROJECTPATH}"
COPY        --chown=appuser:appuser pyproject.toml uv.lock ./
RUN         --mount=type=cache,target=/home/appuser/.cache/uv,sharing=locked,uid=1000,gid=1000 \
            uv sync --no-install-project
