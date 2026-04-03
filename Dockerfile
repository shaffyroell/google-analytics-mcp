FROM python:3.12-slim

WORKDIR /app

# Copy project metadata first so pip can resolve dependencies before copying
# the rest of the source (better layer caching).
COPY pyproject.toml README.md ./
COPY analytics_mcp/ analytics_mcp/

# Install the package with the web extras.
RUN pip install --no-cache-dir ".[web]"

# Railway injects $PORT at runtime; default to 8080 for local testing.
ENV PORT=8080

EXPOSE 8080

CMD ["analytics-mcp-web"]
