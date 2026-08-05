FROM python:3.14-slim

# Core scientific deps compile-free via manylinux wheels; git not needed in the
# image (the host owns the clone and bind-mounts it in).
COPY app/requirements.txt /tmp/requirements.txt

# Bloomberg libs are useless on the VPS (no Terminal) and their wheels may not
# exist for this platform — install them best-effort so a failure never blocks
# the build. Everything else must install cleanly.
RUN grep -vE '^(--extra-index-url|blpapi|xbbg)' /tmp/requirements.txt > /tmp/req_core.txt \
    && pip install --no-cache-dir -r /tmp/req_core.txt
RUN pip install --no-cache-dir --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi || true
RUN pip install --no-cache-dir xbbg || true

# Report PDFs (src/reportkit.py) render via headless Chromium. The playwright pip package alone
# doesn't include the browser binary or the OS-level shared libraries it needs to actually run
# (this base image is missing them) -- --with-deps installs both the apt packages and the
# Chromium build in one step, so PDF generation works out of the box on the VPS.
RUN python -m playwright install --with-deps chromium

WORKDIR /app
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.headless", "true", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--browser.gatherUsageStats", "false"]
