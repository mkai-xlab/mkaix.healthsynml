FROM python:3.12-alpine

WORKDIR /viewer

COPY index.html styles.css app.js ./

USER 65532:65532

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=3s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080', timeout=2)"]

CMD ["python", "-m", "http.server", "8080", "--bind", "0.0.0.0"]
