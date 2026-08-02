FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server .
COPY lib/libonnxruntime.so ./lib/
COPY lib/libonnxruntime_providers_shared.so ./lib/

RUN chmod +x server

EXPOSE 8080

ENTRYPOINT ["./server"]
CMD ["-port", "8080", "-generations", "20", "-population", "64"]
