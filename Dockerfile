# Build stage
FROM golang:1.25-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -ldflags="-s -w" -trimpath -o /server ./cmd/server/

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /server .
COPY lib/libonnxruntime.so ./lib/
COPY lib/libonnxruntime_providers_shared.so ./lib/
EXPOSE 8080
ENTRYPOINT ["./server"]
CMD ["-port", "8080", "-generations", "20", "-population", "64"]
