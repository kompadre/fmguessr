# fmguessr Makefile

.PHONY: all match index play server release docker-build test clean

all: match index play server

match:
	go build -o match ./cmd/match/main.go

index:
	go build -o index ./cmd/index/main.go

play:
	go build -o play ./cmd/play/main.go

server:
	go build -o server ./cmd/server/main.go

release:
	CGO_ENABLED=1 go build -ldflags="-s -w" -trimpath -o server ./cmd/server/

docker-build:
	docker compose build

test: all
	go test -v ./tests/...

clean:
	rm -f match index play server test_match.wav test_out.syx test_codebook.bin
