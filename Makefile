# fmguessr Makefile

.PHONY: all match index play morph server release docker-build test clean

all: match index play morph server

match:
	go build -o match ./cmd/match/main.go

index:
	go build -o index ./cmd/index/main.go

play:
	go build -o play ./cmd/play/main.go

server:
	go build -o server ./cmd/server/main.go

morph:
	go build -o morph ./cmd/morph/main.go

release:
	CGO_ENABLED=1 go build -ldflags="-s -w" -trimpath -o server ./cmd/server/

docker-build:
	docker compose build

test: all
	go test -v ./tests/...

clean:
	rm -f match index play morph server test_match.wav test_out.syx test_codebook.bin
