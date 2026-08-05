# fmguessr Makefile

.PHONY: all match index play morph server release docker-build test test-unit test-e2e clean

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

test: test-unit test-e2e

test-unit:
	go test -v -count=1 -timeout 600s ./pkg/match/...

test-e2e: all
	go test -v -count=1 -timeout 600s ./tests/...

test-smoke:
	go test -v -count=1 -timeout 300s -run 'TestSmoke' ./tests/...

test-regression:
	go test -v -count=1 -timeout 60s -tags regression -run 'TestRegression' ./tests/...

clean:
	rm -f match index play morph server test_match.wav test_out.syx test_codebook.bin
