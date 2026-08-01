# fmguessr Makefile

.PHONY: all match index play server test clean

all: match index play server

match:
	go build -o match ./cmd/match/main.go

index:
	go build -o index ./cmd/index/main.go

play:
	go build -o play ./cmd/play/main.go

server:
	go build -o server ./cmd/server/main.go

test: all
	go test -v ./tests/...

clean:
	rm -f match index play server test_match.wav test_out.syx test_codebook.bin
