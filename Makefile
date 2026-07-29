# fmguessr Makefile

.PHONY: all match index play test clean

all: match index play

match:
	go build -o match ./cmd/match/main.go

index:
	go build -o index ./cmd/index/main.go

play:
	go build -o play ./cmd/play/main.go

test: all
	go test -v ./tests/...

clean:
	rm -f match index play test_match.wav test_out.syx test_codebook.bin
