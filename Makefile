# fmguessr Makefile

.PHONY: all match index play morph server release deploy docker-build test test-unit test-e2e clean

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

deploy: release
	@echo "=== Staging deploy ==="
	@mkdir -p deploy/app/lib deploy/app/html/components deploy/app/html/synth/wasm deploy/app/exports
	cp server deploy/app/
	cp index.html deploy/app/
	cp -r html/style.css html/app.js deploy/app/html/
	cp -r html/components/*.js deploy/app/html/components/
	cp -r html/synth/dx7-synth.js html/synth/dx7-processor.js html/synth/audio-manager.js deploy/app/html/synth/
	cp -r html/synth/wasm/msfa.wasm deploy/app/html/synth/wasm/
	cp lib/libonnxruntime.so deploy/app/lib/
	test -f dx7_codebook.bin && cp dx7_codebook.bin deploy/app/ || echo "WARNING: dx7_codebook.bin not found"
	test -f dx7_codebook_timbre.bin && cp dx7_codebook_timbre.bin deploy/app/ || echo "WARNING: dx7_codebook_timbre.bin not found"
	@echo "=== Deploy staged in deploy/app/ ==="
	@ls -lh deploy/app/

docker-build: deploy
	docker build -t fmguessr -f deploy/Dockerfile deploy/app/

docker-up:
	cd deploy && docker compose up -d

docker-down:
	cd deploy && docker compose down

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
	rm -rf deploy/app
