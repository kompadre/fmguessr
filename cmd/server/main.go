package main

import (
	"encoding/json"
	_ "embed"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/kompadre/fmguessr/pkg/match"
)

//go:embed data/model.onnx
var defaultModelData []byte

type ServerConfig struct {
	Port         int
	MaxQueue     int
	Generations  int
	Population   int
	ModelPath    string
	CodebookPath string
	ExportDir    string
	ModelData    []byte // Embedded ONNX model bytes (overrides ModelPath)
}

type MatchResult struct {
	Rank      int     `json:"rank"`
	Algorithm int     `json:"algorithm"`
	Feedback  int     `json:"feedback"`
	Distance  float64 `json:"distance"`
	WavURL    string  `json:"wav_url"`
}

type Job struct {
	ID              string        `json:"job_id"`
	Status          string        `json:"status"` // "queued", "processing", "completed", "failed"
	QueuePosition   int           `json:"queue_position,omitempty"`
	ProgressPercent int           `json:"progress_percent"`
	Phase           string        `json:"phase,omitempty"`
	Error           string        `json:"error,omitempty"`
	RootNote        string        `json:"root_note,omitempty"`
	MIDINote        int           `json:"midi_note,omitempty"`
	F0Hz            float64       `json:"f0_hz,omitempty"`
	BestDistance    float64       `json:"best_distance,omitempty"`
	SyxURL          string        `json:"syx_url,omitempty"`
	Matches         []MatchResult `json:"matches,omitempty"`
	TimbralMatches  []MatchResult `json:"timbral_matches,omitempty"`
	FullMatches     []MatchResult `json:"full_matches,omitempty"`

	// Internal fields
	audioSamples []float64
	sampleRate   int
	overrides    match.FeatureOverrides
	createdAt    time.Time
}

type Server struct {
	mu        sync.RWMutex
	jobs      map[string]*Job
	jobQueue  chan *Job
	cfg       ServerConfig
	exportDir string
}

func NewServer(cfg ServerConfig) *Server {
	absExport, _ := filepath.Abs(cfg.ExportDir)
	_ = os.MkdirAll(absExport, 0755)

	s := &Server{
		jobs:      make(map[string]*Job),
		jobQueue:  make(chan *Job, cfg.MaxQueue),
		cfg:       cfg,
		exportDir: absExport,
	}

	// Start single-worker processing goroutine
	go s.workerLoop()

	// Start background cleanup ticker
	go s.cleanupLoop()

	return s
}

// workerLoop processes jobs sequentially one at a time
func (s *Server) workerLoop() {
	for job := range s.jobQueue {
		s.processJob(job)
	}
}

func (s *Server) processJob(job *Job) {
	s.mu.Lock()
	job.Status = "processing"
	job.ProgressPercent = 5
	job.Phase = "Initializing Matcher"
	s.mu.Unlock()

	jobDir := filepath.Join(s.exportDir, job.ID)
	if err := os.MkdirAll(jobDir, 0755); err != nil {
		s.failJob(job, fmt.Sprintf("failed to create export directory: %v", err))
		return
	}

	// Truncate PCM audio to max 4.0 seconds
	maxSamples := int(4.0 * float64(job.sampleRate))
	targetSamples := job.audioSamples
	if len(targetSamples) > maxSamples {
		targetSamples = targetSamples[:maxSamples]
	}

	features := match.ExtractFeatures(targetSamples, job.sampleRate)
	targetNorm := match.NormalizeRMS(targetSamples)
	features, midiNote := match.ApplyFeatureOverrides(features, job.overrides)
	noteFreq := 440.0 * math.Pow(2, float64(midiNote-69)/12.0)

	// Configure GA
	gaCfg := match.DefaultGAConfig()
	gaCfg.Generations = s.cfg.Generations
	gaCfg.Population = s.cfg.Population

	matchCfg := match.Config{
		RenderDur:    2 * time.Second,
		MIDINote:     uint8(midiNote),
		NoteFreq:     noteFreq,
		CodebookPath: s.cfg.CodebookPath,
		ModelPath:    s.cfg.ModelPath,
		ModelData:    s.cfg.ModelData,
		Verbose:      true,
		GA:           gaCfg,
		ProgressCallback: func(pct int, phase string) {
			s.mu.Lock()
			job.ProgressPercent = pct
			job.Phase = phase
			s.mu.Unlock()
		},
	}

	// Run matching engine
	result := match.Run(targetNorm, features, matchCfg)

	// Auto-index best match into codebook by default
	gateDur := time.Duration(features.GateMs * float64(time.Millisecond))
	if result.Best != nil {
		patchName := job.ID
		if len(patchName) > 16 {
			patchName = patchName[:16]
		}
		if err := match.IndexCandidateToCodebook(result.Best, patchName, uint8(midiNote), gateDur, matchCfg.RenderDur, matchCfg.CodebookPath); err != nil {
			log.Printf("Job %s: warning: failed to auto-index match into codebook: %v", job.ID, err)
		} else {
			log.Printf("Job %s: auto-indexed best match '%s' into codebook %s", job.ID, patchName, matchCfg.CodebookPath)
		}
	}

	// Export .syx bulk dump containing top 32 matches
	syxPath := filepath.Join(jobDir, "bank.syx")
	syxData := match.SysexBulkDumpMultiple(result.Top)
	if err := os.WriteFile(syxPath, syxData, 0644); err != nil {
		s.failJob(job, fmt.Sprintf("failed to export .syx bank: %v", err))
		return
	}

	// Render Top 5 matches to WAV files using auto-detected key hold gate duration
	var matches []MatchResult
	for rank := 0; rank < 5 && rank < len(result.Top); rank++ {
		cand := result.Top[rank]

		candAudio := match.RenderCandidateWithGate(cand, uint8(midiNote), gateDur, matchCfg.RenderDur)
		candWavPath := filepath.Join(jobDir, fmt.Sprintf("match_%d.wav", rank+1))

		if err := match.SaveWav(candWavPath, candAudio, match.RenderSampleRate); err != nil {
			s.failJob(job, fmt.Sprintf("failed to save candidate WAV: %v", err))
			return
		}

		// Compute spectral distance for this candidate
		candNorm := match.NormalizeRMS(match.TrimOrPad(candAudio, len(targetNorm)))
		candDist := match.SpectralDistance(targetNorm, candNorm)

		matches = append(matches, MatchResult{
			Rank:      rank + 1,
			Algorithm: int(cand.Algorithm + 1),
			Feedback:  int(cand.Feedback),
			Distance:  candDist,
			WavURL:    fmt.Sprintf("/exports/%s/match_%d.wav", job.ID, rank+1),
		})
	}

	// Render Timbral Top 5 to WAV files
	var timbralMatches []MatchResult
	for rank, cand := range result.TimbralTop5 {
		candAudio := match.RenderCandidateWithGate(cand, uint8(midiNote), gateDur, matchCfg.RenderDur)
		wavPath := filepath.Join(jobDir, fmt.Sprintf("timbral_%d.wav", rank+1))

		if err := match.SaveWav(wavPath, candAudio, match.RenderSampleRate); err != nil {
			s.failJob(job, fmt.Sprintf("failed to save timbral candidate WAV: %v", err))
			return
		}

		candNorm := match.NormalizeRMS(match.TrimOrPad(candAudio, len(targetNorm)))
		candDist := match.SpectralDistance(targetNorm, candNorm)

		timbralMatches = append(timbralMatches, MatchResult{
			Rank:      rank + 1,
			Algorithm: int(cand.Algorithm + 1),
			Feedback:  int(cand.Feedback),
			Distance:  candDist,
			WavURL:    fmt.Sprintf("/exports/%s/timbral_%d.wav", job.ID, rank+1),
		})
	}

	// Render Full Top 5 to WAV files
	var fullMatches []MatchResult
	for rank, cand := range result.FullTop5 {
		candAudio := match.RenderCandidateWithGate(cand, uint8(midiNote), gateDur, matchCfg.RenderDur)
		wavPath := filepath.Join(jobDir, fmt.Sprintf("full_%d.wav", rank+1))

		if err := match.SaveWav(wavPath, candAudio, match.RenderSampleRate); err != nil {
			s.failJob(job, fmt.Sprintf("failed to save full candidate WAV: %v", err))
			return
		}

		candNorm := match.NormalizeRMS(match.TrimOrPad(candAudio, len(targetNorm)))
		candDist := match.SpectralDistance(targetNorm, candNorm)

		fullMatches = append(fullMatches, MatchResult{
			Rank:      rank + 1,
			Algorithm: int(cand.Algorithm + 1),
			Feedback:  int(cand.Feedback),
			Distance:  candDist,
			WavURL:    fmt.Sprintf("/exports/%s/full_%d.wav", job.ID, rank+1),
		})
	}

	s.mu.Lock()
	job.Status = "completed"
	job.ProgressPercent = 100
	job.Phase = "Completed"
	job.RootNote = match.MidiNoteName(midiNote)
	job.MIDINote = midiNote
	job.F0Hz = features.F0
	job.BestDistance = result.BestDist
	job.SyxURL = fmt.Sprintf("/exports/%s/bank.syx", job.ID)
	job.Matches = matches
	job.TimbralMatches = timbralMatches
	job.FullMatches = fullMatches
	s.mu.Unlock()
}

func (s *Server) failJob(job *Job, errMsg string) {
	s.mu.Lock()
	job.Status = "failed"
	job.Error = errMsg
	s.mu.Unlock()
}

func (s *Server) cleanupLoop() {
	ticker := time.NewTicker(30 * time.Minute)
	for range ticker.C {
		s.mu.Lock()
		now := time.Now()
		for id, job := range s.jobs {
			if now.Sub(job.createdAt) > 1*time.Hour {
				delete(s.jobs, id)
				jobDir := filepath.Join(s.exportDir, id)
				_ = os.RemoveAll(jobDir)
			}
		}
		s.mu.Unlock()
	}
}

// HTTP Route Handlers

func (s *Server) handleMatch(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	// Aggressive Sanitization: Max 800 KB payload limit (4s @ 48kHz stereo 16-bit)
	maxPayloadBytes := int64(800 * 1024)
	r.Body = http.MaxBytesReader(w, r.Body, maxPayloadBytes)

	// Read multipart form file
	err := r.ParseMultipartForm(maxPayloadBytes)
	if err != nil {
		if hj, ok := w.(http.Hijacker); ok {
			conn, bufrw, hjErr := hj.Hijack()
			if hjErr == nil {
				// Force a hard TCP Reset (RST) on Close instead of a graceful FIN packet
				if tcpConn, ok := conn.(*net.TCPConn); ok {
					tcpConn.SetLinger(0)
				}

				// 1. Write HTTP status line & response body manually
				body := `{"error":"File too large (> 800 KB) or invalid form"}`
				res := "HTTP/1.1 413 Payload Too Large\r\n" +
					"Content-Type: application/json\r\n" +
					"Connection: close\r\n" +
					fmt.Sprintf("Content-Length: %d\r\n\r\n", len(body)) +
					body

				bufrw.WriteString(res)
				bufrw.Flush() // Ensure bytes hit the wire immediately

				// 2. Kill the socket instantly
				conn.Close()
				return
			}
		}

		// Fallback if Hijacker isn't supported (e.g. HTTP/2 or standard middleware wrappers)
		http.Error(w, `{"error":"File too large (> 800 KB) or invalid form"}`, http.StatusRequestEntityTooLarge)
		return
	}
	file, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, `{"error":"Missing 'file' field in form-data"}`, http.StatusBadRequest)
		return
	}
	defer file.Close()

	wavBytes, err := io.ReadAll(file)
	if err != nil {
		http.Error(w, `{"error":"Failed to read uploaded file"}`, http.StatusBadRequest)
		return
	}

	// Validate & decode WAV bytes
	samples, sampleRate, err := match.DecodeWavBytes(wavBytes)
	if err != nil || len(samples) < 512 {
		http.Error(w, `{"error":"Invalid or corrupted WAV file"}`, http.StatusBadRequest)
		return
	}

	// Check Queue Capacity
	s.mu.Lock()
	queuedCount := len(s.jobQueue)
	if queuedCount >= s.cfg.MaxQueue {
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error":        "Queue full. Server is currently processing another request.",
			"queue_length": queuedCount,
			"max_queue":    s.cfg.MaxQueue,
		})
		return
	}

	overrides := parseFormOverrides(r)

	jobID := fmt.Sprintf("job_%d_%d", time.Now().Unix(), time.Now().Nanosecond()%1000000)
	job := &Job{
		ID:            jobID,
		Status:        "queued",
		QueuePosition: queuedCount + 1,
		audioSamples:  samples,
		sampleRate:    sampleRate,
		overrides:     overrides,
		createdAt:     time.Now(),
	}

	s.jobs[jobID] = job
	s.jobQueue <- job
	s.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"job_id":         jobID,
		"status":         "queued",
		"queue_position": queuedCount + 1,
		"status_url":     fmt.Sprintf("/api/status/%s", jobID),
	})
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	jobID := strings.TrimPrefix(r.URL.Path, "/api/status/")
	if jobID == "" {
		http.Error(w, `{"error":"Missing job ID"}`, http.StatusBadRequest)
		return
	}

	s.mu.RLock()
	job, exists := s.jobs[jobID]
	if !exists {
		s.mu.RUnlock()
		http.Error(w, `{"error":"Job not found"}`, http.StatusNotFound)
		return
	}

	// Calculate queue position if queued
	if job.Status == "queued" {
		pos := 1
		for _, j := range s.jobs {
			if j.Status == "queued" && j.createdAt.Before(job.createdAt) {
				pos++
			}
		}
		job.QueuePosition = pos
	}
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(job)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")

	s.mu.RLock()
	queueLen := len(s.jobQueue)
	s.mu.RUnlock()

	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status":          "ok",
		"queue_length":    queueLen,
		"max_queue":       s.cfg.MaxQueue,
		"model_loaded":    s.cfg.ModelPath != "" && fileExists(s.cfg.ModelPath),
		"codebook_loaded": s.cfg.CodebookPath != "" && fileExists(s.cfg.CodebookPath),
	})
}

func parseFormOverrides(r *http.Request) match.FeatureOverrides {
	var ov match.FeatureOverrides

	note := r.FormValue("note")
	if note == "" {
		note = r.FormValue("root_note")
	}
	if note == "" {
		note = r.FormValue("midi_note")
	}
	ov.Note = note

	if val := r.FormValue("attack_ms"); val != "" {
		if floatVal, err := strconv.ParseFloat(val, 64); err == nil {
			ov.AttackMs = &floatVal
		}
	}
	if val := r.FormValue("decay_ms"); val != "" {
		if floatVal, err := strconv.ParseFloat(val, 64); err == nil {
			ov.DecayMs = &floatVal
		}
	}
	if val := r.FormValue("sustain_level"); val != "" {
		if floatVal, err := strconv.ParseFloat(val, 64); err == nil {
			ov.SustainLevel = &floatVal
		}
	}
	if val := r.FormValue("release_ms"); val != "" {
		if floatVal, err := strconv.ParseFloat(val, 64); err == nil {
			ov.ReleaseMs = &floatVal
		}
	}
	if val := r.FormValue("gate_ms"); val != "" {
		if floatVal, err := strconv.ParseFloat(val, 64); err == nil {
			ov.GateMs = &floatVal
		}
	}
	if val := r.FormValue("brightness"); val != "" {
		if floatVal, err := strconv.ParseFloat(val, 64); err == nil {
			ov.Brightness = &floatVal
		}
	}
	if val := r.FormValue("harmonicity"); val != "" {
		if floatVal, err := strconv.ParseFloat(val, 64); err == nil {
			ov.Harmonicity = &floatVal
		}
	}

	return ov
}

func (s *Server) handleProbe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	maxPayloadBytes := int64(800 * 1024)
	r.Body = http.MaxBytesReader(w, r.Body, maxPayloadBytes)

	var wavBytes []byte
	var overrides match.FeatureOverrides

	contentType := r.Header.Get("Content-Type")
	if strings.HasPrefix(contentType, "multipart/form-data") {
		if err := r.ParseMultipartForm(maxPayloadBytes); err == nil {
			overrides = parseFormOverrides(r)
			file, _, err := r.FormFile("file")
			if err == nil {
				defer file.Close()
				wavBytes, _ = io.ReadAll(file)
			}
		}
	} else {
		wavBytes, _ = io.ReadAll(r.Body)
	}

	if len(wavBytes) == 0 {
		http.Error(w, `{"error":"Missing file or empty WAV body"}`, http.StatusBadRequest)
		return
	}

	samples, sampleRate, err := match.DecodeWavBytes(wavBytes)
	if err != nil || len(samples) < 512 {
		http.Error(w, `{"error":"Invalid or corrupted WAV file"}`, http.StatusBadRequest)
		return
	}

	res := match.ProbeAudio(samples, sampleRate, s.cfg.CodebookPath, overrides)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(res)
}

func (s *Server) handleExports(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	fs := http.StripPrefix("/exports/", http.FileServer(http.Dir(s.exportDir)))
	fs.ServeHTTP(w, r)
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

func main() {
	port := flag.Int("port", 8080, "server listening port")
	maxQueue := flag.Int("max-queue", 5, "maximum pending jobs in queue")
	generations := flag.Int("generations", 20, "GA generations")
	population := flag.Int("population", 64, "GA population size")
	modelPath := flag.String("model", "", "path to ONNX model file (default: embedded)")
	codebookPath := flag.String("codebook", "dx7_codebook.bin", "path to pre-indexed dx7_codebook.bin")
	exportDir := flag.String("exports", "./exports", "path to export directory for static files")

	flag.Parse()

	// Model: use embedded data unless -model flag is provided
	var modelData []byte
	if *modelPath != "" {
		if !fileExists(*modelPath) {
			log.Fatalf("FATAL: model file not found: %s\n", *modelPath)
		}
		fmt.Printf("  Model: %s (file)\n", *modelPath)
	} else {
		modelData = defaultModelData
		fmt.Printf("  Model: embedded (%d bytes)\n", len(modelData))
	}

	// Codebook: fail-fast if fingerprint codebook is missing
	if *codebookPath == "" {
		fmt.Fprintln(os.Stderr, "Warning: no codebook path specified (-codebook)")
	} else if !fileExists(*codebookPath) {
		log.Fatalf("FATAL: fingerprint codebook not found: %s\n"+
			"  Run ./index to build it, or pass -codebook flag\n", *codebookPath)
	} else {
		// Check timbre codebook (warn only)
		timbrePath := *codebookPath
		if len(*codebookPath) > 4 && (*codebookPath)[len(*codebookPath)-4:] == ".bin" {
			timbrePath = (*codebookPath)[:len(*codebookPath)-4] + "_timbre.bin"
		} else {
			timbrePath = *codebookPath + "_timbre.bin"
		}
		if !fileExists(timbrePath) {
			fmt.Fprintf(os.Stderr, "Warning: timbre codebook not found: %s\n"+
				"  Run ./index --sync-timbre to create it\n", timbrePath)
		}
	}

	cfg := ServerConfig{
		Port:         *port,
		MaxQueue:     *maxQueue,
		Generations:  *generations,
		Population:   *population,
		ModelPath:    *modelPath,
		CodebookPath: *codebookPath,
		ExportDir:    *exportDir,
		ModelData:    modelData,
	}

	server := NewServer(cfg)

	mux := http.NewServeMux()
	mux.HandleFunc("/api/match", server.handleMatch)
	mux.HandleFunc("/api/probe", server.handleProbe)
	mux.HandleFunc("/api/status/", server.handleStatus)
	mux.HandleFunc("/api/health", server.handleHealth)
	mux.HandleFunc("/exports/", server.handleExports)

	fmt.Printf("fmguessr REST API Server listening on http://localhost:%d\n", cfg.Port)
	fmt.Printf("  Max Queue: %d, Generations: %d, Population: %d\n", cfg.MaxQueue, cfg.Generations, cfg.Population)
	fmt.Printf("  Codebook: %s (exists: %v)\n", cfg.CodebookPath, fileExists(cfg.CodebookPath))
	fmt.Printf("  Exports Dir: %s\n", server.exportDir)

	if err := http.ListenAndServe(fmt.Sprintf(":%d", cfg.Port), mux); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}
