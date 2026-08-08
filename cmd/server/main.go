package main

import (
	"context"
	"encoding/base64"
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
	Port           int
	MaxQueue       int
	Generations    int
	Population     int
	ModelPath      string
	CodebookPath   string
	ClassifierPath string
	ClassifierOnly bool
	BudgetSeconds  float64
	ExportDir      string
	ModelData      []byte // Embedded ONNX model bytes (overrides ModelPath)
	EncoderPath    string // Path to 1D ResNet ONNX encoder (optional)
	OptMargin      float64 // Skip optimization on candidates worse than best * (1 + margin)
}

type MatchResult struct {
	Rank      int     `json:"rank"`
	Algorithm int     `json:"algorithm"`
	Feedback  int     `json:"feedback"`
	Distance  float64 `json:"distance"`
	WavURL    string  `json:"wav_url"`
	VoiceData string  `json:"voice_data,omitempty"` // base64-encoded 128-byte patch
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
	Favorites       []MatchResult `json:"favorites,omitempty"`
	FormantMatches  []MatchResult `json:"formant_matches,omitempty"`

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
	sessionsDir string
	encoder   *match.AudioEncoder // Loaded once at startup, nil if not provided
}

func NewServer(cfg ServerConfig) *Server {
	absExport, _ := filepath.Abs(cfg.ExportDir)
	_ = os.MkdirAll(absExport, 0755)
	sessionsDir := filepath.Join(absExport, "sessions")
	_ = os.MkdirAll(sessionsDir, 0755)

	s := &Server{
		jobs:        make(map[string]*Job),
		jobQueue:    make(chan *Job, cfg.MaxQueue),
		cfg:         cfg,
		exportDir:   absExport,
		sessionsDir: sessionsDir,
	}

	// Load encoder model if provided
	if cfg.EncoderPath != "" {
		enc, err := match.NewAudioEncoder(cfg.EncoderPath)
		if err != nil {
			log.Printf("Warning: encoder model not loaded: %v (trajectory distance disabled)", err)
		} else {
			s.encoder = enc
			log.Printf("Loaded encoder model: %s", cfg.EncoderPath)
		}
	}

	// Start single-worker processing goroutine
	go s.workerLoop()

	// Start background cleanup ticker
	go s.cleanupLoop()

	return s
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
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

	// Trajectory distance: compute or load from shared session cache
	var trajectorySpec *match.TrajectorySpec
	var lfoSpectrum []float64
	var phaseBounds match.PhaseBounds
	if s.encoder != nil {
		audioHash := match.AudioHash(targetSamples)
		sessionPath := filepath.Join(s.sessionsDir, fmt.Sprintf("%x.bin", audioHash))

		// Try to load existing session
		if session, err := match.LoadSession(sessionPath); err == nil && session.AudioHash == audioHash {
			trajectorySpec = session.Trajectory
			lfoSpectrum = session.LFOSpectrum
			phaseBounds = session.PhaseBounds
			if job.overrides.Note != "" {
				log.Printf("Job %s: loaded cached trajectory session", job.ID)
			}
		} else {
			// Compute trajectory
			traj := match.ExtractTrajectory(s.encoder, targetNorm)
			if traj != nil {
				phaseBounds = match.DetectPhaseBounds(targetSamples, job.sampleRate)
				lfoSpectrum = match.ComputeLFOSpectrum(traj)
				trajectorySpec = traj

				// Save to shared sessions directory (atomic write)
				if err := match.SaveSession(sessionPath, &match.SessionData{
					AudioHash:   audioHash,
					Trajectory:  traj,
					PhaseBounds: phaseBounds,
					LFOSpectrum: lfoSpectrum,
				}); err != nil {
					log.Printf("Job %s: warning: could not save session: %v", job.ID, err)
				}
			}
		}
	}

	// Configure GA
	gaCfg := match.DefaultGAConfig()
	gaCfg.Generations = s.cfg.Generations
	gaCfg.Population = s.cfg.Population

	// Set up context with budget
	var ctx context.Context
	if s.cfg.BudgetSeconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), time.Duration(s.cfg.BudgetSeconds*float64(time.Second)))
		defer cancel()
		log.Printf("Job %s: budget set to %.1fs", job.ID, s.cfg.BudgetSeconds)
	} else {
		ctx = context.Background()
	}

	matchCfg := match.Config{
		RenderDur:       2 * time.Second,
		MIDINote:        uint8(midiNote),
		NoteFreq:        noteFreq,
		CodebookPath:    s.cfg.CodebookPath,
		ModelPath:       s.cfg.ModelPath,
		ModelData:       s.cfg.ModelData,
		ClassifierPath:  s.cfg.ClassifierPath,
		ClassifierOnly:  s.cfg.ClassifierOnly,
		Verbose:         true,
		GA:              gaCfg,
		Ctx:             ctx,
		Encoder:           s.encoder,
		TargetTrajectory:  trajectorySpec,
		TargetLFOSpectrum: lfoSpectrum,
		PhaseBounds:       phaseBounds,
		TrajectoryAlphaStart: 0.1,
		TrajectoryAlphaEnd:   0.8,
		TrajectoryConfig:     match.DefaultTrajectoryConfig(),
		OptMargin:            s.cfg.OptMargin,
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
		candDist := match.MelCosineDistanceCached(match.ComputeTargetMel(targetNorm, match.RenderSampleRate), candNorm, match.RenderSampleRate)

		matches = append(matches, MatchResult{
			Rank:      rank + 1,
			Algorithm: int(cand.Algorithm + 1),
			Feedback:  int(cand.Feedback),
			Distance:  candDist,
			WavURL:    fmt.Sprintf("/exports/%s/match_%d.wav", job.ID, rank+1),
			VoiceData: base64.StdEncoding.EncodeToString(cand.Encode()),
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
		candDist := match.MelCosineDistanceCached(match.ComputeTargetMel(targetNorm, match.RenderSampleRate), candNorm, match.RenderSampleRate)

		timbralMatches = append(timbralMatches, MatchResult{
			Rank:      rank + 1,
			Algorithm: int(cand.Algorithm + 1),
			Feedback:  int(cand.Feedback),
			Distance:  candDist,
			WavURL:    fmt.Sprintf("/exports/%s/timbral_%d.wav", job.ID, rank+1),
			VoiceData: base64.StdEncoding.EncodeToString(cand.Encode()),
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
		candDist := match.MelCosineDistanceCached(match.ComputeTargetMel(targetNorm, match.RenderSampleRate), candNorm, match.RenderSampleRate)

		fullMatches = append(fullMatches, MatchResult{
			Rank:      rank + 1,
			Algorithm: int(cand.Algorithm + 1),
			Feedback:  int(cand.Feedback),
			Distance:  candDist,
			WavURL:    fmt.Sprintf("/exports/%s/full_%d.wav", job.ID, rank+1),
			VoiceData: base64.StdEncoding.EncodeToString(cand.Encode()),
		})
	}

	// Render Formant Top 5 to WAV files
	var formantMatches []MatchResult
	for rank, cand := range result.FormantTop5 {
		candAudio := match.RenderCandidateWithGate(cand, uint8(midiNote), gateDur, matchCfg.RenderDur)
		wavPath := filepath.Join(jobDir, fmt.Sprintf("formant_%d.wav", rank+1))

		if err := match.SaveWav(wavPath, candAudio, match.RenderSampleRate); err != nil {
			s.failJob(job, fmt.Sprintf("failed to save formant candidate WAV: %v", err))
			return
		}

		candNorm := match.NormalizeRMS(match.TrimOrPad(candAudio, len(targetNorm)))
		candDist := match.MelCosineDistanceCached(match.ComputeTargetMel(targetNorm, match.RenderSampleRate), candNorm, match.RenderSampleRate)

		formantMatches = append(formantMatches, MatchResult{
			Rank:      rank + 1,
			Algorithm: int(cand.Algorithm + 1),
			Feedback:  int(cand.Feedback),
			Distance:  candDist,
			WavURL:    fmt.Sprintf("/exports/%s/formant_%d.wav", job.ID, rank+1),
			VoiceData: base64.StdEncoding.EncodeToString(cand.Encode()),
		})
	}

	// Build Favorites: candidates in both TimbralTop5 and FullTop5
	var favoriteMatches []MatchResult
	fullSigs := make(map[string]bool, len(result.FullTop5))
	for _, c := range result.FullTop5 {
		fullSigs[string(c.Encode())] = true
	}
	for _, c := range result.TimbralTop5 {
		if fullSigs[string(c.Encode())] {
			favAudio := match.RenderCandidateWithGate(c, uint8(midiNote), gateDur, matchCfg.RenderDur)
			favID := fmt.Sprintf("fav_%d", len(favoriteMatches)+1)
			favPath := filepath.Join(jobDir, favID+".wav")
			if err := match.SaveWav(favPath, favAudio, match.RenderSampleRate); err != nil {
				continue
			}
			favNorm := match.NormalizeRMS(match.TrimOrPad(favAudio, len(targetNorm)))
			favDist := match.MelCosineDistanceCached(match.ComputeTargetMel(targetNorm, match.RenderSampleRate), favNorm, match.RenderSampleRate)
			favoriteMatches = append(favoriteMatches, MatchResult{
				Rank:      len(favoriteMatches) + 1,
				Algorithm: int(c.Algorithm + 1),
				Feedback:  int(c.Feedback),
				Distance:  favDist,
				WavURL:    fmt.Sprintf("/exports/%s/%s.wav", job.ID, favID),
			})
		}
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
	job.FormantMatches = formantMatches
	job.Favorites = favoriteMatches
	s.mu.Unlock()

	// Persist job result for recovery after server restart
	resultPath := filepath.Join(jobDir, "result.json")
	if jsonData, err := json.MarshalIndent(job, "", "  "); err == nil {
		_ = os.WriteFile(resultPath, jsonData, 0644)
	}
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

		// Prune stale session files (>1 hour old)
		if s.sessionsDir != "" {
			sessionFiles, _ := filepath.Glob(filepath.Join(s.sessionsDir, "*.bin"))
			for _, f := range sessionFiles {
				info, err := os.Stat(f)
				if err == nil && now.Sub(info.ModTime()) > 1*time.Hour {
					os.Remove(f)
				}
			}
		}
	}
}

// HTTP Route Handlers

func (s *Server) handleMatch(w http.ResponseWriter, r *http.Request) {
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
	jobID := strings.TrimPrefix(r.URL.Path, "/api/status/")
	if jobID == "" {
		http.Error(w, `{"error":"Missing job ID"}`, http.StatusBadRequest)
		return
	}

	s.mu.RLock()
	job, exists := s.jobs[jobID]
	if !exists {
		s.mu.RUnlock()
		// Job not in memory — check for persisted result (server may have restarted)
		resultPath := filepath.Join(s.exportDir, jobID, "result.json")
		if data, err := os.ReadFile(resultPath); err == nil {
			var recovered Job
			if jsonErr := json.Unmarshal(data, &recovered); jsonErr == nil {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(recovered)
				return
			}
		}
		// Check if job directory exists (job was created but not completed)
		jobDir := filepath.Join(s.exportDir, jobID)
		if _, err := os.Stat(jobDir); err == nil {
			// Directory exists but no result.json — job was interrupted
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]string{
				"job_id": jobID,
				"status": "failed",
				"error":  "Server restarted before job completed",
			})
			return
		}
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

func (s *Server) handlePlay(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST required"}`, http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		SyxURL     string `json:"syx_url"`
		Slot       int    `json:"slot"`
		Note       int    `json:"note"`
		VoiceData  string `json:"voice_data"`     // base64-encoded 128-byte patch (alternative to syx_url+slot)
		DurationMs int    `json:"duration_ms"`     // optional: note duration in ms (from piano roll)
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 8000)).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}

	if req.Slot < 0 || req.Slot > 31 {
		http.Error(w, `{"error":"slot must be 0-31"}`, http.StatusBadRequest)
		return
	}
	if req.Note < 0 || req.Note > 127 {
		http.Error(w, `{"error":"note must be 0-127"}`, http.StatusBadRequest)
		return
	}

	// Extract 128-byte voice patch
	var patch []byte
	if req.VoiceData != "" {
		// Decode base64 voice data directly
		var decErr error
		patch, decErr = match.DecodeBase64(req.VoiceData)
		if decErr != nil || len(patch) != 128 {
			http.Error(w, `{"error":"invalid voice_data (must be base64-encoded 128 bytes)"}`, http.StatusBadRequest)
			return
		}
	} else {
		// Read from .syx file
		syxPath := filepath.Join(s.exportDir, strings.TrimPrefix(req.SyxURL, "/exports/"))
		syxData, err := os.ReadFile(syxPath)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":"cannot read .syx: %v"}`, err), http.StatusNotFound)
			return
		}
		patchStart := 6 + req.Slot*128
		patchEnd := patchStart + 128
		if patchEnd > len(syxData) {
			http.Error(w, `{"error":".syx file too short for requested slot"}`, http.StatusBadRequest)
			return
		}
		patch = syxData[patchStart:patchEnd]
	}

	// Decode into Candidate and render
	cand := match.DecodeVoicePatch(patch)
	gateDur := 500 * time.Millisecond
	totalDur := 3 * time.Second
	if req.DurationMs > 0 {
		durMs := req.DurationMs
		if durMs < 300 {
			durMs = 300
		}
		gateDur = time.Duration(float64(durMs)*0.6) * time.Millisecond
		totalDur = time.Duration(durMs) * time.Millisecond
	}
	audio := match.RenderCandidateWithGate(cand, uint8(req.Note), gateDur, totalDur)

	// Save WAV to exports
	playID := fmt.Sprintf("play_%d", time.Now().UnixNano())
	playDir := filepath.Join(s.exportDir, playID)
	os.MkdirAll(playDir, 0755)
	wavPath := filepath.Join(playDir, "play.wav")
	if err := match.SaveWav(wavPath, audio, match.RenderSampleRate); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"save wav: %v"}`, err), http.StatusInternalServerError)
		return
	}

	wavURL := fmt.Sprintf("/exports/%s/play.wav", playID)
	_ = json.NewEncoder(w).Encode(map[string]string{"wav_url": wavURL})
}

func (s *Server) handleExports(w http.ResponseWriter, r *http.Request) {
	fs := http.StripPrefix("/exports/", http.FileServer(http.Dir(s.exportDir)))
	fs.ServeHTTP(w, r)
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	http.ServeFile(w, r, "index.html")
}

func (s *Server) handleStatic(w http.ResponseWriter, r *http.Request) {
	fs := http.StripPrefix("/html/", http.FileServer(http.Dir("html")))
	fs.ServeHTTP(w, r)
}

func (s *Server) handleVoice(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST required"}`, http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		SyxURL string `json:"syx_url"`
		Slot   int    `json:"slot"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 8000)).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}

	syxPath := filepath.Join(s.exportDir, strings.TrimPrefix(req.SyxURL, "/exports/"))
	syxData, err := os.ReadFile(syxPath)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"cannot read .syx: %v"}`, err), http.StatusNotFound)
		return
	}

	patchStart := 6 + req.Slot*128
	patchEnd := patchStart + 128
	if patchEnd > len(syxData) {
		http.Error(w, `{"error":".syx file too short for requested slot"}`, http.StatusBadRequest)
		return
	}

	patch := syxData[patchStart:patchEnd]
	cand := match.DecodeVoicePatch(patch)

	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"algorithm":  cand.Algorithm,
		"feedback":   cand.Feedback,
		"voice_data": patch,
	})
}

func (s *Server) handleMorph(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST required"}`, http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		SourceData string  `json:"source_data"` // base64-encoded 128-byte voice patch
		TargetData string  `json:"target_data"` // base64-encoded 128-byte voice patch
		Note       int     `json:"note"`
		Ratio      float64 `json:"ratio"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 8000)).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}

	if len(req.SourceData) == 0 || len(req.TargetData) == 0 {
		http.Error(w, `{"error":"source_data and target_data required"}`, http.StatusBadRequest)
		return
	}

	srcBytes, err := match.DecodeBase64(req.SourceData)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"invalid source_data: %v"}`, err), http.StatusBadRequest)
		return
	}
	tgtBytes, err := match.DecodeBase64(req.TargetData)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"invalid target_data: %v"}`, err), http.StatusBadRequest)
		return
	}

	if len(srcBytes) != 128 {
		http.Error(w, fmt.Sprintf(`{"error":"source must be 128 bytes, got %d"}`, len(srcBytes)), http.StatusBadRequest)
		return
	}
	if len(tgtBytes) != 128 {
		http.Error(w, fmt.Sprintf(`{"error":"target must be 128 bytes, got %d"}`, len(tgtBytes)), http.StatusBadRequest)
		return
	}

	source := match.DecodeVoicePatch(srcBytes)
	target := match.DecodeVoicePatch(tgtBytes)

	note := req.Note
	if note == 0 {
		note = 60
	}
	ratio := req.Ratio
	if ratio < -1 || ratio > 1 {
		ratio = 0.5
	}

	result := match.Morph(match.MorphConfig{
		Source:  source,
		Target:  target,
		Note:    uint8(note),
		Dur:     2 * time.Second,
		Ratio:   ratio,
		MaxIter: 100,
		Verbose: false,
	})

	// Export morphed patch as .syx
	morphID := fmt.Sprintf("morph_%d", time.Now().UnixNano())
	morphDir := filepath.Join(s.exportDir, morphID)
	os.MkdirAll(morphDir, 0755)

	syxPath := filepath.Join(morphDir, "morphed.syx")
	syxData := result.Morphed.SysexBulkDump()
	if err := os.WriteFile(syxPath, syxData, 0644); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"failed to write .syx: %v"}`, err), http.StatusInternalServerError)
		return
	}

	// Render the morphed patch for preview
	morphedAudio := match.RenderCandidateWithGate(result.Morphed, uint8(note), 500*time.Millisecond, 2*time.Second)
	wavPath := filepath.Join(morphDir, "morphed.wav")
	if err := match.SaveWav(wavPath, morphedAudio, match.RenderSampleRate); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"save wav: %v"}`, err), http.StatusInternalServerError)
		return
	}

	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"voice_data":   base64.StdEncoding.EncodeToString(result.Morphed.Encode()),
		"syx_url":      fmt.Sprintf("/exports/%s/morphed.syx", morphID),
		"wav_url":      fmt.Sprintf("/exports/%s/morphed.wav", morphID),
		"source_dist":  result.SourceDist,
		"morph_dist":   result.MorphDist,
		"ratio":        result.TargetRatio,
		"alpha":        result.Alpha,
		"iterations":   result.Iterations,
	})
}

func (s *Server) handleExportSyx(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST required"}`, http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Patches []struct {
			SyxURL    string `json:"syx_url"`
			Slot      int    `json:"slot"`
			Name      string `json:"name"`
			VoiceData string `json:"voice_data"` // base64-encoded 128-byte patch (alternative to syx_url)
		} `json:"patches"`
		Name string `json:"name"` // optional: export filename (without .syx)
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 80000)).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}
	if len(req.Patches) == 0 || len(req.Patches) > 32 {
		http.Error(w, `{"error":"need 1-32 patches"}`, http.StatusBadRequest)
		return
	}

	// Collect candidates from source .syx files or voice_data
	var candidates []*match.Candidate
	var names []string
	for _, p := range req.Patches {
		var voiceData []byte
		var err error

		if p.VoiceData != "" {
			voiceData, err = match.DecodeBase64(p.VoiceData)
			if err != nil || len(voiceData) != 128 {
				http.Error(w, fmt.Sprintf(`{"error":"invalid voice_data for patch %q"}`, p.Name), http.StatusBadRequest)
				return
			}
		} else {
			syxPath := filepath.Join(s.exportDir, strings.TrimPrefix(p.SyxURL, "/exports/"))
			syxData, err := os.ReadFile(syxPath)
			if err != nil {
				http.Error(w, fmt.Sprintf(`{"error":"cannot read .syx: %v"}`, err), http.StatusNotFound)
				return
			}
			patchStart := 6 + p.Slot*128
			patchEnd := patchStart + 128
			if patchEnd > len(syxData) {
				http.Error(w, `{"error":".syx file too short for requested slot"}`, http.StatusBadRequest)
				return
			}
			voiceData = syxData[patchStart:patchEnd]
		}
		cand := match.DecodeVoicePatch(voiceData)
		candidates = append(candidates, cand)

		name := p.Name
		if name == "" {
			name = cand.Name
		}
		names = append(names, name)
	}

	// Sanitize and deduplicate names
	sanitized := make([]string, len(names))
	for i, n := range names {
		sanitized[i] = match.SanitizeVoiceName(n)
	}
	sanitized = match.DeduplicateVoiceNames(sanitized)
	for i, c := range candidates {
		c.Name = sanitized[i]
	}

	// Generate combined .syx bank
	syxData := match.SysexBulkDumpMultiple(candidates)

	// Save to exports
	exportID := fmt.Sprintf("export_%d", time.Now().UnixMilli())
	exportDir := filepath.Join(s.exportDir, exportID)
	_ = os.MkdirAll(exportDir, 0755)
	exportName := "patches"
	if req.Name != "" {
		exportName = match.SanitizeVoiceName(req.Name)
	}
	syxPath := filepath.Join(exportDir, exportName+".syx")
	if err := os.WriteFile(syxPath, syxData, 0644); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"write failed: %v"}`, err), http.StatusInternalServerError)
		return
	}

	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"syx_url": fmt.Sprintf("/exports/%s/%s.syx", exportID, exportName),
		"count":   len(candidates),
	})
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
	classifierPath := flag.String("classifier", "", "path to patch classifier ONNX model (optional)")
	classifierOnly := flag.Bool("classifier-only", false, "use classifier predictions only, skip GA")
	budgetSeconds := flag.Float64("budget-seconds", 0, "stop matching after N seconds (0=unlimited)")
	exportDir := flag.String("exports", "./exports", "path to export directory for static files")
	encoderPath := flag.String("encoder", "", "path to 1D ResNet ONNX encoder model for trajectory distance (optional)")
	optMargin := flag.Float64("opt-margin", 0.3, "skip optimization on candidates worse than best * (1 + margin)")

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
		Port:           *port,
		MaxQueue:       *maxQueue,
		Generations:    *generations,
		Population:     *population,
		ModelPath:      *modelPath,
		CodebookPath:   *codebookPath,
		ClassifierPath: *classifierPath,
		ClassifierOnly: *classifierOnly,
		BudgetSeconds:  *budgetSeconds,
		ExportDir:      *exportDir,
		ModelData:      modelData,
		EncoderPath:    *encoderPath,
		OptMargin:      *optMargin,
	}

	server := NewServer(cfg)

	mux := http.NewServeMux()
	mux.HandleFunc("/api/match", server.handleMatch)
	mux.HandleFunc("/api/probe", server.handleProbe)
	mux.HandleFunc("/api/play", server.handlePlay)
	mux.HandleFunc("/api/voice", server.handleVoice)
	mux.HandleFunc("/api/morph", server.handleMorph)
	mux.HandleFunc("/api/export-syx", server.handleExportSyx)
	mux.HandleFunc("/api/status/", server.handleStatus)
	mux.HandleFunc("/api/health", server.handleHealth)
	mux.HandleFunc("/exports/", server.handleExports)
	mux.HandleFunc("/html/", server.handleStatic)
	mux.HandleFunc("/", server.handleIndex)

	fmt.Printf("fmguessr REST API Server listening on http://localhost:%d\n", cfg.Port)
	fmt.Printf("  Max Queue: %d, Generations: %d, Population: %d\n", cfg.MaxQueue, cfg.Generations, cfg.Population)
	if cfg.BudgetSeconds > 0 {
		fmt.Printf("  Budget: %.1fs\n", cfg.BudgetSeconds)
	}
	fmt.Printf("  Codebook: %s (exists: %v)\n", cfg.CodebookPath, fileExists(cfg.CodebookPath))
	fmt.Printf("  Exports Dir: %s\n", server.exportDir)

	if err := http.ListenAndServe(fmt.Sprintf(":%d", cfg.Port), corsMiddleware(mux)); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}
