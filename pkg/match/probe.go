package match

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// FeatureOverrides holds user-supplied overrides for sensible target audio features.
type FeatureOverrides struct {
	Note         string   // MIDI note string ("C3") or number string ("60")
	AttackMs     *float64 // Attack time override in ms
	DecayMs      *float64 // Decay time override in ms
	SustainLevel *float64 // Sustain level override (0.0 - 1.0)
	ReleaseMs    *float64 // Release time override in ms
	GateMs       *float64 // Key hold gate duration override in ms
	Brightness   *float64 // Brightness score override (0.0 - 1.0)
	Harmonicity  *float64 // Harmonicity score override (0.0 - 1.0)
}

// ProbeResult holds detected envelope, timing, pitch, and formant analysis.
type ProbeResult struct {
	// Envelope & Timing
	AttackMs     float64 `json:"attack_ms"`
	DecayMs      float64 `json:"decay_ms"`
	SustainLevel float64 `json:"sustain_level"`
	ReleaseMs    float64 `json:"release_ms"`
	GateMs       float64 `json:"gate_ms"`
	NoteOnMs     float64 `json:"note_on_ms"`
	NoteOffMs    float64 `json:"note_off_ms"`

	// Pitch & Spectral Heuristics
	F0Hz               float64 `json:"f0_hz"`
	F0Confidence       float64 `json:"f0_confidence"`
	MIDINote           int     `json:"midi_note"`
	RootNote           string  `json:"root_note"`
	Brightness         float64 `json:"brightness"`
	Harmonicity        float64 `json:"harmonicity"`
	SpectralCentroidHz float64 `json:"spectral_centroid_hz"`

	// Formant Analysis (dominant spectral resonances)
	F1Hz float64 `json:"f1_hz,omitempty"`
	F2Hz float64 `json:"f2_hz,omitempty"`
	F3Hz float64 `json:"f3_hz,omitempty"`
}

// ParseNoteStringOrNumber converts a musical note string (e.g., "C3", "A4", "60") or number to MIDI note 0-127 and note frequency in Hz.
func ParseNoteStringOrNumber(raw string) (uint8, float64, error) {
	s := strings.TrimSpace(strings.ToUpper(raw))
	if s == "" {
		return 60, 261.63, fmt.Errorf("empty note string")
	}

	// Try numeric parse first
	if val, err := strconv.Atoi(s); err == nil {
		if val < 0 {
			val = 0
		}
		if val > 127 {
			val = 127
		}
		freq := 440.0 * math.Pow(2, float64(val-69)/12.0)
		return uint8(val), freq, nil
	}

	// Parse musical string: e.g. "C3", "C#3", "Db3"
	notes := map[string]int{
		"C": 0, "C#": 1, "DB": 1,
		"D": 2, "D#": 3, "EB": 3,
		"E": 4,
		"F": 5, "F#": 6, "GB": 6,
		"G": 7, "G#": 8, "AB": 8,
		"A": 9, "A#": 10, "BB": 10,
		"B": 11,
	}

	var pitchStr string
	var octaveStr string

	if len(s) >= 2 && (s[1] == '#' || s[1] == 'B') {
		pitchStr = s[:2]
		octaveStr = s[2:]
	} else if len(s) >= 1 {
		pitchStr = s[:1]
		octaveStr = s[1:]
	}

	pitchClass, ok := notes[pitchStr]
	if !ok {
		return 60, 261.63, fmt.Errorf("invalid note pitch class: %s", pitchStr)
	}

	octave := 4 // Default C4
	if octaveStr != "" {
		if oVal, err := strconv.Atoi(octaveStr); err == nil {
			octave = oVal
		}
	}

	midiNote := (octave + 1) * 12 + pitchClass
	if midiNote < 0 {
		midiNote = 0
	}
	if midiNote > 127 {
		midiNote = 127
	}

	freq := 440.0 * math.Pow(2, float64(midiNote-69)/12.0)
	return uint8(midiNote), freq, nil
}

// ApplyFeatureOverrides applies non-nil user feature overrides onto TargetFeatures.
func ApplyFeatureOverrides(f TargetFeatures, overrides FeatureOverrides) (TargetFeatures, int) {
	midiNote := 60
	if f.F0 > 20 && f.F0Confidence > 0.3 {
		midiNote = 69 + int(math.Round(12.0*math.Log2(f.F0/440.0)))
		if midiNote < 0 {
			midiNote = 0
		}
		if midiNote > 127 {
			midiNote = 127
		}
	}

	if overrides.Note != "" {
		if noteNum, freq, err := ParseNoteStringOrNumber(overrides.Note); err == nil {
			midiNote = int(noteNum)
			f.F0 = freq
			f.F0Confidence = 1.0
		}
	}

	if overrides.AttackMs != nil {
		f.AttackMs = *overrides.AttackMs
	}
	if overrides.DecayMs != nil {
		f.DecayMs = *overrides.DecayMs
	}
	if overrides.SustainLevel != nil {
		f.SustainLevel = *overrides.SustainLevel
	}
	if overrides.ReleaseMs != nil {
		f.ReleaseMs = *overrides.ReleaseMs
	}
	if overrides.GateMs != nil {
		f.GateMs = *overrides.GateMs
	}
	if overrides.Brightness != nil {
		f.BrightnessScore = *overrides.Brightness
	}
	if overrides.Harmonicity != nil {
		f.Harmonicity = *overrides.Harmonicity
	}

	return f, midiNote
}

// ProbeAudio analyzes audio samples and computes envelope, pitch, and formant features.
// Codebook matching and "seen before" detection are handled by the match pipeline (CodebookSearch step).
func ProbeAudio(samples []float64, sampleRate int, overrides FeatureOverrides) ProbeResult {
	norm := NormalizeRMS(samples)
	features := ExtractFeatures(norm, sampleRate)
	features, midiNote := ApplyFeatureOverrides(features, overrides)

	noteOnMs := float64(features.AttackPeakSample-int(features.AttackMs*float64(sampleRate)/1000.0)) * 1000.0 / float64(sampleRate)
	if noteOnMs < 0 {
		noteOnMs = 0
	}
	noteOffMs := noteOnMs + features.GateMs

	res := ProbeResult{
		AttackMs:           features.AttackMs,
		DecayMs:            features.DecayMs,
		SustainLevel:       features.SustainLevel,
		ReleaseMs:          features.ReleaseMs,
		GateMs:             features.GateMs,
		NoteOnMs:           noteOnMs,
		NoteOffMs:          noteOffMs,
		F0Hz:               features.F0,
		F0Confidence:       features.F0Confidence,
		MIDINote:           midiNote,
		RootNote:           MidiNoteName(midiNote),
		Brightness:         features.BrightnessScore,
		Harmonicity:        features.Harmonicity,
		SpectralCentroidHz: features.SpectralCentroidHz,
	}

	// Formant extraction (~1ms)
	formants := ExtractFormants(norm, sampleRate, 3)
	if len(formants) > 0 {
		res.F1Hz = formants[0].Frequency
	}
	if len(formants) > 1 {
		res.F2Hz = formants[1].Frequency
	}
	if len(formants) > 2 {
		res.F3Hz = formants[2].Frequency
	}

	return res
}
