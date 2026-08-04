package match

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/kompadre/fmguessr/external/msfa"
)

// TimbreCodebookEntry represents a pre-computed timbre thumbnail.
type TimbreCodebookEntry struct {
	Name          string
	Candidate     *Candidate
	TimbreSpec    [TimbreSpecBins]float64
	CompositePeak PeakShape // Precomputed composite peak shape from carrier envelopes
}

// TimbreCodebook holds a collection of pre-indexed DX7 timbre thumbnails.
type TimbreCodebook struct {
	Entries []TimbreCodebookEntry
}

// AddEntriesDeduplicated appends non-duplicate entries to the timbre codebook.
// Returns the number of newly added entries.
func (tc *TimbreCodebook) AddEntriesDeduplicated(newEntries []TimbreCodebookEntry) int {
	existing := make(map[string]bool, len(tc.Entries))
	for _, entry := range tc.Entries {
		sig := timbreSignature(entry)
		existing[sig] = true
	}

	added := 0
	for _, entry := range newEntries {
		sig := timbreSignature(entry)
		if !existing[sig] {
			existing[sig] = true
			tc.Entries = append(tc.Entries, entry)
			added++
		}
	}

	return added
}

func timbreSignature(entry TimbreCodebookEntry) string {
	return string(entry.Candidate.Encode())
}

// TimbreSpecDistance computes Euclidean distance between two timbre spectra.
func TimbreSpecDistance(a, b [TimbreSpecBins]float64) float64 {
	sum := 0.0
	for i := 0; i < TimbreSpecBins; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// SearchKNNTimbre finds the k closest timbre entries by Euclidean distance.
func (tc *TimbreCodebook) SearchKNNTimbre(targetSpec [TimbreSpecBins]float64, k int) []*Candidate {
	if len(tc.Entries) == 0 {
		return nil
	}

	type matchResult struct {
		cand *Candidate
		dist float64
	}

	results := make([]matchResult, len(tc.Entries))
	for i, entry := range tc.Entries {
		results[i] = matchResult{
			cand: entry.Candidate,
			dist: TimbreSpecDistance(targetSpec, entry.TimbreSpec),
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	if k > len(results) {
		k = len(results)
	}

	cands := make([]*Candidate, k)
	for i := 0; i < k; i++ {
		cands[i] = results[i].cand.Copy()
	}

	return cands
}

// SearchKNNTimbreWithDistances finds the k closest timbre entries by Euclidean distance and returns both candidates and distances.
func (tc *TimbreCodebook) SearchKNNTimbreWithDistances(targetSpec [TimbreSpecBins]float64, k int) ([]*Candidate, []float64) {
	if len(tc.Entries) == 0 {
		return nil, nil
	}

	type matchResult struct {
		cand *Candidate
		dist float64
	}

	results := make([]matchResult, len(tc.Entries))
	for i, entry := range tc.Entries {
		results[i] = matchResult{
			cand: entry.Candidate,
			dist: TimbreSpecDistance(targetSpec, entry.TimbreSpec),
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	if k > len(results) {
		k = len(results)
	}

	cands := make([]*Candidate, k)
	dists := make([]float64, k)
	for i := 0; i < k; i++ {
		cands[i] = results[i].cand.Copy()
		dists[i] = results[i].dist
	}

	return cands, dists
}

// SearchKNNTimbrePeak finds codebook entries whose composite peak shape best matches
// the target peak shape. Returns up to maxResults entries sorted by PeakShapeDistance.
// This is a zero-render metadata filter — pure float math, ~0.01ms per entry.
func (tc *TimbreCodebook) SearchKNNTimbrePeak(targetPeak PeakShape, maxResults int) []TimbreCodebookEntry {
	if len(tc.Entries) == 0 {
		return nil
	}

	type entryDist struct {
		entry TimbreCodebookEntry
		dist  float64
	}

	results := make([]entryDist, len(tc.Entries))
	for i, entry := range tc.Entries {
		results[i] = entryDist{
			entry: entry,
			dist:  PeakShapeDistance(targetPeak, entry.CompositePeak),
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})

	if maxResults > len(results) {
		maxResults = len(results)
	}

	out := make([]TimbreCodebookEntry, maxResults)
	for i := 0; i < maxResults; i++ {
		out[i] = results[i].entry
	}
	return out
}

// Save writes the timbre codebook to a binary file.
func (tc *TimbreCodebook) Save(filePath string) error {
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	// Header: Magic "DXCT" + Version 2 + uint32 EntryCount
	header := make([]byte, 12)
	copy(header[0:4], []byte("DXCT"))
	binary.LittleEndian.PutUint32(header[4:8], 2)
	binary.LittleEndian.PutUint32(header[8:12], uint32(len(tc.Entries)))
	if _, err := f.Write(header); err != nil {
		return err
	}

	for _, entry := range tc.Entries {
		// Name (16 bytes padded)
		nameBytes := make([]byte, 16)
		copy(nameBytes, []byte(entry.Name))
		if _, err := f.Write(nameBytes); err != nil {
			return err
		}

		// Candidate 128-byte raw voice patch
		voicePatch := entry.Candidate.Encode()
		if _, err := f.Write(voicePatch); err != nil {
			return err
		}

		// TimbreSpec (256 x float64 = 2048 bytes)
		specBuf := make([]byte, TimbreSpecBins*8)
		for i, val := range entry.TimbreSpec {
			binary.LittleEndian.PutUint64(specBuf[i*8:(i+1)*8], math.Float64bits(val))
		}
		if _, err := f.Write(specBuf); err != nil {
			return err
		}

		// CompositePeak (4 x float64 = 32 bytes)
		peakBuf := make([]byte, 32)
		binary.LittleEndian.PutUint64(peakBuf[0:8], math.Float64bits(entry.CompositePeak.AttackTimeMs))
		binary.LittleEndian.PutUint64(peakBuf[8:16], math.Float64bits(entry.CompositePeak.TransientWidth))
		binary.LittleEndian.PutUint64(peakBuf[16:24], math.Float64bits(entry.CompositePeak.DecaySlope))
		binary.LittleEndian.PutUint64(peakBuf[24:32], math.Float64bits(entry.CompositePeak.CrestFactor))
		if _, err := f.Write(peakBuf); err != nil {
			return err
		}
	}

	return nil
}

// BackfillFromCodebook renders each fingerprint codebook entry through the MSFA synth
// and extracts the timbre spectrum, adding any missing entries to the timbre codebook.
// Returns the number of new entries added.
func (tc *TimbreCodebook) BackfillFromCodebook(cb *Codebook, noteNum uint8, dur time.Duration) int {
	// Build set of existing signatures in timbre codebook
	existing := make(map[string]bool, len(tc.Entries))
	for _, entry := range tc.Entries {
		existing[string(entry.Candidate.Encode())] = true
	}

	// Render each DXCB entry missing from DXCT
	added := 0
	for _, entry := range cb.Entries {
		sig := string(entry.Candidate.Encode())
		if existing[sig] {
			continue
		}

		audio := RenderCandidate(entry.Candidate, noteNum, dur)
		timbreSpec := ExtractTimbreWindow(audio, RenderSampleRate)
		tc.Entries = append(tc.Entries, TimbreCodebookEntry{
			Name:          entry.Name,
			Candidate:     entry.Candidate.Copy(),
			TimbreSpec:    timbreSpec,
			CompositePeak: CompositePeakShape(entry.Candidate.Algorithm, entry.Candidate.Ops),
		})
		existing[sig] = true
		added++
	}
	return added
}

// LoadTimbreCodebook reads a timbre codebook from a binary file.
func LoadTimbreCodebook(filePath string) (*TimbreCodebook, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 12)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, err
	}

	if string(header[0:4]) != "DXCT" {
		return nil, fmt.Errorf("invalid timbre codebook magic: %s", header[0:4])
	}

	version := binary.LittleEndian.Uint32(header[4:8])
	count := binary.LittleEndian.Uint32(header[8:12])
	tc := &TimbreCodebook{Entries: make([]TimbreCodebookEntry, 0, count)}

	for i := uint32(0); i < count; i++ {
		nameBytes := make([]byte, 16)
		if _, err := io.ReadFull(f, nameBytes); err != nil {
			return nil, err
		}
		name := strings.Trim(string(nameBytes), " \t\r\n\x00")

		voicePatch := make([]byte, 128)
		if _, err := io.ReadFull(f, voicePatch); err != nil {
			return nil, err
		}

		cand := DecodeVoicePatch(voicePatch)

		specBuf := make([]byte, TimbreSpecBins*8)
		if _, err := io.ReadFull(f, specBuf); err != nil {
			return nil, err
		}

		var spec [TimbreSpecBins]float64
		for j := 0; j < TimbreSpecBins; j++ {
			spec[j] = math.Float64frombits(binary.LittleEndian.Uint64(specBuf[j*8 : (j+1)*8]))
		}

		var compositePeak PeakShape
		if version >= 2 {
			peakBuf := make([]byte, 32)
			if _, err := io.ReadFull(f, peakBuf); err != nil {
				return nil, err
			}
			compositePeak.AttackTimeMs = math.Float64frombits(binary.LittleEndian.Uint64(peakBuf[0:8]))
			compositePeak.TransientWidth = math.Float64frombits(binary.LittleEndian.Uint64(peakBuf[8:16]))
			compositePeak.DecaySlope = math.Float64frombits(binary.LittleEndian.Uint64(peakBuf[16:24]))
			compositePeak.CrestFactor = math.Float64frombits(binary.LittleEndian.Uint64(peakBuf[24:32]))
		} else {
			// Version 1: compute composite peak on load from envelope bytes
			compositePeak = CompositePeakShape(cand.Algorithm, cand.Ops)
		}

		tc.Entries = append(tc.Entries, TimbreCodebookEntry{
			Name:          name,
			Candidate:     cand,
			TimbreSpec:    spec,
			CompositePeak: compositePeak,
		})
	}

	return tc, nil
}

// IndexSysexBankTimbre parses a 32-voice DX7 .syx file and computes timbre fingerprints in parallel.
func IndexSysexBankTimbre(syxPath string, noteNum uint8, dur time.Duration) ([]TimbreCodebookEntry, error) {
	tempSynth := msfa.NewSynth(float64(RenderSampleRate))
	if err := tempSynth.LoadPatch(syxPath); err != nil {
		tempSynth.Free()
		return nil, err
	}

	bankData := make([]byte, 4096)
	tempSynth.GetBank(bankData)
	names := make([]string, 32)
	for i := 0; i < 32; i++ {
		names[i] = tempSynth.Programs[i]
	}
	tempSynth.Free()

	entries := make([]TimbreCodebookEntry, 32)

	numWorkers := runtime.GOMAXPROCS(0)
	if numWorkers > 32 {
		numWorkers = 32
	}

	taskChan := make(chan int, 32)
	for i := 0; i < 32; i++ {
		taskChan <- i
	}
	close(taskChan)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			workerSynth := msfa.NewSynth(float64(RenderSampleRate))
			defer workerSynth.Free()
			workerSynth.SetBank(bankData)

			for slot := range taskChan {
				name := strings.TrimSpace(names[slot])
				if name == "" {
					name = fmt.Sprintf("Patch_%d", slot+1)
				}

				workerSynth.ChangeProgram(0, uint8(slot))
				workerSynth.WriteMidi([]byte{0x90, noteNum, 100})

				sustainSamples := int(float64(RenderSampleRate) * dur.Seconds())
				pcm := make([]int16, sustainSamples)
				workerSynth.GetSamples(pcm)

				workerSynth.WriteMidi([]byte{0x80, noteNum, 0})
				drainSamples := RenderSampleRate / 4
				drain := make([]int16, drainSamples)
				workerSynth.GetSamples(drain)

				totalSamples := sustainSamples + drainSamples
				samples := make([]float64, totalSamples)
				for i := 0; i < sustainSamples; i++ {
					samples[i] = float64(pcm[i]) / 32768.0
				}
				for i := 0; i < drainSamples; i++ {
					samples[sustainSamples+i] = float64(drain[i]) / 32768.0
				}

				timbreSpec := ExtractTimbreWindow(samples, RenderSampleRate)
				rawPatch := bankData[slot*128 : (slot+1)*128]
				cand := DecodeVoicePatch(rawPatch)
				compositePeak := CompositePeakShape(cand.Algorithm, cand.Ops)

				entries[slot] = TimbreCodebookEntry{
					Name:          name,
					Candidate:     cand,
					TimbreSpec:    timbreSpec,
					CompositePeak: compositePeak,
				}
			}
		}()
	}
	wg.Wait()

	return entries, nil
}
