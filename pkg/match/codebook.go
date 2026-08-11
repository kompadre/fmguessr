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
	"unsafe"

	"github.com/kompadre/fmguessr/external/msfa"
)

// CodebookEntry represents a pre-computed patch thumbnail in the index.
type CodebookEntry struct {
	Name        string
	Candidate   *Candidate
	Fingerprint Fingerprint
}

// Codebook holds a collection of pre-indexed DX7 patch thumbnails.
type Codebook struct {
	Entries []CodebookEntry
}

// AddEntriesDeduplicated appends non-duplicate entries to the codebook.
// Returns the number of newly added entries.
func (cb *Codebook) AddEntriesDeduplicated(newEntries []CodebookEntry) int {
	existing := make(map[string]bool, len(cb.Entries))
	for _, entry := range cb.Entries {
		sig := patchSignature(entry)
		existing[sig] = true
	}

	added := 0
	for _, entry := range newEntries {
		sig := patchSignature(entry)
		if !existing[sig] {
			existing[sig] = true
			cb.Entries = append(cb.Entries, entry)
			added++
		}
	}

	return added
}

func patchSignature(entry CodebookEntry) string {
	return string(entry.Candidate.Encode())
}
func (cb *Codebook) SearchKNN(targetFp Fingerprint, k int) []*Candidate {
	if len(cb.Entries) == 0 {
		return nil
	}

	type matchResult struct {
		cand *Candidate
		dist float64
	}

	results := make([]matchResult, len(cb.Entries))
	for i, entry := range cb.Entries {
		results[i] = matchResult{
			cand: entry.Candidate,
			dist: targetFp.CosineDistance(entry.Fingerprint),
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

// SearchKNNWithDistances finds the k closest entries by cosine distance and returns both candidates and distances.
func (cb *Codebook) SearchKNNWithDistances(targetFp Fingerprint, k int) ([]*Candidate, []float64) {
	if len(cb.Entries) == 0 {
		return nil, nil
	}

	type matchResult struct {
		cand *Candidate
		dist float64
	}

	results := make([]matchResult, len(cb.Entries))
	for i, entry := range cb.Entries {
		results[i] = matchResult{
			cand: entry.Candidate,
			dist: targetFp.CosineDistance(entry.Fingerprint),
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

// Save writes the codebook to a binary file.
func (cb *Codebook) Save(filePath string) error {
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	// Header: Magic "DXCB" + Version 1 + uint32 EntryCount
	header := make([]byte, 12)
	copy(header[0:4], []byte("DXCB"))
	binary.LittleEndian.PutUint32(header[4:8], 1)
	binary.LittleEndian.PutUint32(header[8:12], uint32(len(cb.Entries)))
	if _, err := f.Write(header); err != nil {
		return err
	}

	for _, entry := range cb.Entries {
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

		// Fingerprint (32 x float64 = 256 bytes)
		fpBuf := make([]byte, FingerprintDim*8)
		for i, val := range entry.Fingerprint {
			binary.LittleEndian.PutUint64(fpBuf[i*8:(i+1)*8], mathFloat64bits(val))
		}
		if _, err := f.Write(fpBuf); err != nil {
			return err
		}
	}

	return nil
}

// LoadCodebookAny loads either a standard fingerprint codebook ("DXCB") or a timbre codebook ("DXCT").
func LoadCodebookAny(filePath string) (*Codebook, error) {
	cb, err := LoadCodebook(filePath)
	if err == nil {
		return cb, nil
	}

	tcb, errT := LoadTimbreCodebook(filePath)
	if errT == nil {
		cbAny := &Codebook{
			Entries: make([]CodebookEntry, len(tcb.Entries)),
		}
		for i, entry := range tcb.Entries {
			cbAny.Entries[i] = CodebookEntry{
				Name:      entry.Name,
				Candidate: entry.Candidate.Copy(),
			}
		}
		return cbAny, nil
	}

	return nil, fmt.Errorf("failed to load codebook (%v, %v)", err, errT)
}

// LoadCodebook reads a codebook from a binary file.
func LoadCodebook(filePath string) (*Codebook, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	header := make([]byte, 12)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, err
	}

	if string(header[0:4]) != "DXCB" {
		return nil, fmt.Errorf("invalid codebook magic: %s", header[0:4])
	}

	count := binary.LittleEndian.Uint32(header[8:12])
	cb := &Codebook{Entries: make([]CodebookEntry, 0, count)}

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

		fpBuf := make([]byte, FingerprintDim*8)
		if _, err := io.ReadFull(f, fpBuf); err != nil {
			return nil, err
		}

		var fp Fingerprint
		for j := 0; j < FingerprintDim; j++ {
			bits := binary.LittleEndian.Uint64(fpBuf[j*8 : (j+1)*8])
			fp[j] = mathFloat64frombits(bits)
		}

		cb.Entries = append(cb.Entries, CodebookEntry{
			Name:        name,
			Candidate:   cand,
			Fingerprint: fp,
		})
	}

	return cb, nil
}

// IndexSysexBank parses a 32-voice DX7 .syx file and computes fingerprints for each patch in parallel.
func IndexSysexBank(syxPath string, noteNum uint8, dur time.Duration) ([]CodebookEntry, error) {
	// First, load the patch once to get the names and bank data
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

	entries := make([]CodebookEntry, 32)
	
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
			
			// Each worker needs its own synth instance
			workerSynth := msfa.NewSynth(float64(RenderSampleRate))
			defer workerSynth.Free()
			workerSynth.SetBank(bankData)

			for slot := range taskChan {
				name := strings.TrimSpace(names[slot])
				if name == "" {
					name = fmt.Sprintf("Patch_%d", slot+1)
				}

				// Render the patch audio
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

				fp := ComputeFingerprint(samples, RenderSampleRate)

				// Decode slot patch bytes into Candidate struct
				rawPatch := bankData[slot*128 : (slot+1)*128]
				cand := DecodeVoicePatch(rawPatch)

				entries[slot] = CodebookEntry{
					Name:        name,
					Candidate:   cand,
					Fingerprint: fp,
				}
			}
		}()
	}
	wg.Wait()

	return entries, nil
}

// DecodeVoicePatch unpacks a 128-byte DX7 raw voice patch into a Candidate struct.
func DecodeVoicePatch(patch []byte) *Candidate {
	c := &Candidate{
		Algorithm: patch[110] & 0x1F,
		Feedback:  patch[111] & 0x07,
		Fitness:   math.MaxFloat64,
		Scores:    nanScores(),
	}

	if len(patch) >= 128 {
		copy(c.RawBase[:], patch[:128])
		// Extract voice name from bytes 118-127
		name := string(patch[118:128])
		// Trim trailing spaces
		for len(name) > 0 && name[len(name)-1] == ' ' {
			name = name[:len(name)-1]
		}
		if len(name) > 0 {
			c.Name = name
		}
	}

	for op := 0; op < 6; op++ {
		base := op * 17
		c.Ops[op] = OpParams{
			Rate1:       patch[base+0],
			Rate2:       patch[base+1],
			Rate3:       patch[base+2],
			Rate4:       patch[base+3],
			Level1:      patch[base+4],
			Level2:      patch[base+5],
			Level3:      patch[base+6],
			Level4:      patch[base+7],
			OutputLevel: patch[base+14],
			Mode:        patch[base+15] & 1,
			FreqCoarse:  (patch[base+15] >> 1) & 0x1F,
			FreqFine:    patch[base+16],
		}
	}

	for i := 0; i < NumCandidateParams; i++ {
		c.Direction[i] = 1
		c.Velocity[i] = 1.0
	}

	return c
}

func mathFloat64bits(f float64) uint64 {
	return *(*uint64)(unsafe.Pointer(&f))
}

func mathFloat64frombits(b uint64) float64 {
	return *(*float64)(unsafe.Pointer(&b))
}

// IndexCandidateToCodebook renders candidate audio using gateDur key hold duration,
// extracts fingerprint & timbre window, and appends candidate to codebook files.
func IndexCandidateToCodebook(c *Candidate, name string, noteNum uint8, gateDur, duration time.Duration, codebookPath string) error {
	if codebookPath == "" {
		codebookPath = "dx7_codebook.bin"
	}
	if name == "" {
		name = "MATCH_GEN"
	}
	if gateDur <= 0 {
		gateDur = duration
	}

	// 1. Render candidate audio with key hold gate duration
	candAudio := RenderCandidateWithGate(c, noteNum, gateDur, duration)

	// 2. Index into standard fingerprint codebook
	fp := ComputeFingerprint(candAudio, RenderSampleRate)
	cbEntry := CodebookEntry{
		Name:        name,
		Candidate:   c.Copy(),
		Fingerprint: fp,
	}

	cb, err := LoadCodebook(codebookPath)
	if err != nil || cb == nil {
		cb = &Codebook{}
	}
	cb.AddEntriesDeduplicated([]CodebookEntry{cbEntry})
	if err := cb.Save(codebookPath); err != nil {
		return fmt.Errorf("save codebook: %w", err)
	}

	// 3. Index into timbre codebook
	timbrePath := codebookPath
	if len(codebookPath) > 4 && codebookPath[len(codebookPath)-4:] == ".bin" {
		timbrePath = codebookPath[:len(codebookPath)-4] + "_timbre.bin"
	} else {
		timbrePath = codebookPath + "_timbre.bin"
	}

	timbreSpec := ExtractTimbreWindow(candAudio, RenderSampleRate)
	tEntry := TimbreCodebookEntry{
		Name:       name,
		Candidate:  c.Copy(),
		TimbreSpec: timbreSpec,
	}

	tcb, err := LoadTimbreCodebook(timbrePath)
	if err != nil || tcb == nil {
		tcb = &TimbreCodebook{}
	}
	tcb.AddEntriesDeduplicated([]TimbreCodebookEntry{tEntry})
	if err := tcb.Save(timbrePath); err != nil {
		return fmt.Errorf("save timbre codebook: %w", err)
	}

	return nil
}
