package msfa

/*
#cgo LDFLAGS: -lm -lpthread
#include "wrapper.h"
*/
import "C"
import "sync"
import "unsafe"
import "os"

// Synth represents a unified wrapper over the MSFA core and its stream buffer.
type Synth struct {
	ringBuffer      C.DX7RingBuffer
	instance        C.DX7Instance
	PatchPath       string
	Programs        [32]string
	ChannelPrograms [16]uint8
}

var globalOnce sync.Once
var globalSampleRate float64

// InitGlobal initializes the MSFA global lookup tables exactly once.
// Must be called before any NewSynth. Subsequent calls are no-ops.
func InitGlobal(sampleRate float64) {
	globalOnce.Do(func() {
		globalSampleRate = sampleRate
		C.dx7_global_init(C.double(sampleRate))
	})
}

// NewSynth creates a new synthesizer instance.
// Calls InitGlobal to ensure lookup tables are initialized exactly once.
func NewSynth(sampleRate float64) *Synth {
	InitGlobal(sampleRate)

	rb := C.dx7_ringbuffer_create()
	inst := C.dx7_synth_create(rb)

	return &Synth{
		ringBuffer: rb,
		instance:   inst,
	}
}

// Free cleans up the allocated C++ objects allocated on the heap.
func (s *Synth) Free() {
	if s.instance != nil {
		C.dx7_synth_destroy(s.instance)
		s.instance = nil
	}
	if s.ringBuffer != nil {
		C.dx7_ringbuffer_destroy(s.ringBuffer)
		s.ringBuffer = nil
	}
}

// WriteMidi pushes raw MIDI bytes or SysEx data directly into the synth's input stream.
func (s *Synth) WriteMidi(data []byte) {
	if len(data) == 0 {
		return
	}
	C.dx7_ringbuffer_write(s.ringBuffer, (*C.uint8_t)(unsafe.Pointer(&data[0])), C.int(len(data)))
}

// GetSamples renders the next block of 16-bit mono PCM audio samples.
func (s *Synth) GetSamples(out []int16) {
	if len(out) == 0 {
		return
	}
	C.dx7_get_samples(s.instance, C.int(len(out)), (*C.int16_t)(unsafe.Pointer(&out[0])))
}

// GetChannelSamples reads per-channel pre-filter audio (int32) captured during
// the last GetSamples call. Buffer must be at least n_samples int32s.
func (s *Synth) GetChannelSamples(channel int, out []int32) {
	if len(out) == 0 || channel < 0 || channel >= 16 {
		return
	}
	C.dx7_get_channel_samples(s.instance, C.int(channel), C.int(len(out)),
		(*C.int32_t)(unsafe.Pointer(&out[0])))
}

func (s *Synth) LoadPatch(path string) error {
	syx, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if len(syx) >= 4104 {
		// Bulk dump: SetBank loads synchronously, no need for MIDI program changes
		s.SetBank(syx[6:4102])
	} else {
		// Single voice or other? Original logic for fallback
		start := 0x7c
		interval := 0xf0 - 0x70
		length := (0x85 - start) + 1
		prog := 0
		for i := start; i < len(syx); i += interval {
			s.Programs[prog] = extractName(syx[i : i+length])
			prog++
		}
		s.WriteMidi(syx)
	}
	s.PatchPath = path
	return nil
}

func (s *Synth) ChangeProgram(channel uint8, program uint8) {
	s.WriteMidi([]byte{0xC0 + channel, program})
	s.ChannelPrograms[channel] = program
}

func ChangeVolume(channel uint8, volume int32) {
	C.dx7_set_channel_volume(C.int32_t(channel), C.int32_t(volume))
}

func (s *Synth) SetBank(data []byte) {
	if len(data) < 4096 {
		return
	}
	C.dx7_set_bank(s.instance, (*C.uint8_t)(unsafe.Pointer(&data[0])))
	// Update Go-side names
	for i := 0; i < 32; i++ {
		voiceData := data[i*128 : (i+1)*128]
		s.Programs[i] = extractName(voiceData)
	}
}

func (s *Synth) SetVoice(slot int, data []byte) {
	if slot < 0 || slot >= 32 || len(data) < 128 {
		return
	}
	C.dx7_set_voice(s.instance, C.int(slot), (*C.uint8_t)(unsafe.Pointer(&data[0])))
	s.Programs[slot] = extractName(data)
}

func (s *Synth) GetBank(out []byte) {
	if len(out) < 4096 {
		return
	}
	C.dx7_get_bank(s.instance, (*C.uint8_t)(unsafe.Pointer(&out[0])))
}

func extractName(data []byte) string {
	nameBytes := data[118:128]
	var res []byte
	for _, b := range nameBytes {
		if b >= 32 && b <= 126 {
			res = append(res, b)
		} else {
			res = append(res, ' ')
		}
	}
	return string(res)
}
