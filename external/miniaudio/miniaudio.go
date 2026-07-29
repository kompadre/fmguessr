package miniaudio

/*
#cgo !windows LDFLAGS: -lm -lpthread -ldl
#include "miniaudio_backend.h"
*/
import "C"
import (
	"errors"
	"unsafe"
)

// Init initializes miniaudio device with 2 ringbuffers (bufferFrames per channel).
func Init(sampleRate uint32, bufferFrames uint32) error {
	res := C.audio_init(C.uint32_t(sampleRate), C.uint32_t(bufferFrames))
	if res != 0 {
		return errors.New("failed to initialize miniaudio device")
	}
	return nil
}

// WriteChannels pushes independent Left and Right float32 channel samples into the ringbuffers.
// Returns the number of frames written.
func WriteChannels(left []float32, right []float32) int {
	if len(left) == 0 || len(left) != len(right) {
		return 0
	}
	pLeft := (*C.float)(unsafe.Pointer(&left[0]))
	pRight := (*C.float)(unsafe.Pointer(&right[0]))
	frames := C.uint32_t(len(left))

	return int(C.audio_write_channels(pLeft, pRight, frames))
}

// AvailableWriteSpace returns the number of frames available for writing in the ringbuffers.
func AvailableWriteSpace() int {
	return int(C.audio_available_write_space())
}

// Close stops miniaudio device and releases ringbuffer resources.
func Close() {
	C.audio_close()
}
