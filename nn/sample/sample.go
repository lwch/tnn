package sample

import (
	"encoding/binary"
	"io"

	"github.com/lwch/runtime"
)

type sampleHeader struct {
	BatchSize   uint32
	FeatureSize uint32
	LabelSize   uint32
}

// Write write samples to w
func Write(w io.Writer, inputs, outputs [][]float64) {
	var hdr sampleHeader
	hdr.BatchSize = uint32(len(inputs))
	hdr.FeatureSize = uint32(len(inputs[0]))
	hdr.LabelSize = uint32(len(outputs[0]))
	runtime.Assert(binary.Write(w, binary.BigEndian, &hdr))
	for _, v := range inputs {
		runtime.Assert(binary.Write(w, binary.BigEndian, v))
	}
	for _, v := range outputs {
		runtime.Assert(binary.Write(w, binary.BigEndian, v))
	}
}

// Read read samples from reader
func Read(r io.Reader) ([][]float64, [][]float64) {
	var hdr sampleHeader
	runtime.Assert(binary.Read(r, binary.BigEndian, &hdr))
	inputs := make([][]float64, hdr.BatchSize)
	outputs := make([][]float64, hdr.BatchSize)
	for i := range inputs {
		inputs[i] = make([]float64, hdr.FeatureSize)
		runtime.Assert(binary.Read(r, binary.BigEndian, inputs[i]))
	}
	for i := range outputs {
		outputs[i] = make([]float64, hdr.LabelSize)
		runtime.Assert(binary.Read(r, binary.BigEndian, outputs[i]))
	}
	return inputs, outputs
}
