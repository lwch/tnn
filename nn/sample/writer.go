package sample

import (
	"encoding/binary"
	"io"
	"sync"
)

// Writer sample writer
type Writer struct {
	hdr         sampleHeader
	w           io.WriteSeeker
	writeHeader bool
	m           sync.Mutex
}

// NewWriter create sample writer
func NewWriter(w io.WriteSeeker) *Writer {
	var ret Writer
	ret.w = w
	return &ret
}

// Close rewrite header
func (w *Writer) Close() error {
	w.m.Lock()
	defer w.m.Unlock()
	_, err := w.w.Seek(0, io.SeekStart)
	if err != nil {
		return err
	}
	return binary.Write(w.w, binary.BigEndian, w.hdr)
}

// WriteSample write sample data
func (w *Writer) WriteSample(features, labels []float32) error {
	w.m.Lock()
	defer w.m.Unlock()
	_, err := w.w.Seek(0, io.SeekEnd)
	if err != nil {
		return err
	}
	w.hdr.BatchSize++
	w.hdr.FeatureSize = uint32(len(features))
	w.hdr.LabelSize = uint32(len(labels))
	if !w.writeHeader {
		err = binary.Write(w.w, binary.BigEndian, w.hdr)
		if err != nil {
			return err
		}
		w.writeHeader = true
	}
	err = binary.Write(w.w, binary.BigEndian, features)
	if err != nil {
		return err
	}
	return binary.Write(w.w, binary.BigEndian, labels)
}
