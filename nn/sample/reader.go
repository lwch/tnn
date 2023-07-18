package sample

import (
	"encoding/binary"
	"io"
	"reflect"
	"sync"
)

// Reader sample reader
type Reader struct {
	hdr sampleHeader
	r   io.ReadSeeker
	m   sync.Mutex
}

// NewReader create sample reader
func NewReader(r io.ReadSeeker) (*Reader, error) {
	var ret Reader
	ret.r = r
	err := binary.Read(r, binary.BigEndian, &ret.hdr)
	if err != nil {
		return nil, err
	}
	return &ret, nil
}

// BatchSize get batch size
func (r *Reader) BatchSize() uint32 {
	return r.hdr.BatchSize
}

// FeatureSize get feature size
func (r *Reader) FeatureSize() uint32 {
	return r.hdr.FeatureSize
}

// LabelSize get label size
func (r *Reader) LabelSize() uint32 {
	return r.hdr.LabelSize
}

// ReadSample read sample data
func (r *Reader) ReadSample(idx uint32, features, labels []float64) error {
	r.m.Lock()
	defer r.m.Unlock()
	sampleSize := (r.hdr.FeatureSize + r.hdr.LabelSize) * 8
	_, err := r.r.Seek(int64(reflect.TypeOf(r.hdr).Size())+int64(idx)*int64(sampleSize), io.SeekStart)
	if err != nil {
		return err
	}
	err = binary.Read(r.r, binary.BigEndian, features[:r.hdr.FeatureSize])
	if err != nil {
		return err
	}
	return binary.Read(r.r, binary.BigEndian, labels[:r.hdr.LabelSize])
}
