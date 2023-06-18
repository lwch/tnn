package sample

import (
	"encoding/binary"
	"io"
	"sync"
)

// Reader sample reader
type Reader struct {
	hdr           sampleHeader
	r             io.ReadSeeker
	featurePrefix int64
	labelPrefix   int64
	m             sync.Mutex
}

// NewReader create sample reader
func NewReader(r io.ReadSeeker) (*Reader, error) {
	var ret Reader
	ret.r = r
	err := binary.Read(r, binary.BigEndian, &ret.hdr)
	if err != nil {
		return nil, err
	}
	ret.featurePrefix, err = r.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}
	ret.labelPrefix = ret.featurePrefix + int64(ret.hdr.BatchSize*ret.hdr.FeatureSize*4)
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

// ReadFeature read feature data
func (r *Reader) ReadFeature(idx uint32, data []float32) error {
	r.m.Lock()
	defer r.m.Unlock()
	_, err := r.r.Seek(r.featurePrefix+int64(idx)*int64(r.hdr.FeatureSize)*4, io.SeekStart)
	if err != nil {
		return err
	}
	return binary.Read(r.r, binary.BigEndian, data)
}

// ReadLabel read label data
func (r *Reader) ReadLabel(idx uint32, data []float32) error {
	r.m.Lock()
	defer r.m.Unlock()
	_, err := r.r.Seek(r.labelPrefix+int64(idx)*int64(r.hdr.LabelSize)*4, io.SeekStart)
	if err != nil {
		return err
	}
	return binary.Read(r.r, binary.BigEndian, data)
}

// WriteTo write to
func (r *Reader) WriteTo(w io.Writer) (int64, error) {
	r.m.Lock()
	defer r.m.Unlock()
	_, err := r.r.Seek(0, io.SeekStart)
	if err != nil {
		return 0, err
	}
	return io.Copy(w, r.r)
}
