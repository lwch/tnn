package sample

import (
	"io"
	"testing"
)

type buffer struct {
	data   []byte
	offset int
}

func (b *buffer) Write(p []byte) (n int, err error) {
	if b.offset < len(b.data) {
		copy(b.data[b.offset:], p)
	} else {
		b.data = append(b.data, p...)
	}
	b.offset += len(p)
	return len(p), nil
}

func (b *buffer) Read(p []byte) (n int, err error) {
	n = copy(p, b.data[b.offset:])
	b.offset += n
	return n, nil
}

func (b *buffer) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case io.SeekStart:
		b.offset = int(offset)
	case io.SeekCurrent:
		b.offset += int(offset)
	case io.SeekEnd:
		b.offset = len(b.data) + int(offset)
	}
	return int64(b.offset), nil
}

func (b *buffer) Bytes() []byte {
	return b.data
}

func TestSample(t *testing.T) {
	var buf buffer
	w := NewWriter(&buf)
	defer w.Close()
	for i := 0; i < 10; i++ {
		err := w.WriteSample([]float64{float64(i), float64(i + 1)}, []float64{float64(i + 2)})
		if err != nil {
			t.Fatal(err)
		}
	}
	w.Close()
	_, err := buf.Seek(0, io.SeekStart)
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewReader(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if r.BatchSize() != 10 {
		t.Fatal("invalid batch size")
	}
	if r.FeatureSize() != 2 {
		t.Fatal("invalid feature size")
	}
	if r.LabelSize() != 1 {
		t.Fatal("invalid label size")
	}
	for i := 0; i < 10; i++ {
		features := make([]float64, 2)
		labels := make([]float64, 1)
		err := r.ReadSample(uint32(i), features, labels)
		if err != nil {
			t.Fatal(err)
		}
		if features[0] != float64(i) {
			t.Fatal("invalid feature")
		}
		if features[1] != float64(i+1) {
			t.Fatal("invalid feature")
		}
		if labels[0] != float64(i+2) {
			t.Fatal("invalid label")
		}
	}
}
