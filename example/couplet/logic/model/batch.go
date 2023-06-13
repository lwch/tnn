package model

import "github.com/lwch/tnn/example/couplet/logic/sample"

type batch struct {
	data []*sample.Sample
}

func (b *batch) append(p *sample.Sample) {
	b.data = append(b.data, p)
}

func (b *batch) size() int {
	return len(b.data)
}
