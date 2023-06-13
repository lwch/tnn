package model

type batch struct {
	data []int
}

func (b *batch) append(i int) {
	b.data = append(b.data, i)
}
