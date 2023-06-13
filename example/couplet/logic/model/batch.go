package model

type pair struct {
	x []int
	y int
}

type batch struct {
	data []pair
}

func (b *batch) append(p pair) {
	b.data = append(b.data, p)
}

func (b *batch) size() int {
	return len(b.data)
}
