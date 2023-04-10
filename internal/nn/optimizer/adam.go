package optimizer

// type Adam struct {
// 	weightDecay
// 	lr           float64
// 	beta1, beta2 float64
// 	epsilon      float64

// 	t    int
// 	m, v mat.Dense
// }

// func NewAdam(lr, wd, beta1, beta2, epsilon float64) *Adam {
// 	return &Adam{
// 		weightDecay: weightDecay(wd),
// 		lr:          lr,
// 		beta1:       beta1,
// 		beta2:       beta2,
// 		epsilon:     epsilon,
// 	}
// }

// func (adam *Adam) Update(weights, delta *mat.Dense) {
// 	adam.t++
// }

// func (adam *Adam) compute(i, j int, delta float64) float64 {
// 	fmt.Println(i, j)
// 	m := adam.m.At(i, j)
// 	v := adam.v.At(i, j)
// 	m += (1 - adam.beta1) * (delta - m)
// 	v += (1 - adam.beta2) * (math.Pow(delta, 2) - v)
// 	adam.m.Set(i, j, m)
// 	adam.v.Set(i, j, v)
// 	m /= 1 - math.Pow(adam.beta1, float64(adam.t))
// 	v /= 1 - math.Pow(adam.beta2, float64(adam.t))
// 	return -adam.lr * m / (math.Pow(v, 0.5) + adam.epsilon)
// }
