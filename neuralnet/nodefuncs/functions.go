package nodefuncs

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

///
/// Softmax
///
type Softmax struct{}

func (s Softmax) CalcVal(x float64, v mat.Vector) float64 {
	num := math.Exp(x)
	denom := float64(0)

	for idx := 0; idx < v.Len(); idx++ {
		denom += math.Exp(v.AtVec(idx))
	}

	return num / denom
}

func (s Softmax) CalcDiff(x float64, v mat.Vector) float64 {
	sMax := s.CalcVal(x, v)
	return sMax * (1 - sMax)
}

///
/// Relu Def
///
type Relu struct{}

func (r Relu) CalcVal(x float64, _ mat.Vector) float64 {
	return math.Max(0, x)
}

func (r Relu) CalcDiff(x float64, _ mat.Vector) float64 {
	if x > 0 {
		return 1
	}

	return float64(0)
}

///
/// Sigmoid Def
///
type Sigmoid struct{}

func (s Sigmoid) CalcVal(x float64, _ mat.Vector) float64 {
	return float64(1.0) / (float64(1.0) + math.Exp(-x))
}

func (s Sigmoid) CalcDiff(x float64, _ mat.Vector) float64 {
	sig := s.CalcVal(x, nil)
	return sig * (float64(1) - sig)
}

///
/// Identity Def
///
type Identity struct{}

func (s Identity) CalcVal(x float64, _ mat.Vector) float64 {
	return x
}

func (s Identity) CalcDiff(x float64, _ mat.Vector) float64 {
	return float64(0)
}

func ApplyFunc(vec *mat.VecDense, nodefunc func(float64, mat.Vector) float64) {
	r := vec.Len()
	for rowIdx := 0; rowIdx < r; rowIdx++ {
		vec.SetVec(rowIdx, nodefunc(vec.AtVec(rowIdx), vec))
	}
}
