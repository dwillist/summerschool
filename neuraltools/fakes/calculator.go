package fakes

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

type Calculator struct {
	CalculateCall struct {
		sync.Mutex
		CallCount int
		Receives  struct {
			VecDense *mat.VecDense
		}
		Returns struct {
			VecDense *mat.VecDense
			Error    error
		}
		Stub func(*mat.VecDense) (*mat.VecDense, error)
	}
}

func (f *Calculator) Calculate(param1 *mat.VecDense) (*mat.VecDense, error) {
	f.CalculateCall.Lock()
	defer f.CalculateCall.Unlock()
	f.CalculateCall.CallCount++
	f.CalculateCall.Receives.VecDense = param1
	if f.CalculateCall.Stub != nil {
		return f.CalculateCall.Stub(param1)
	}
	return f.CalculateCall.Returns.VecDense, f.CalculateCall.Returns.Error
}
