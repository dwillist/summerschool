package fakes

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

type Network struct {
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
	GenerateDeltaCall struct {
		sync.Mutex
		CallCount int
		Receives  struct {
			VecDense *mat.VecDense
		}
		Returns struct {
			VecDenseSlice []*mat.VecDense
			Error         error
		}
		Stub func(*mat.VecDense) ([]*mat.VecDense, error)
	}
	UpdateCall struct {
		sync.Mutex
		CallCount int
		Receives  struct {
			VecDenseSlice []*mat.VecDense
		}
		Returns struct {
			Error error
		}
		Stub func([]*mat.VecDense) error
	}
}

func (f *Network) Calculate(param1 *mat.VecDense) (*mat.VecDense, error) {
	f.CalculateCall.Lock()
	defer f.CalculateCall.Unlock()
	f.CalculateCall.CallCount++
	f.CalculateCall.Receives.VecDense = param1
	if f.CalculateCall.Stub != nil {
		return f.CalculateCall.Stub(param1)
	}
	return f.CalculateCall.Returns.VecDense, f.CalculateCall.Returns.Error
}
func (f *Network) GenerateDelta(param1 *mat.VecDense) ([]*mat.VecDense, error) {
	f.GenerateDeltaCall.Lock()
	defer f.GenerateDeltaCall.Unlock()
	f.GenerateDeltaCall.CallCount++
	f.GenerateDeltaCall.Receives.VecDense = param1
	if f.GenerateDeltaCall.Stub != nil {
		return f.GenerateDeltaCall.Stub(param1)
	}
	return f.GenerateDeltaCall.Returns.VecDenseSlice, f.GenerateDeltaCall.Returns.Error
}
func (f *Network) Update(param1 []*mat.VecDense) error {
	f.UpdateCall.Lock()
	defer f.UpdateCall.Unlock()
	f.UpdateCall.CallCount++
	f.UpdateCall.Receives.VecDenseSlice = param1
	if f.UpdateCall.Stub != nil {
		return f.UpdateCall.Stub(param1)
	}
	return f.UpdateCall.Returns.Error
}
