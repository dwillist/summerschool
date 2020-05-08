package neuraltools

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type DataPair struct {
	Input    *mat.VecDense
	Solution *mat.VecDense
}

func NewDataPair(inputs, solutions []*mat.VecDense) ([]DataPair, error) {
	var result []DataPair

	if len(inputs) != len(solutions) {
		return result, fmt.Errorf("input and solution of unequal cardenality %v, %v", len(inputs), len(solutions))
	}

	for idx := 0; idx < len(inputs); idx++ {
		curInput := inputs[idx]
		curSolution := solutions[idx]

		result = append(result, DataPair{
			Input:    curInput,
			Solution: curSolution,
		})
	}

	return result, nil
}

//go:generate faux --interface Network --output fakes/network.go
type Network interface {
	Calculator
	GenerateDelta(*mat.VecDense) ([]*mat.VecDense, error)
	Update([]*mat.VecDense) error
}

//go:generate faux --interface Calculator --output fakes/calculator.go
type Calculator interface {
	Calculate(*mat.VecDense) (*mat.VecDense, error)
}

// mutates the network
func Train(network Network, batchSize int, data ...DataPair) error {
	if batchSize != 1 {
		return fmt.Errorf("unimplemented batch size != 1, %v received", batchSize)
	}

	for idx, datum := range data {
		_, err := network.Calculate(datum.Input)
		if err != nil {
			return fmt.Errorf("network calculation failed on input at index: %v", idx)
		}

		delta, err := network.GenerateDelta(datum.Solution)
		if err != nil {
			return fmt.Errorf("network delta generation failed on solution at index: %v", idx)
		}

		err = network.Update(delta)
		if err != nil {
			return fmt.Errorf("network update failed on delta at index: %v", idx)
		}
	}

	return nil
}

func Test(network Calculator, judge func(*mat.VecDense, *mat.VecDense) bool, data ...DataPair) (correct int, err error) {
	result := 0

	for idx, datum := range data {
		actual, err := network.Calculate(datum.Input)

		if err != nil {
			return 0, fmt.Errorf("error on input %d calculation: %s", idx, err)
		} else if judge(actual, datum.Solution) {
			result++
		}
	}

	return result, nil
}

// assumes len(actual) == len(expected)
func MaxJudge(actual, expected *mat.VecDense) bool {
	if actual.Len() != expected.Len() {
		panic("MaxJudge requires identical length vectors")
	} else if actual.Len() == 0 {
		panic("MaxJudge requires non-empty vectors")
	}
	actualMaxIndex := 0
	rawActual := actual.RawVector()
	maxActual := rawActual.Data[0]

	expectedMaxIndex := 0
	rawExpected := expected.RawVector()
	maxExpected := rawExpected.Data[0]

	for idx := 0; idx < actual.Len(); idx++ {
		curActual := rawActual.Data[idx]
		if curActual > maxActual {
			maxActual = curActual
			actualMaxIndex = idx
		}

		curExpected := rawExpected.Data[idx]
		if curExpected > maxExpected {
			maxExpected = curExpected
			expectedMaxIndex = idx
		}
	}

	return actualMaxIndex == expectedMaxIndex
}

func TestAndTrain(network Network, epochCount, batchSize int, judge func(*mat.VecDense, *mat.VecDense) bool, data ...DataPair) (correctList []int, err error) {
	var result []int

	for epoch := 0; epoch < epochCount; epoch++ {
		correct, err := Test(network, judge, data...)
		if err != nil {
			return result, err
		}

		result = append(result, correct)

		err = Train(network, batchSize, data...)
		if err != nil {
			return result, err
		}
	}

	return result, nil
}
