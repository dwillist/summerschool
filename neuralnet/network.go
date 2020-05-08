package neuralnet

// TODO: use sparse matrix implementaion (way faster....)
import (
	"fmt"
	"math/rand"

	"github.com/dwillist/summerschool/v2/neuralnet/nodefuncs"
	"gonum.org/v1/gonum/mat"
)

type NodeFunc interface {
	CalcVal(float64, mat.Vector) float64
	CalcDiff(float64, mat.Vector) float64
}

type Config struct {
	LayerConfigs []LayerConfig
	WeightInit   func() float64
}

type LayerConfig struct {
	Size int
	Func NodeFunc
}

type Network struct {
	InputSize    int
	OutputSize   int
	LayerConfigs []LayerConfig
	Weights      []*mat.Dense
	Bias         []*mat.VecDense
	Activation   []*mat.VecDense
	Zval         []*mat.VecDense
}

func InitOne() float64 {
	return 1.0
}

func InitRandom() float64 {
	return rand.Float64()
}

func NewNetwork(config Config) (Network, error) {
	result := Network{}
	result.LayerConfigs = config.LayerConfigs
	// error cases
	switch {
	case result.Len() == 0:
		return result, fmt.Errorf("layerConfig must contain at least 1 element")
	case result.LayerConfigs[0].Func != nil:
		panic("balls")
	}

	result.InputSize = result.LayerConfigs[0].Size

	// set up Bias and Weight values
	prevSize := 0

	for _, lconfig := range result.LayerConfigs {
		switch {
		case lconfig.Size <= 0:
			return Network{}, fmt.Errorf("invalid layer size: %v", lconfig.Size)
		case prevSize != 0:
			initVals := make([]float64, lconfig.Size*prevSize)
			for i := 0; i < lconfig.Size*prevSize; i++ {
				initVals[i] = config.WeightInit()
			}

			result.Weights = append(result.Weights, mat.NewDense(lconfig.Size, prevSize, initVals))

			fallthrough
		default:
			result.Bias = append(result.Bias, mat.NewVecDense(lconfig.Size, nil))
			result.Activation = append(result.Activation, mat.NewVecDense(lconfig.Size, nil))
			result.Zval = append(result.Zval, mat.NewVecDense(lconfig.Size, nil))
			prevSize = lconfig.Size
		}
	}

	result.OutputSize = prevSize

	return result, nil
}

func (n *Network) Len() int {
	return len(n.LayerConfigs)
}

func (n *Network) Reset() {
	n.Activation = nil
	n.Zval = nil
}

// TODO: switch implementation to use sparce matracies datastructures.

func (n *Network) Calculate(input *mat.VecDense) (*mat.VecDense, error) {
	n.Reset()

	r := input.Len()

	if r != n.InputSize {
		return nil, fmt.Errorf("invalid input size: %v", r)
	}
	// set up input Z-value
	inputZval := mat.VecDenseCopyOf(input)
	n.Zval = append(n.Zval, inputZval)

	// set up input activation
	prevActivation := mat.VecDenseCopyOf(input)
	n.Activation = append(n.Activation, prevActivation)

	configIdx := 1
	weightsIdx := 0

	for configIdx < n.Len() {
		// mult prevOutput by weights
		newZ := mat.NewVecDense(n.LayerConfigs[configIdx].Size, nil)

		newZ.MulVec(n.Weights[weightsIdx], prevActivation)
		// add bias
		newZ.AddVec(newZ, n.Bias[configIdx])

		n.Zval = append(n.Zval, newZ)
		// apply function
		newActivation := mat.VecDenseCopyOf(newZ)
		nodefuncs.ApplyFunc(newActivation, n.LayerConfigs[configIdx].Func.CalcVal)

		n.Activation = append(n.Activation, newActivation)
		prevActivation = newActivation

		// increment indicies
		configIdx++
		weightsIdx++
	}

	return mat.VecDenseCopyOf(prevActivation), nil
}

func (n *Network) generateInitialDelta(solution *mat.VecDense) (*mat.VecDense, error) {
	if solution.Len() != n.OutputSize {
		return nil, fmt.Errorf("invalid solution dimension: %d, expected %d", solution.Len(), n.OutputSize)
	}

	layerSub := mat.NewVecDense(n.OutputSize, nil)
	layerCount := n.Len()
	layerSub.SubVec(n.Activation[layerCount-1], solution) //(a - y)

	// diffVector := mat.VecDenseCopyOf(n.Zval[layerCount-1])
	// nodefuncs.ApplyFunc(diffVector, n.LayerConfigs[layerCount-1].Func.CalcDiff)

	// result := mat.NewVecDense(n.OutputSize, nil)
	// result.MulElemVec(diffVector, layerSub)

	// return result, nil
	return layerSub, nil
}

func (n *Network) GenerateDelta(solution *mat.VecDense) ([]*mat.VecDense, error) {
	var result []*mat.VecDense

	initial, err := n.generateInitialDelta(solution)

	if err != nil {
		panic(err)
	}

	result = append(result, initial)
	prevDiff := initial
	// iterate backwards through layers
	for layerIndex := n.Len() - 2; layerIndex > 0; layerIndex-- {
		weightMat := n.Weights[layerIndex].T()

		r, c := weightMat.Dims()

		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
			}
		}

		mulResult := mat.NewVecDense(n.LayerConfigs[layerIndex].Size, nil)

		mulResult.MulVec(weightMat, prevDiff)

		// diffVector := mat.VecDenseCopyOf(n.Zval[layerIndex])
		// nodefuncs.ApplyFunc(diffVector, n.LayerConfigs[layerIndex].Func.CalcDiff)
		// newResult := mat.NewVecDense(n.LayerConfigs[layerIndex].Size, nil)
		//
		// newResult.MulElemVec(mulResult, diffVector)
		//
		// result = append(result, newResult)
		// prevDiff = newResult
		result = append(result, mulResult)
		prevDiff = mulResult
	}

	i := 0
	j := len(result) - 1

	for i < j {
		result[i], result[j] = result[j], result[i]
		i++
		j--
	}

	return result, nil
}

func (n *Network) Update(delta []*mat.VecDense) error {
	if err := n.updateBias(delta); err != nil {
		return err
	}

	return n.updateWeights(delta)
}

func (n *Network) updateBias(delta []*mat.VecDense) error {
	var deltaIndex int
	learningRate := float64(.01)

	for biasIndex := 1; biasIndex < len(n.Bias); biasIndex++ {
		deltaIndex = biasIndex - 1
		scaledDelta := mat.VecDenseCopyOf(delta[deltaIndex])
		nodefuncs.ApplyFunc(scaledDelta, func(x float64, _ mat.Vector) float64 { return x * learningRate })

		n.Bias[biasIndex].SubVec(n.Bias[biasIndex], scaledDelta)
	}

	return nil
}

func (n *Network) updateWeights(delta []*mat.VecDense) error {
	learningRate := float64(.01)

	for weightIndex := 0; weightIndex < len(n.Weights); weightIndex++ {
		prevActivation := n.Activation[weightIndex]
		curDelta := mat.VecDenseCopyOf(delta[weightIndex])
		nodefuncs.ApplyFunc(curDelta, func(x float64, _ mat.Vector) float64 { return x * learningRate })

		weightDelta := mat.NewDense(curDelta.Len(), prevActivation.Len(), nil)
		weightDelta.Mul(curDelta, prevActivation.TVec())
		n.Weights[weightIndex].Sub(n.Weights[weightIndex], weightDelta)
	}

	return nil
}
