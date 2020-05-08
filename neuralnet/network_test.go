package neuralnet_test

import (
	"testing"

	//"fmt"

	"github.com/sclevine/spec"
	"github.com/dwillist/summerschool/v2/neuralnet"
	"github.com/dwillist/summerschool/v2/neuralnet/nodefuncs"
	"gonum.org/v1/gonum/mat"

	. "github.com/onsi/gomega"
)

type TestFunc struct{}

func (t TestFunc) CalcVal(_ float64, _ mat.Vector) float64 {
	return float64(1)
}

func (t TestFunc) CalcDiff(_ float64, _ mat.Vector) float64 {
	return float64(1)
}

func testNetwork(t *testing.T, context spec.G, it spec.S) {
	var (
		Expect = NewWithT(t).Expect
	)
	context("NewNetwork", func() {
		it("succeeds", func() {
			network, err := neuralnet.NewNetwork(neuralnet.Config{
				LayerConfigs: []neuralnet.LayerConfig{
					{
						Size: 2,
					},
					{
						Size: 3,
					},
				},
				WeightInit: neuralnet.InitOne,
			})
			Expect(err).NotTo(HaveOccurred())

			Expect(network.InputSize).To(Equal(2))
			Expect(network.OutputSize).To(Equal(3))

			Expect(network.Bias).To(HaveLen(2))
			r, c := network.Bias[0].Dims()
			Expect(r).To(Equal(2))
			Expect(c).To(Equal(1))
			r, c = network.Bias[1].Dims()
			Expect(r).To(Equal(3))
			Expect(c).To(Equal(1))

			Expect(network.Activation).To(HaveLen(2))
			r, c = network.Activation[0].Dims()
			Expect(r).To(Equal(2))
			Expect(c).To(Equal(1))
			r, c = network.Activation[1].Dims()
			Expect(r).To(Equal(3))
			Expect(c).To(Equal(1))

			Expect(network.Zval).To(HaveLen(2))
			r, c = network.Zval[0].Dims()
			Expect(r).To(Equal(2))
			Expect(c).To(Equal(1))
			r, c = network.Zval[1].Dims()
			Expect(r).To(Equal(3))
			Expect(c).To(Equal(1))

			Expect(network.Weights).To(HaveLen(1))
			r, c = network.Weights[0].Dims()
			Expect(r).To(Equal(3))
			Expect(c).To(Equal(2))

		})

		context("failure cases", func() {
			it("when config has no entries", func() {
				_, err := neuralnet.NewNetwork(neuralnet.Config{})
				Expect(err).To(MatchError("layerConfig must contain at least 1 element"))
			})

			it("when a layer has invalid size", func() {
				_, err := neuralnet.NewNetwork(neuralnet.Config{
					LayerConfigs: []neuralnet.LayerConfig{
						{
							Size: 0,
						},
					},
					WeightInit: neuralnet.InitOne,
				})
				Expect(err).To(MatchError("invalid layer size: 0"))
			})
		})
	})

	context("Reset", func() {
		var network neuralnet.Network
		it.Before(func() {
			var err error
			network, err = neuralnet.NewNetwork(neuralnet.Config{
				LayerConfigs: []neuralnet.LayerConfig{
					{
						Size: 2,
					},
					{
						Size: 3,
					},
				},
				WeightInit: neuralnet.InitOne,
			})
			Expect(err).NotTo(HaveOccurred())
			x := mat.NewVecDense(2, []float64{1, 2})
			network.Activation = []*mat.VecDense{x}
		})

		it("resets network state from a previous calculation", func() {
			Expect(network.Activation).To(HaveLen(1))
			network.Reset()
			Expect(network.Activation).To(HaveLen(0))

		})
	})

	context("Calculate", func() {
		var network neuralnet.Network
		it.Before(func() {
			var err error
			network, err = neuralnet.NewNetwork(neuralnet.Config{
				LayerConfigs: []neuralnet.LayerConfig{
					{
						Size: 3,
					},
					{
						Size: 2,
						Func: nodefuncs.Identity{},
					},
					{
						Size: 1,
						Func: nodefuncs.Identity{},
					},
				},
				WeightInit: neuralnet.InitOne,
			})
			Expect(err).NotTo(HaveOccurred())
			// initialize some values here
			//newWeights := mat.NewDense(3,2, nil)
			testApply := func(i, j int, val float64) float64 {
				return float64(i) + float64(j)
			}
			network.Weights[0].Apply(testApply, network.Weights[0])
			network.Weights[1].Apply(testApply, network.Weights[1])

		})
		it("calculates outputs for network", func() {
			input := mat.NewVecDense(3, []float64{1, 2, 3})
			output, err := network.Calculate(input)
			Expect(err).NotTo(HaveOccurred())

			r := output.Len()
			Expect(r).To(Equal(1))

			// Input layer Activations
			Expect(network.Activation[0].Len()).To(Equal(3))
			Expect(network.Activation[0].AtVec(0)).To(Equal(float64(1)))
			Expect(network.Activation[0].AtVec(1)).To(Equal(float64(2)))
			Expect(network.Activation[0].AtVec(2)).To(Equal(float64(3)))

			// First Layer Activations
			Expect(network.Activation[1].Len()).To(Equal(2))
			Expect(network.Activation[1].AtVec(0)).To(Equal(float64(8)))
			Expect(network.Activation[1].AtVec(1)).To(Equal(float64(14)))

			// Output Layer Activations
			Expect(network.Activation[2].Len()).To(Equal(1))
			Expect(network.Activation[2].AtVec(0)).To(Equal(float64(14)))

			Expect(output.AtVec(0)).To(BeNumerically("~", 14))
		})
	})
	context("GenerateDelta", func() {
		var network neuralnet.Network
		context("For a two layer network", func() {
			it.Before(func() {
				var err error
				network, err = neuralnet.NewNetwork(neuralnet.Config{
					LayerConfigs: []neuralnet.LayerConfig{
						{
							Size: 4,
						},
						{
							Size: 3,
							Func: TestFunc{},
						},
					},
					WeightInit: neuralnet.InitOne,
				})
				Expect(err).NotTo(HaveOccurred())

				Expect(len(network.Activation)).To(Equal(2))
				Expect(len(network.Zval)).To(Equal(2))

				network.Activation[1] = mat.NewVecDense(3, []float64{1, 2, 3})
				network.Zval[1] = mat.NewVecDense(3, []float64{0, 0, 0})
			})
			it("Calculates delta matrix", func() {
				delta, err := network.GenerateDelta(mat.NewVecDense(3, []float64{0, 0, 0}))

				Expect(err).NotTo(HaveOccurred())
				Expect(len(delta)).To(Equal(1))
				Expect(delta[0].RawVector().Data).To(Equal([]float64{1, 2, 3}))
			})
		})

		context("for multi layered networks", func() {
			it.Before(func() {
				var err error
				network, err = neuralnet.NewNetwork(neuralnet.Config{
					LayerConfigs: []neuralnet.LayerConfig{
						{
							Size: 5,
						},
						{
							Size: 4,
							Func: TestFunc{},
						},
						{
							Size: 3,
							Func: TestFunc{},
						},
						{
							Size: 2,
							Func: TestFunc{},
						},
					},
					WeightInit: neuralnet.InitOne,
				})
				Expect(err).NotTo(HaveOccurred())
			})
			it("Calculates delta matrix", func() {
				Expect(len(network.Activation)).To(Equal(4))
				Expect(len(network.Zval)).To(Equal(4))

				network.Activation[1] = mat.NewVecDense(4, []float64{1, 2, 3, 4})
				network.Zval[1] = mat.NewVecDense(4, []float64{0, 0, 0, 0})

				network.Activation[2] = mat.NewVecDense(3, []float64{1, 2, 3})
				network.Zval[2] = mat.NewVecDense(3, []float64{0, 0, 0})

				network.Activation[3] = mat.NewVecDense(2, []float64{1, 2})
				network.Zval[3] = mat.NewVecDense(2, []float64{0, 0})

				delta, err := network.GenerateDelta(mat.NewVecDense(2, []float64{0, 0}))

				Expect(err).NotTo(HaveOccurred())
				Expect(len(delta)).To(Equal(3))
				Expect(delta[0].RawVector().Data).To(Equal([]float64{9, 9, 9, 9}))
				Expect(delta[1].RawVector().Data).To(Equal([]float64{3, 3, 3}))
				Expect(delta[2].RawVector().Data).To(Equal([]float64{1, 2}))
			})
		})
	})

	context("Update", func() {
		context("When applying an update", func() {
			var (
				network neuralnet.Network
				delta   []*mat.VecDense
			)
			it.Before(func() {
				var err error
				network, err = neuralnet.NewNetwork(neuralnet.Config{
					LayerConfigs: []neuralnet.LayerConfig{
						{
							Size: 4,
						},
						{
							Size: 3,
							Func: TestFunc{},
						},
						{
							Size: 2,
							Func: TestFunc{},
						},
					},
					WeightInit: neuralnet.InitOne,
				})
				Expect(err).NotTo(HaveOccurred())

				delta = append(delta, mat.NewVecDense(3, []float64{1, 2, 3}))
				delta = append(delta, mat.NewVecDense(2, []float64{1, 2}))

				network.Activation[0] = mat.NewVecDense(4, []float64{1, 1, 1, 1})
				network.Activation[1] = mat.NewVecDense(3, []float64{1, 1, 1})
				network.Activation[2] = mat.NewVecDense(2, []float64{1, 1})
			})

			it("succeeds", func() {
				Expect(network.Update(delta)).To(Succeed())
				Expect(network.Bias).To(HaveLen(3))

				Expect(network.Bias[1]).To(Equal(mat.NewVecDense(3, []float64{-0.01, -0.02, -0.03})))
				Expect(network.Bias[2]).To(Equal(mat.NewVecDense(2, []float64{-0.01, -0.02})))

				Expect(network.Weights).To(HaveLen(2))
				Expect(network.Weights[0]).To(Equal(mat.NewDense(3, 4, []float64{
					0.99, 0.99, 0.99, 0.99,
					0.98, 0.98, 0.98, 0.98,
					0.97, 0.97, 0.97, 0.97,
				}),
				))
				Expect(network.Weights[1]).To(Equal(mat.NewDense(2, 3, []float64{
					0.99, 0.99, 0.99,
					0.98, 0.98, 0.98,
				}),
				))
			})
		})
	})
}
