package nodefuncs_test

import (
	"testing"

	"github.com/dwillist/summerschool/v2/neuralnet/nodefuncs"

	"github.com/sclevine/spec"

	. "github.com/onsi/gomega"
	"gonum.org/v1/gonum/mat"
)

func testNodeFuncs(t *testing.T, context spec.G, it spec.S) {
	var (
		Expect = NewWithT(t).Expect
	)

	context("Node Functions", func() {

		context("SoftMax", func() {
			var softMax nodefuncs.Softmax

			context("CalcVal", func() {
				it("calculates correctly", func() {

					y1 := softMax.CalcVal(2, mat.NewVecDense(3, []float64{2, 1, 0.1}))
					Expect(y1).To(BeNumerically("~", float64(0.6590011388859679)))

					y2 := softMax.CalcVal(1, mat.NewVecDense(3, []float64{2, 1, 0.1}))
					Expect(y2).To(BeNumerically("~", float64(0.2424329707047139)))

					y3 := softMax.CalcVal(0.1, mat.NewVecDense(3, []float64{2, 1, 0.1}))
					Expect(y3).To(BeNumerically("~", float64(0.09856589040931818)))

					Expect(y1 + y2 + y3).To(BeNumerically("~", float64(1)))
				})
			})

			context("CalcDiff", func() {
				it("calculates correctly", func() {

					y1 := softMax.CalcDiff(2, mat.NewVecDense(3, []float64{2, 1, 0.1}))
					Expect(y1).To(BeNumerically("~", float64(0.22471863783296514)))

					y2 := softMax.CalcDiff(1, mat.NewVecDense(3, []float64{2, 1, 0.1}))
					Expect(y2).To(BeNumerically("~", float64(0.1836592254200012)))

					y3 := softMax.CalcDiff(0.1, mat.NewVecDense(3, []float64{2, 1, 0.1}))
					Expect(y3).To(BeNumerically("~", float64(0.08885065565713646)))
				})
			})
		})

		context("Relu", func() {
			var relu nodefuncs.Relu

			context("CalcVal", func() {
				it("Calculates Correctly", func() {
					y := relu.CalcVal(0, nil)
					Expect(y).To(BeNumerically("~", float64(0)))

					y = relu.CalcVal(1, nil)
					Expect(y).To(BeNumerically("~", float64(1)))

					y = relu.CalcVal(-1, nil)
					Expect(y).To(BeNumerically("~", float64(0)))
				})
			})

			context("CalcDiff", func() {
				it("Calculates Correctly", func() {
					y := relu.CalcDiff(-1, nil)
					Expect(y).To(BeNumerically("~", float64(0)))

					y = relu.CalcDiff(2, nil)
					Expect(y).To(BeNumerically("~", float64(1)))
				})
			})
		})

		context("Sigmoid", func() {
			var sigmoid nodefuncs.Sigmoid

			context("CalcVal", func() {
				it("Calculates Correctly", func() {
					y := sigmoid.CalcVal(0, nil)
					Expect(y).To(BeNumerically("~", float64(0.5)))

					y = sigmoid.CalcVal(1, nil)
					Expect(y).To(BeNumerically("~", float64(0.73105857863000)))
				})
			})

			context("CalcDiff", func() {
				it("Calculates Correctly", func() {
					y := sigmoid.CalcDiff(0, nil)
					Expect(y).To(BeNumerically("~", float64(0.25)))

					y = sigmoid.CalcDiff(1, nil)
					Expect(y).To(BeNumerically("~", float64(0.19661193324148185)))
				})
			})
		})

		context("Identity", func() {
			var id nodefuncs.Identity

			context("CalcVal", func() {
				it("Calculates Correctly", func() {
					y := id.CalcVal(0, nil)
					Expect(y).To(BeNumerically("~", float64(0)))

					y = id.CalcVal(1, nil)
					Expect(y).To(BeNumerically("~", float64(1)))
				})
			})

			context("CalcDiff", func() {
				it("Calculates Correctly", func() {
					y := id.CalcDiff(3, nil)
					Expect(y).To(BeNumerically("~", float64(0)))
				})
			})
		})
	})

	context("ApplyFunc", func() {
		var vec *mat.VecDense
		it.Before(func() {
			vec = mat.NewVecDense(3, nil)
		})

		it("updates values of underlying matrix", func() {
			nodefuncs.ApplyFunc(vec, func(x float64, _ mat.Vector) float64 { return x + 5 })

			for _, val := range vec.RawVector().Data {
				Expect(val).To(BeNumerically("~", 5))
			}
		})
	})
}
