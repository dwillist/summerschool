package neuraltools_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/sclevine/spec"
	"github.com/dwillist/summerschool/v2/neuraltools"
	"github.com/dwillist/summerschool/v2/neuraltools/fakes"
	"gonum.org/v1/gonum/mat"

	. "github.com/onsi/gomega"
)

func testTools(t *testing.T, context spec.G, it spec.S) {
	var (
		Expect    = NewWithT(t).Expect
		fakeJudge = func(a, b *mat.VecDense) bool { return mat.Equal(a, b) }
	)

	context("NewDataPair", func() {
		var (
			inputs    []*mat.VecDense
			solutions []*mat.VecDense
		)

		it.Before(func() {
			inputs = []*mat.VecDense{
				mat.NewVecDense(1, []float64{1}),
				mat.NewVecDense(1, []float64{2}),
			}

			solutions = []*mat.VecDense{
				mat.NewVecDense(1, []float64{2}),
				mat.NewVecDense(1, []float64{3}),
			}
		})

		it("succeeds", func() {
			out, err := neuraltools.NewDataPair(inputs, solutions)
			Expect(err).NotTo(HaveOccurred())
			Expect(out).To(HaveLen(2))

			Expect(out[0]).To(Equal(
				neuraltools.DataPair{
					Input:    inputs[0],
					Solution: solutions[0],
				}),
			)

			Expect(out[1]).To(Equal(
				neuraltools.DataPair{
					Input:    inputs[1],
					Solution: solutions[1],
				}),
			)
		})

		context("failure cases", func() {
			context("input and solution are of unequal size", func() {
				it("returns an error", func() {
					inputs = []*mat.VecDense{
						mat.NewVecDense(1, nil),
					}

					_, err := neuraltools.NewDataPair(inputs, solutions)
					Expect(err).To(MatchError("input and solution of unequal cardenality 1, 2"))
				})
			})
		})
	})

	context("Train", func() {
		var (
			trainingData []neuraltools.DataPair
			network      *fakes.Network
		)

		it.Before(func() {
			trainingData = []neuraltools.DataPair{
				{
					Input:    mat.NewVecDense(2, nil),
					Solution: mat.NewVecDense(2, nil),
				},
			}

			network = &fakes.Network{}
		})

		it("succeeds", func() {
			Expect(neuraltools.Train(network, 1, trainingData...)).To(Succeed())
		})

		context("falure cases", func() {
			context("network Calculate fails", func() {
				it("returns an error", func() {
					network.CalculateCall.Returns.Error = errors.New("error")

					err := neuraltools.Train(network, 1, trainingData...)
					Expect(err).To(MatchError("network calculation failed on input at index: 0"))
				})
			})

			context("network GenerateDelta fails", func() {
				it("returns an error", func() {
					network.GenerateDeltaCall.Returns.Error = errors.New("error")

					err := neuraltools.Train(network, 1, trainingData...)
					Expect(err).To(HaveOccurred())
					Expect(err).To(MatchError("network delta generation failed on solution at index: 0"))
				})
			})

			context("network Update fails", func() {
				it("returns an error", func() {
					network.UpdateCall.Returns.Error = errors.New("error")

					err := neuraltools.Train(network, 1, trainingData...)
					Expect(err).To(MatchError("network update failed on delta at index: 0"))
				})
			})
		})
	})

	context("Test", func() {
		var (
			trainingData []neuraltools.DataPair
			network      *fakes.Calculator
		)

		it.Before(func() {
			trainingData = []neuraltools.DataPair{
				{
					Input:    mat.NewVecDense(3, []float64{0, 1, 0}),
					Solution: mat.NewVecDense(3, []float64{0, 0, 0}),
				},
			}

			network = &fakes.Calculator{}
		})

		it("calculates correct count", func() {
			index := 0
			network.CalculateCall.Stub = func(input *mat.VecDense) (*mat.VecDense, error) {
				if input.AtVec(index) == float64(0) {
					index++
					return mat.NewVecDense(3, []float64{0, 0, 0}), nil
				}
				index++
				return mat.NewVecDense(3, []float64{1, 1, 1}), nil
			}

			correct, err := neuraltools.Test(network, fakeJudge, trainingData...)
			Expect(err).NotTo(HaveOccurred())

			Expect(correct).To(Equal(1))
		})
		context("failure cases", func() {
			it("fails during calculation", func() {
				network.CalculateCall.Returns.Error = fmt.Errorf("error occurred")
				_, err := neuraltools.Test(network, fakeJudge, trainingData...)

				Expect(err).To(MatchError("error on input 0 calculation: error occurred"))
			})
		})
	})

	context("MaxJudge", func() {
		context("when indicies of max elements are equal", func() {
			it("return true", func() {
				Expect(neuraltools.MaxJudge(
					mat.NewVecDense(3, []float64{1, 2, 3}),
					mat.NewVecDense(3, []float64{5, 2, 7}),
				)).To(Equal(true))
			})
		})

		context("when indicies of max elements are NOT equal", func() {
			it("return false", func() {
				Expect(neuraltools.MaxJudge(
					mat.NewVecDense(3, []float64{1, 2, 3}),
					mat.NewVecDense(3, []float64{5, 2, 1}),
				)).To(Equal(false))
			})
		})
	})
}
