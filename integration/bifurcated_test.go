package integration_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/sclevine/spec"
	"github.com/dwillist/summerschool/v2/neuralnet"
	"github.com/dwillist/summerschool/v2/neuralnet/nodefuncs"
	"github.com/dwillist/summerschool/v2/neuraltools"
	"gonum.org/v1/gonum/mat"

	. "github.com/onsi/gomega"
)

func testBifurcated(t *testing.T, context spec.G, it spec.S) {
	var (
		Expect = NewWithT(t).Expect
	)

	context("Bifurcated Data", func() {
		var (
			network neuralnet.Network
		)
		context("when using Bifurcated Data to trian a network", func() {
			it.Before(func() {
				var err error
				network, err = neuralnet.NewNetwork(neuralnet.Config{
					LayerConfigs: []neuralnet.LayerConfig{
						{
							Size: 2,
						},
						{
							Size: 2,
							Func: nodefuncs.Sigmoid{},
						},
						{
							Size: 2,
							Func: nodefuncs.Sigmoid{},
						},
					},
					WeightInit: neuralnet.InitRandom,
				})

				Expect(err).NotTo(HaveOccurred())
			})
			it("succeeds", func() {
				var generatedData struct {
					Inputs    [][]float64 `json:"inputs"`
					Solutions [][]float64 `json:"solutions"`
				}

				rawDataPath := filepath.Join("testdata", "bifurcated-test.json")
				rawDataReader, err := os.Open(rawDataPath)
				Expect(err).NotTo(HaveOccurred())

				Expect(json.NewDecoder(rawDataReader).Decode(&generatedData)).To(Succeed())

				var inputVecs []*mat.VecDense
				var solutionVecs []*mat.VecDense

				for _, input := range generatedData.Inputs {
					inputVecs = append(inputVecs, mat.NewVecDense(2, input))
				}

				for _, solution := range generatedData.Solutions {
					solutionVecs = append(solutionVecs, mat.NewVecDense(2, solution))
				}

				dataPairs, err := neuraltools.NewDataPair(inputVecs, solutionVecs)
				Expect(err).NotTo(HaveOccurred())

				correctList, err := neuraltools.TestAndTrain(&network, 100, 1, neuraltools.MaxJudge, dataPairs...)
				Expect(err).NotTo(HaveOccurred())

				Expect(correctList).To(HaveLen(100))

				finalCorrectCount := correctList[len(correctList)-1]
				Expect(finalCorrectCount).To(BeNumerically(">", 90))

			})
		})
	})
}
