package integration_test

import (
	"path/filepath"
	"testing"

	"github.com/sclevine/spec"
	"github.com/dwillist/summerschool/v2/integration"
	"github.com/dwillist/summerschool/v2/neuralnet"
	"github.com/dwillist/summerschool/v2/neuralnet/nodefuncs"
	"github.com/dwillist/summerschool/v2/neuraltools"

	. "github.com/onsi/gomega"
)

func testMNIST(t *testing.T, context spec.G, it spec.S) {
	var (
		Expect = NewWithT(t).Expect
	)

	context("integration", func() {
		context("when using the MNIST dataset to train a network", func() {
			var (
				trainLabels integration.LabelSet
				testLabels  integration.LabelSet

				trainImages integration.ImageSet
				testImages  integration.ImageSet

				network neuralnet.Network
			)
			it.Before(func() {
				var err error
				trainLabels, err = integration.NewLabelSet(filepath.Join("testdata", "train-labels-idx1-ubyte.gz"))
				Expect(err).NotTo(HaveOccurred())
				testLabels, err = integration.NewLabelSet(filepath.Join("testdata", "t10k-labels-idx1-ubyte.gz"))
				Expect(err).NotTo(HaveOccurred())

				trainImages, err = integration.NewImageSet(filepath.Join("testdata", "train-images-idx3-ubyte.gz"))
				Expect(err).NotTo(HaveOccurred())
				testImages, err = integration.NewImageSet(filepath.Join("testdata", "t10k-images-idx3-ubyte.gz"))
				Expect(err).NotTo(HaveOccurred())

				// setup network
				network, err = neuralnet.NewNetwork(neuralnet.Config{
					LayerConfigs: []neuralnet.LayerConfig{
						{
							Size: 28 * 28,
						},
						{
							Size: 10,
							Func: nodefuncs.Sigmoid{},
						},
					},
					WeightInit: neuralnet.InitRandom,
				})

				Expect(err).NotTo(HaveOccurred())
			})
			it("succeeds", func() {
				//resizeing
				trainSize := 60000
				trainImages.Images = trainImages.Images[:trainSize]
				trainImages.Count = int32(trainSize)
				trainLabels.Labels = trainLabels.Labels[:trainSize]
				trainLabels.Count = int32(trainSize)

				testSize := 10000
				testImages.Images = testImages.Images[:testSize]
				testImages.Count = int32(testSize)
				testLabels.Labels = testLabels.Labels[:testSize]
				testLabels.Count = int32(testSize)

				trainData, err := neuraltools.NewDataPair(trainImages.Vectorize(), trainLabels.Vectorize())
				Expect(err).NotTo(HaveOccurred())

				testData, err := neuraltools.NewDataPair(testImages.Vectorize(), testLabels.Vectorize())
				Expect(err).NotTo(HaveOccurred())

				epochCount := 10
				// start := time.Now()
				for i := 0; i < epochCount; i++ {
					err = neuraltools.Train(&network, 1, trainData...)
					Expect(err).NotTo(HaveOccurred())
					//fmt.Printf("epoch %d, completed in %s\n", i+1, time.Since(start))
					//start = time.Now()

				}

				correctCount, err := neuraltools.Test(&network, neuraltools.MaxJudge, testData...)
				Expect(err).NotTo(HaveOccurred())

				Expect(correctCount).To(BeNumerically(">=", 8000))
			})
		})
	})
}
