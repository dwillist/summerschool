package integration_test

import (
	"math/rand"
	"testing"

	"github.com/sclevine/spec"
	"github.com/sclevine/spec/report"

	"gonum.org/v1/gonum/blas/blas64"
	blas_netlib "gonum.org/v1/netlib/blas/netlib"
)

func TestUnitSummerSchool(t *testing.T) {
	rand.Seed(92)

	// Use optimized libs for matrix mult
	blas64.Use(blas_netlib.Implementation{}) // This improves Mul time from ~3s to ~0.6s

	suite := spec.New("Integration", spec.Report(report.Terminal{}))
	suite("Test Bifurcated data", testBifurcated)
	suite("Test Target data", testTarget)
	suite("Test MNIST", testMNIST)
	suite.Run(t)
}
