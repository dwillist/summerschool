package neuralnet_test

import (
	"math/rand"
	"testing"

	"github.com/sclevine/spec"
	"github.com/sclevine/spec/report"
)

func TestUnitSummerSchool(t *testing.T) {
	rand.Seed(92)

	suite := spec.New("neuralnet", spec.Report(report.Terminal{}))
	suite("Network", testNetwork)
	suite.Run(t)
}
