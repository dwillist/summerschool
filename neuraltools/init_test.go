package neuraltools_test

import (
	"testing"

	"github.com/sclevine/spec"
	"github.com/sclevine/spec/report"
)

func TestUnitSummerSchool(t *testing.T) {
	suite := spec.New("neuraltools", spec.Report(report.Terminal{}))
	suite("Tools", testTools)
	suite.Run(t)
}
