package nodefuncs_test

import (
	"testing"

	"github.com/sclevine/spec"
	"github.com/sclevine/spec/report"
)

func TestUnitNodeFuncs(t *testing.T) {
	suite := spec.New("NodeFuncs", spec.Report(report.Terminal{}))
	suite("NodeFuncs", testNodeFuncs)
	suite.Run(t)
}
