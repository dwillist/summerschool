package integration

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"gonum.org/v1/gonum/mat"
)

type Header struct {
	MagicNumber int32
	Count       int32
}

type ImageHeader struct {
	Header
	RowCount    int32
	ColumnCount int32
}

type ImageSet struct {
	ImageHeader
	Images []Image
}

type LabelSet struct {
	Header
	Labels []Label
}

type Image struct {
	RawData []byte
	Cols    int32
	Rows    int32
}

type Label byte

type Parsable interface {
	Parse(io.Reader) error
}

func (l Label) String() string {
	return string(byte(l) + byte(48))
}

func (l Label) Vec() *mat.VecDense {
	rawData := make([]float64, 10)
	rawData[int(l)] = 1.0

	return mat.NewVecDense(10, rawData)
}

func (i Image) String() string {
	var result string

	for r := int32(0); r < i.Rows; r++ {
		for c := int32(0); c < i.Cols; c++ {
			result += fmt.Sprintf("%03d", (i.RawData[r*i.Cols+c] + byte(48)))
		}

		result += "\n"
	}

	return result
}

func (i Image) Vec() *mat.VecDense {
	totalSize := int(i.Cols * i.Rows)
	rawVec := make([]float64, totalSize)

	for idx := 0; idx < totalSize; idx++ {
		rawVec[idx] = float64(i.RawData[idx]) / float64(255)
	}

	return mat.NewVecDense(totalSize, rawVec)
}

func (ls *LabelSet) Parse(input io.Reader) error {
	headerSize := binary.Size(ls.Header)
	headerReader := io.LimitReader(input, int64(headerSize))

	err := binary.Read(headerReader, binary.BigEndian, &ls.Header)
	if err != nil {
		return fmt.Errorf("error parsing label header: %s", err)
	}

	ls.Labels = make([]Label, ls.Count)
	err = binary.Read(input, binary.BigEndian, &ls.Labels)

	if err != nil {
		return fmt.Errorf("error parsing lable data: %s", err)
	}

	return nil
}

func (is *ImageSet) Parse(input io.Reader) error {
	headerSize := binary.Size(is.ImageHeader)
	headerReader := io.LimitReader(input, int64(headerSize))

	err := binary.Read(headerReader, binary.BigEndian, &is.ImageHeader)
	if err != nil {
		return fmt.Errorf("error parsing label header: %s", err)
	}

	var img Image
	img.Rows = is.ImageHeader.RowCount
	img.Cols = is.ImageHeader.ColumnCount

	for i := int32(0); i < is.Count; i++ {
		img.RawData = make([]byte, img.Rows*img.Cols)
		imgReader := io.LimitReader(input, int64(img.Rows*img.Cols))
		err = binary.Read(imgReader, binary.BigEndian, &img.RawData)

		if err != nil {
			return fmt.Errorf("error parsing image at index %d", i)
		}

		is.Images = append(is.Images, img)
	}

	return nil
}

// parse label set from gz encoded file
func newSet(filepath string, data Parsable) error {
	fileReader, err := os.Open(filepath)
	if err != nil {
		return err
	}

	defer fileReader.Close()

	fz, err := gzip.NewReader(fileReader)
	if err != nil {
		return err
	}

	defer fz.Close()

	return data.Parse(fz)
}

func NewImageSet(filepath string) (ImageSet, error) {
	var result ImageSet
	err := newSet(filepath, &result)

	if err != nil {
		return result, err
	}

	return result, nil
}

func NewLabelSet(filepath string) (LabelSet, error) {
	var result LabelSet
	err := newSet(filepath, &result)

	if err != nil {
		return result, err
	}

	return result, nil
}

func (ls LabelSet) Vectorize() []*mat.VecDense {
	result := make([]*mat.VecDense, ls.Count)

	for idx, label := range ls.Labels {
		result[idx] = label.Vec()
	}

	return result
}

func (is ImageSet) Vectorize() []*mat.VecDense {
	result := make([]*mat.VecDense, is.Count)

	for idx, image := range is.Images {
		result[idx] = image.Vec()
	}

	return result
}
