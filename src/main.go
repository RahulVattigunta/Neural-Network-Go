package main

import (
    "encoding/csv"
    "errors"
    "fmt"
    "log"
    "math"
    "math/rand"
    "os"
    "strconv"
    "time"

    "gonum.org/v1/gonum/floats"
    "gonum.org/v1/gonum/mat"
)

type neuralNet struct {
    config  neuralNetConfig
    wHidden *mat.Dense
    bHidden *mat.Dense
    wOut    *mat.Dense
    bOut    *mat.Dense
}

type neuralNetConfig struct {
    inputNeurons  int
    outputNeurons int
    hiddenNeurons int
    numEpochs     int
    learningRate  float64
}

func newNetwork(config neuralNetConfig) *neuralNet {
    return &neuralNet{config: config}
}

func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
    return sigmoid(x) * (1.0 - sigmoid(x))
}

func (nn *neuralNet) train(x, y *mat.Dense) error {
    randSource := rand.NewSource(time.Now().UnixNano())
    randGen := rand.New(randSource)

    wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
    bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
    wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
    bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

    for _, param := range [][]float64{
        wHidden.RawMatrix().Data,
        bHidden.RawMatrix().Data,
        wOut.RawMatrix().Data,
        bOut.RawMatrix().Data,
    } {
        for i := range param {
            param[i] = randGen.Float64()
        }
    }

    output := new(mat.Dense)

    if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
        return err
    }

    nn.wHidden = wHidden
    nn.bHidden = bHidden
    nn.wOut = wOut
    nn.bOut = bOut

    return nil
}

func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {
    for i := 0; i < nn.config.numEpochs; i++ {
        hiddenLayerInput := new(mat.Dense)
        hiddenLayerInput.Mul(x, wHidden)
        addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
        hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

        hiddenLayerActivations := new(mat.Dense)
        hiddenLayerActivations.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, hiddenLayerInput)

        outputLayerInput := new(mat.Dense)
        outputLayerInput.Mul(hiddenLayerActivations, wOut)
        addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
        outputLayerInput.Apply(addBOut, outputLayerInput)
        output.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, outputLayerInput)

        networkError := new(mat.Dense)
        networkError.Sub(y, output)

        slopeOutputLayer := new(mat.Dense)
        slopeOutputLayer.Apply(func(_, _ int, v float64) float64 { return sigmoidPrime(v) }, output)

        dOutput := new(mat.Dense)
        dOutput.MulElem(networkError, slopeOutputLayer)

        errorAtHiddenLayer := new(mat.Dense)
        errorAtHiddenLayer.Mul(dOutput, wOut.T())

        slopeHiddenLayer := new(mat.Dense)
        slopeHiddenLayer.Apply(func(_, _ int, v float64) float64 { return sigmoidPrime(v) }, hiddenLayerActivations)

        dHiddenLayer := new(mat.Dense)
        dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

        wOutAdj := new(mat.Dense)
        wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
        wOutAdj.Scale(nn.config.learningRate, wOutAdj)
        wOut.Add(wOut, wOutAdj)

        wHiddenAdj := new(mat.Dense)
        wHiddenAdj.Mul(x.T(), dHiddenLayer)
        wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
        wHidden.Add(wHidden, wHiddenAdj)
    }
    return nil
}

func main() {
    config := neuralNetConfig{
        inputNeurons:  4,
        outputNeurons: 3,
        hiddenNeurons: 3,
        numEpochs:     5000,
        learningRate:  0.3,
    }

    network := newNetwork(config)

    fmt.Println("Neural network initialized!")
}
