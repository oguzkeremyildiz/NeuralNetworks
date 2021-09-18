package NeuralNetworks.Net;

import java.io.Serializable;
import java.util.LinkedList;

import Math.*;
import NeuralNetworks.ActivationFunction.Activation;
import NeuralNetworks.ActivationFunction.ActivationFunction;
import NeuralNetworks.ActivationFunction.TanH;
import NeuralNetworks.InstanceList.VectorizedInstanceList;

public class LSTM extends RecurrentNeuralNetwork implements Serializable {

    private final ActivationFunction tanh;

    public LSTM(int seed, LinkedList<Integer> hiddenLayers, VectorizedInstanceList instanceList, Activation activation) {
        super(seed, hiddenLayers, instanceList, activation, NetworkType.LSTM);
        if (activation.equals(Activation.TANH)) {
            tanh = null;
        } else {
            tanh = new TanH();
        }
    }

    @Override
    protected void setWeights(LinkedList<Matrix> deltaWeights, LinkedList<Matrix> oldDeltaWeights, double momentum) {

    }

    @Override
    protected void feedForward() {

    }

    @Override
    public void train(int epoch, double learningRate, double etaDecrease, double momentum) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {

    }

    @Override
    protected LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) {
        return null;
    }
}
