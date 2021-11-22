package NeuralNetworks.Net;

import NeuralNetworks.ActivationFunction.*;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.LinkedList;

import Math.*;
import NeuralNetworks.InstanceList.InstanceList;
import NeuralNetworks.Layer.LSTMLayer;
import NeuralNetworks.Layer.RecurrentLayer;
import NeuralNetworks.Neuron.Bias;
import NeuralNetworks.InstanceList.BasicInstanceList;
import NeuralNetworks.Layer.Layer;
import NeuralNetworks.Neuron.LSTMBias;

public abstract class Net<T> implements Serializable {

    protected final int seed;
    protected final ActivationFunction function;
    protected final BasicInstanceList<T> instanceList;
    protected final Layer[] layers;
    protected final Bias[] biases;

    public Net(int seed, Activation activation, BasicInstanceList<T> instanceList, LinkedList<Integer> hiddenLayers, NetworkType type) {
        this.instanceList = instanceList;
        hiddenLayers.addFirst(instanceList.getInput());
        hiddenLayers.addLast(instanceList.getOutput());
        this.layers = new Layer[hiddenLayers.size()];
        switch (type) {
            case NEURALNETWORK:
                int inputSize = ((InstanceList) instanceList).inputSize();
                if (inputSize > -1) {
                    hiddenLayers.set(0, inputSize);
                }
                for (int i = 0; i < hiddenLayers.size(); i++) {
                    if (i + 1 < hiddenLayers.size()) {
                        this.layers[i] = new Layer(hiddenLayers.get(i), hiddenLayers.get(i + 1), seed);
                    } else {
                        this.layers[i] = new Layer(hiddenLayers.get(i));
                    }
                }
                break;
            case RECURRENTNEURALNETWORK:
                this.layers[0] = new Layer(hiddenLayers.get(0), hiddenLayers.get(1), seed);
                for (int i = 1; i < hiddenLayers.size(); i++) {
                    if (i + 1 < hiddenLayers.size()) {
                        this.layers[i] = new RecurrentLayer(hiddenLayers.get(i), hiddenLayers.get(i + 1), seed);
                    } else {
                        this.layers[i] = new Layer(hiddenLayers.get(i));
                    }
                }
                break;
            case LSTM:
                this.layers[0] = new LSTMLayer(hiddenLayers.get(0), hiddenLayers.get(1), seed, false);
                for (int i = 1; i < hiddenLayers.size(); i++) {
                    if (i + 1 < hiddenLayers.size()) {
                        this.layers[i] = new LSTMLayer(hiddenLayers.get(i), hiddenLayers.get(i + 1), seed, true);
                    } else {
                        this.layers[i] = new Layer(hiddenLayers.get(i));
                    }
                }
                break;
            default:
                break;
        }
        ActivationFunction function;
        switch (activation) {
            case SIGMOID:
                function = new Sigmoid();
                break;
            case RELU:
                function = new ReLU();
                break;
            case TANH:
                function = new TanH();
                break;
            case LEAKYRELU:
                function = new LeakyReLU();
                break;
            case ELU:
                function = new ELU();
                break;
            default:
                function = new Linear();
                break;
        }
        this.function = function;
        this.seed = seed;
        biases = new Bias[hiddenLayers.size() - 1];
        if (type == NetworkType.LSTM) {
            for (int i = 0; i < biases.length - 1; i++) {
                biases[i] = new LSTMBias(seed, hiddenLayers.get(i + 1));
            }
            biases[biases.length - 1] = new Bias(seed, hiddenLayers.get(biases.length));
        } else {
            for (int i = 0; i < biases.length; i++) {
                biases[i] = new Bias(seed, hiddenLayers.get(i + 1));
            }
        }
    }

    protected abstract void setWeights(LinkedList<Matrix> deltaWeights, LinkedList<Matrix> oldDeltaWeights, double momentum);

    protected abstract void feedForward() throws VectorSizeMismatch;

    protected abstract LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch;

    public abstract void train(int epoch, double learningRate, double etaDecrease, double momentum) throws MatrixRowColumnMismatch, MatrixDimensionMismatch, VectorSizeMismatch;

    protected void calculateRMinusY(LinkedList<Matrix> deltaWeights, int classInfo, double learningRate) throws MatrixRowColumnMismatch {
        Matrix matrix = new Matrix(layers[layers.length - 1].size(), 1);
        if (matrix.getRow() > 1) {
            for (int j = 0; j < matrix.getRow(); j++) {
                if (classInfo == j) {
                    matrix.setValue(j, 0, learningRate * (1 - layers[layers.length - 1].getNeuron(j).getValue()));
                } else {
                    matrix.setValue(j, 0, learningRate * -layers[layers.length - 1].getNeuron(j).getValue());
                }
            }
        } else {
            matrix.setValue(0, 0, learningRate * (classInfo - layers[layers.length - 1].getNeuron(0).getValue()));
        }
        deltaWeights.addFirst(matrix);
        deltaWeights.set(0, deltaWeights.getFirst().multiply(layers[layers.length - 2].neuronsToMatrix()));
        if (layers.length > 2) {
            deltaWeights.addFirst(matrix);
        }
    }

    protected void calculateError(int i, LinkedList<Matrix> deltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        deltaWeights.set(0, layers[i + 1].weightsToMatrix().multiply(deltaWeights.getFirst()));
        deltaWeights.set(0, deltaWeights.getFirst().elementProduct(function.calculateBack(layers[i + 1].neuronsToVector())));
    }

    public void save(String fileName) {
        try {
            FileOutputStream outFile = new FileOutputStream(fileName);
            ObjectOutputStream outObject = new ObjectOutputStream(outFile);
            outObject.writeObject(this);
        } catch (IOException var5) {
            var5.printStackTrace();
        }
    }
}
