package NeuralNetworks.Nets;

import NeuralNetworks.ActivationFunctions.*;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.LinkedList;

import Math.*;
import NeuralNetworks.Bias;
import NeuralNetworks.InstanceList.BasicInstanceList;
import NeuralNetworks.Layer;

public abstract class Net<T> implements Serializable {

    protected final int seed;
    protected final ActivationFunction function;
    protected final BasicInstanceList<T> instanceList;
    protected final Layer[] layers;
    protected final Bias[] biases;

    public Net(int seed, Activation activation, BasicInstanceList<T> instanceList, LinkedList<Integer> hiddenLayers) {
        this.instanceList = instanceList;
        hiddenLayers.addFirst(instanceList.getInput());
        hiddenLayers.addLast(instanceList.getOutput());
        this.layers = new Layer[hiddenLayers.size()];
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
            default:
                function = new Linear();
                break;
        }
        this.function = function;
        this.seed = seed;
        biases = new Bias[hiddenLayers.size() - 1];
        for (int i = 0; i < biases.length; i++) {
            biases[i] = new Bias(seed, hiddenLayers.get(i + 1));
        }
    }

    protected abstract void setWeights(LinkedList<Matrix> deltaWeights, LinkedList<Matrix> oldDeltaWeights, double momentum);

    protected abstract void feedForward();

    protected abstract LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch;

    public abstract void train(int epoch, double learningRate, double etaDecrease, double momentum) throws MatrixRowColumnMismatch, MatrixDimensionMismatch;

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

    protected Matrix calculateError(int i, LinkedList<Matrix> deltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        deltaWeights.set(0, layers[i + 1].weightsToMatrix().multiply(deltaWeights.getFirst()));
        deltaWeights.set(0, deltaWeights.getFirst().elementProduct(function.calculateBack(layers[i + 1].neuronsToVector())));
        return deltaWeights.getFirst();
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
