package NeuralNetworks.Net;

import NeuralNetworks.ActivationFunction.*;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.LinkedList;

import Math.*;
import NeuralNetworks.Initializer.Initializer;
import NeuralNetworks.Initializer.InitializerFunction;
import NeuralNetworks.Initializer.RandomInitializer;
import NeuralNetworks.Initializer.XavierInitializer;
import NeuralNetworks.Neuron.Bias;
import NeuralNetworks.InstanceList.BasicInstanceList;
import NeuralNetworks.Layer.Layer;

public abstract class Net<T> implements Serializable {

    protected final int seed;
    protected final LinkedList<ActivationFunction> function;
    protected final BasicInstanceList<T> instanceList;
    protected final Layer[] layers;
    protected final Bias[] biases;

    public Net(int seed, LinkedList<Activation> activations, BasicInstanceList<T> instanceList, LinkedList<Integer> hiddenLayers) {
        this.instanceList = instanceList;
        hiddenLayers.addFirst(instanceList.getInput());
        hiddenLayers.addLast(instanceList.getOutput());
        this.layers = new Layer[hiddenLayers.size()];
        this.function = new LinkedList<>();
        for (Activation activation : activations) {
            switch (activation) {
                case SIGMOID:
                    function.add(new Sigmoid());
                    break;
                case RELU:
                    function.add(new ReLU());
                    break;
                case TANH:
                    function.add(new TanH());
                    break;
                case LEAKYRELU:
                    function.add(new LeakyReLU());
                    break;
                case ELU:
                    function.add(new ELU());
                    break;
                default:
                    function.add(new Linear());
                    break;
            }
        }
        this.seed = seed;
        biases = new Bias[hiddenLayers.size() - 1];
    }

    protected InitializerFunction findInitializerFunction(Initializer initializer, int size) {
        InitializerFunction initializerFunction;
        switch (initializer) {
            case XAVIER:
                initializerFunction = new XavierInitializer(size);
                break;
            default:
                initializerFunction = new RandomInitializer();
        }
        return initializerFunction;
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
                    matrix.setValue(j, 0, learningRate * (1 - layers[layers.length - 1].getValue(j)));
                } else {
                    matrix.setValue(j, 0, learningRate * -layers[layers.length - 1].getValue(j));
                }
            }
        } else {
            matrix.setValue(0, 0, learningRate * (classInfo - layers[layers.length - 1].getValue(0)));
        }
        deltaWeights.addFirst(matrix);
        deltaWeights.set(0, deltaWeights.getFirst().multiply(layers[layers.length - 2].neuronsToMatrix()));
        if (layers.length > 2) {
            deltaWeights.addFirst(matrix);
        }
    }

    protected void calculateError(int i, LinkedList<Matrix> deltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        deltaWeights.set(0, layers[i + 1].weightsToMatrix().multiply(deltaWeights.getFirst()));
        deltaWeights.set(0, deltaWeights.getFirst().elementProduct(function.get(i).calculateBack(layers[i + 1].neuronsToVector())));
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
