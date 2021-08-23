package NeuralNetworks.Nets;

import NeuralNetworks.ActivationFunctions.*;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.LinkedList;

import Math.*;

public abstract class Net implements Serializable {

    protected final int seed;
    protected final ActivationFunction function;

    public Net(int seed, Activation activation) {
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
            default:
                function = new Linear();
                break;
        }
        this.function = function;
        this.seed = seed;
    }

    protected abstract void feedForward();

    protected abstract LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch;

    public abstract void train(int epoch, double learningRate, double etaDecrease, double momentum) throws MatrixRowColumnMismatch, MatrixDimensionMismatch;

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
