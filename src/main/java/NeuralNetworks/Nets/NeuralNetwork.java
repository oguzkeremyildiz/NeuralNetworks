package NeuralNetworks.Nets;

import NeuralNetworks.ActivationFunctions.*;
import NeuralNetworks.Bias;
import NeuralNetworks.Instance;
import NeuralNetworks.InstanceList.InstanceList;
import NeuralNetworks.Layer;

import java.io.*;
import java.util.LinkedList;
import Math.*;

public class NeuralNetwork extends Net implements Serializable {

    private final Layer[] layers;
    private final InstanceList instanceList;
    private final Bias[] biases;

    public NeuralNetwork(int seed, LinkedList<Integer> hiddenLayers, InstanceList instanceList, Activation activation) {
        super(seed, activation);
        this.instanceList = instanceList;
        hiddenLayers.addFirst(instanceList.getInput());
        hiddenLayers.addLast(instanceList.getOutput());
        this.layers = new Layer[hiddenLayers.size()];
        for (int i = 0; i < hiddenLayers.size(); i++) {
            if (i + 1 < hiddenLayers.size()) {
                this.layers[i] = new Layer(hiddenLayers.get(i), hiddenLayers.get(i + 1), seed);
            } else {
                this.layers[i] = new Layer(hiddenLayers.get(i));
            }
        }
        biases = new Bias[hiddenLayers.size() - 1];
        for (int i = 0; i < biases.length; i++) {
            biases[i] = new Bias(seed, layers[i + 1].size());
        }
    }

    protected void feedForward() {
        for (int i = 0; i < layers.length - 1; i++) {
            for (int j = 0; j < layers[i + 1].size(); j++) {
                double sum = 0.0;
                for (int k = 0; k < layers[i].size(); k++) {
                    sum += layers[i].getNeuron(k).getWeight(j) * layers[i].getNeuron(k).getValue();
                }
                sum += biases[i].getValue(j);
                if (i + 1 != layers.length - 1) {
                    sum = function.calculateForward(sum);
                }
                layers[i + 1].getNeuron(j).setValue(sum);
            }
        }
        if (layers[layers.length - 1].size() > 2) {
            layers[layers.length - 1].softmax();
        }
    }

    private void createInputVector(Instance<String> instance) {
        for (int i = 0; i < layers[0].size(); i++) {
            layers[0].getNeuron(i).setValue(Double.parseDouble(instance.get(i)));
        }
    }

    private Matrix calculateError(int i, LinkedList<Matrix> deltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        deltaWeights.set(0, layers[i + 1].weightsToMatrix().multiply(deltaWeights.getFirst()));
        deltaWeights.set(0, deltaWeights.getFirst().elementProduct(function.calculateBack(layers[i + 1].neuronsToVector())));
        return deltaWeights.getFirst();
    }

    private void setWeights(LinkedList<Matrix> deltaWeights, LinkedList<Matrix> oldDeltaWeights, double momentum) {
        for (int t = 0; t < deltaWeights.size(); t++) {
            Matrix weights = deltaWeights.get(t);
            for (int i = 0; i < weights.getRow(); i++) {
                for (int j = 0; j < weights.getColumn(); j++) {
                    if (!oldDeltaWeights.isEmpty()) {
                        weights.addValue(i, j, momentum * oldDeltaWeights.get(t).getValue(i, j));
                    }
                    if (j > 0) {
                        layers[t].getNeuron(j - 1).addWeight(i, weights.getValue(i, j));
                    } else {
                        biases[t].addWeight(i, weights.getValue(i, j));
                    }
                }
            }
        }
    }

    private void calculateRMinusY(LinkedList<Matrix> deltaWeights, int classInfo, double learningRate) throws MatrixRowColumnMismatch {
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

    private LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        LinkedList<Matrix> deltaWeights = new LinkedList<>();
        calculateRMinusY(deltaWeights, classInfo, learningRate);
        for (int i = layers.length - 3; i > -1; i--) {
            Matrix currentError = calculateError(i, deltaWeights);
            deltaWeights.set(0, deltaWeights.getFirst().multiply(layers[i].neuronsToMatrix()));
            if (i > 0) {
                deltaWeights.addFirst(currentError);
            }
        }
        setWeights(deltaWeights, oldDeltaWeights, momentum);
        return deltaWeights;
    }

    public void train(int epoch, double learningRate, double etaDecrease, double momentum) throws MatrixDimensionMismatch, MatrixRowColumnMismatch {
        LinkedList<Matrix> oldDeltaWeights = new LinkedList<>();
        for (int i = 0; i < epoch; i++) {
            instanceList.shuffle(seed);
            for (int j = 0; j < instanceList.size(); j++) {
                createInputVector(instanceList.getInstance(j));
                feedForward();
                oldDeltaWeights = backpropagation(instanceList.get(instanceList.getInstance(j).getLast()), learningRate, momentum, oldDeltaWeights);
            }
            learningRate *= etaDecrease;
        }
    }

    public String predict(Instance<String> instance) {
        createInputVector(instance);
        feedForward();
        if (instanceList.getOutput() == 1) {
            double outputValue = layers[layers.length - 1].getNeuron(0).getValue();
            if (outputValue >= 0.5) {
                return instanceList.get(1);
            }
            return instanceList.get(0);
        }
        double bestValue = Integer.MIN_VALUE;
        int bestNeuron = -1;
        for (int i = 0; i < layers[layers.length - 1].size(); i++) {
            if (layers[layers.length - 1].getNeuron(i).getValue() > bestValue) {
                bestValue = layers[layers.length - 1].getNeuron(i).getValue();
                bestNeuron = i;
            }
        }
        return instanceList.get(bestNeuron);
    }

    public double test(InstanceList list) {
        int correct = 0;
        int total = 0;
        for (int i = 0; i < list.size(); i++) {
            Instance<String> instance = list.getInstance(i);
            if (instance.getLast().equals(predict(instance))) {
                correct++;
            }
            total++;
        }
        return correct * 100.00 / total;
    }
}