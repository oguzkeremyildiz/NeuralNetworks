package NeuralNetworks.Net;

import NeuralNetworks.ActivationFunction.Activation;
import NeuralNetworks.InstanceList.Instance;
import NeuralNetworks.InstanceList.VectorizedInstanceList;

import java.io.Serializable;
import java.util.*;
import java.util.Vector;

import Math.*;
import NeuralNetworks.Layer.RecurrentLayer;
import NeuralNetworks.Neuron.RecurrentNeuron;

public class RecurrentNeuralNetwork extends Net<java.util.Vector<String>> implements Serializable {

    public RecurrentNeuralNetwork(int seed, LinkedList<Integer> hiddenLayers, VectorizedInstanceList instanceList, Activation activation) {
        super(seed, activation, instanceList, hiddenLayers, NetworkType.RECURRENTNEURALNETWORK);
    }

    public RecurrentNeuralNetwork(int seed, LinkedList<Integer> hiddenLayers, VectorizedInstanceList instanceList, Activation activation, NetworkType type) {
        super(seed, activation, instanceList, hiddenLayers, type);
    }

    private void createInputVector(java.util.Vector<String> inputLayer) {
        for (int i = 0; i < layers[0].size(); i++) {
            layers[0].getNeuron(i).setValue(Double.parseDouble(inputLayer.get(i)));
        }
    }

    protected void setOldValues() {
        for (int k = 1; k < layers.length - 1; k++) {
            ((RecurrentLayer) layers[k]).setValues();
        }
    }

    protected void setOldValuesToZero() {
        for (int k = 1; k < layers.length - 1; k++) {
            ((RecurrentLayer) layers[k]).setValuesToZero();
        }
    }

    @Override
    protected void feedForward() throws VectorSizeMismatch {
        for (int i = 0; i < layers.length - 1; i++) {
            for (int j = 0; j < layers[i + 1].size(); j++) {
                double sum = 0.0;
                for (int k = 0; k < layers[i].size(); k++) {
                    sum += layers[i].getNeuron(k).getWeight(j) * layers[i].getNeuron(k).getValue();
                }
                sum += biases[i].getValue(j);
                if (i + 1 != layers.length - 1) {
                    for (int k = 0; k < layers[i + 1].size(); k++) {
                        sum += ((RecurrentNeuron) layers[i + 1].getNeuron(k)).getRecurrentWeight(j) * ((RecurrentNeuron) layers[i + 1].getNeuron(k)).getOldValue();
                    }
                    sum = function.calculateForward(sum);
                }
                layers[i + 1].getNeuron(j).setValue(sum);
            }
        }
        if (layers[layers.length - 1].size() > 2) {
            layers[layers.length - 1].softmax();
        }
    }

    @Override
    protected void setWeights(LinkedList<Matrix> deltaWeights, LinkedList<Matrix> oldDeltaWeights, double momentum) {
        for (int t = 0; t < deltaWeights.size() - 1; t += 2) {
            Matrix weights = deltaWeights.get(t + 1);
            Matrix recurrentWeights = deltaWeights.get(t);
            for (int i = 0; i < weights.getRow(); i++) {
                for (int j = 0; j < weights.getColumn(); j++) {
                    if (!oldDeltaWeights.isEmpty()) {
                        weights.addValue(i, j, momentum * oldDeltaWeights.get(t + 1).getValue(i, j));
                    }
                    if (j > 0) {
                        layers[t / 2].getNeuron(j - 1).addWeight(i, weights.getValue(i, j));
                    } else {
                        biases[t / 2].addWeight(i, weights.getValue(i, j));
                    }
                }
            }
            for (int i = 0; i < recurrentWeights.getRow(); i++) {
                for (int j = 0; j < recurrentWeights.getColumn(); j++) {
                    if (!oldDeltaWeights.isEmpty()) {
                        recurrentWeights.addValue(i, j, momentum * oldDeltaWeights.get(t).getValue(i, j));
                    }
                    ((RecurrentNeuron)layers[(t / 2) + 1].getNeuron(j)).addRecurrentWeight(i, recurrentWeights.getValue(i, j));
                }
            }
        }
        Matrix weights = deltaWeights.getLast();
        for (int i = 0; i < weights.getRow(); i++) {
            for (int j = 0; j < weights.getColumn(); j++) {
                if (!oldDeltaWeights.isEmpty()) {
                    weights.addValue(i, j, momentum * oldDeltaWeights.getLast().getValue(i, j));
                }
                if (j > 0) {
                    layers[layers.length - 2].getNeuron(j - 1).addWeight(i, weights.getValue(i, j));
                } else {
                    biases[layers.length - 2].addWeight(i, weights.getValue(i, j));
                }
            }
        }
    }

    @Override
    protected LinkedList<Matrix> backpropagation(int classInfo, double learningRate, double momentum, LinkedList<Matrix> oldDeltaWeights) throws MatrixRowColumnMismatch, MatrixDimensionMismatch {
        LinkedList<Matrix> deltaWeights = new LinkedList<>();
        calculateRMinusY(deltaWeights, classInfo, learningRate);
        for (int i = layers.length - 3; i > -1; i--) {
            Matrix currentError = calculateError(i, deltaWeights);
            deltaWeights.set(0, deltaWeights.getFirst().multiply(layers[i].neuronsToMatrix()));
            deltaWeights.addFirst(currentError.multiply(((RecurrentLayer)layers[i + 1]).oldNeuronsToMatrix()));
            if (i > 0) {
                deltaWeights.addFirst(currentError);
            }
        }
        setWeights(deltaWeights, oldDeltaWeights, momentum);
        return deltaWeights;
    }

    @Override
    public void train(int epoch, double learningRate, double etaDecrease, double momentum) throws MatrixRowColumnMismatch, MatrixDimensionMismatch, VectorSizeMismatch {
        LinkedList<Matrix> oldDeltaWeights = new LinkedList<>();
        for (int i = 0; i < epoch; i++) {
            instanceList.shuffle(seed);
            for (int j = 0; j < instanceList.size(); j++) {
                Instance<Vector<String>> instance = instanceList.getInstance(j);
                for (int k = 0; k < instance.size(); k += 2) {
                    createInputVector(instance.get(k));
                    String classInfo = instance.get(k + 1).get(0);
                    feedForward();
                    oldDeltaWeights = backpropagation(instanceList.get(classInfo), learningRate, momentum, oldDeltaWeights);
                    setOldValues();
                }
                setOldValuesToZero();
            }
            learningRate *= etaDecrease;
        }
    }

    public LinkedList<String> predict(Instance<java.util.Vector<String>> instance) throws VectorSizeMismatch {
        setOldValuesToZero();
        LinkedList<String> classes = new LinkedList<>();
        for (int i = 0; i < instance.size(); i += 2) {
            createInputVector(instance.get(i));
            feedForward();
            if (instanceList.getOutput() == 1) {
                double outputValue = layers[layers.length - 1].getNeuron(0).getValue();
                if (outputValue >= 0.5) {
                    classes.add(instanceList.get(1));
                }
                classes.add(instanceList.get(0));
            } else {
                double bestValue = Integer.MIN_VALUE;
                int bestNeuron = -1;
                for (int j = 0; j < layers[layers.length - 1].size(); j++) {
                    if (layers[layers.length - 1].getNeuron(j).getValue() > bestValue) {
                        bestValue = layers[layers.length - 1].getNeuron(j).getValue();
                        bestNeuron = j;
                    }
                }
                classes.add(instanceList.get(bestNeuron));
            }
            setOldValues();
        }
        return classes;
    }

    public double test(VectorizedInstanceList list) throws VectorSizeMismatch {
        int correct = 0;
        int total = 0;
        for (int i = 0; i < list.size(); i++) {
            Instance<java.util.Vector<String>> instance = list.getInstance(i);
            LinkedList<String> predictedList = predict(instance);
            LinkedList<String> actualList = list.collectClassInfos(instance);
            for (int j = 0; j < actualList.size(); j++) {
                if (actualList.get(j).equals(predictedList.get(j))) {
                    correct++;
                }
                total++;
            }
        }
        return correct * 100.00 / total;
    }
}
