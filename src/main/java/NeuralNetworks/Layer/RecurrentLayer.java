package NeuralNetworks.Layer;

import java.io.Serializable;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;
import Math.*;
import NeuralNetworks.Initializer.InitializerFunction;

public class RecurrentLayer extends Layer implements Serializable {

    protected double[][] recurrentWeights;
    private final double[] oldValues;

    public RecurrentLayer(int s, LinkedList<Integer> hiddenLayers, int seed, InitializerFunction function) {
        super(s, hiddenLayers, seed, function);
        Random random = new Random(seed);
        this.oldValues = new double[size];
        this.recurrentWeights = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int k = 0; k < size; k++) {
                recurrentWeights[i][k] = function.calculate(random);
            }
        }
    }

    public double getRecurrentWeight(int neuronIndex, int weightIndex) {
        return recurrentWeights[neuronIndex][weightIndex];
    }

    public void addRecurrentWeight(int neuronIndex, int weightIndex, double weight) {
        recurrentWeights[neuronIndex][weightIndex] += weight;
    }

    public double getOldValue(int i) {
        return oldValues[i];
    }

    public void setValuesToZero() {
        Arrays.fill(oldValues, 0.0);
    }

    public void setValues() {
        for (int i = 0; i < oldValues.length; i++) {
            oldValues[i] = values[i];
        }
    }

    public Matrix oldNeuronsToMatrix() {
        Matrix neurons = new Matrix(1, size);
        for (int i = 0; i < size; i++) {
            neurons.setValue(0, i, oldValues[i]);
        }
        return neurons;
    }
}
