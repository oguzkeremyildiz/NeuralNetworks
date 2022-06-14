package NeuralNetworks.Layer;

import NeuralNetworks.Initializer.InitializerFunction;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.Random;
import Math.*;

public class LSTMLayer extends RecurrentLayer implements Serializable {

    private final LinkedList<Vector> vectors;
    private final double[][][] lstmWeights;
    private final double[][][] recurrentLstmWeights;
    private final double[] contextValues;
    private final double[] oldContextValues;

    public LSTMLayer(int s, LinkedList<Integer> hiddenLayers, int seed, boolean isHidden, InitializerFunction function) {
        super(s, hiddenLayers, seed, function);
        this.vectors = new LinkedList<>();
        for (int i = 0; i < 5; i++) {
            vectors.add(new Vector(size, 0));
        }
        Random random = new Random(seed);
        this.lstmWeights = new double[3][size][nextSize];
        this.recurrentLstmWeights = new double[3][size][size];
        this.contextValues = new double[size];
        this.oldContextValues = new double[size];
        for (int i = 0; i < lstmWeights.length; i++) {
            for (int j = 0; j < lstmWeights[i].length; j++) {
                for (int k = 0; k < lstmWeights[i][j].length; k++) {
                    this.lstmWeights[i][j][k] = function.calculate(random);
                }
                if (isHidden) {
                    for (int k = 0; k < recurrentLstmWeights[i][j].length; k++) {
                        this.recurrentLstmWeights[i][j][k] = function.calculate(random);
                    }
                }
            }
        }
    }

    public void setContextValuesToZero() {
        for (int i = 0; i < contextValues.length; i++) {
            contextValues[i] = 0.0;
            oldContextValues[i] = 0.0;
        }
    }

    public void setContextValues() {
        for (int i = 0; i < contextValues.length; i++) {
            oldContextValues[i] = contextValues[i];
        }
    }

    public Vector oldContextValuesToVector() {
        Vector vector = new Vector(size, 0);
        for (int i = 0; i < size; i++) {
            vector.addValue(i, oldContextValues[i]);
        }
        return vector;
    }

    public void clear() {
        for (Vector vector : this.vectors) {
            vector.clear();
        }
    }

    public void setContextValue(int index, double value) {
        contextValues[index] = value;
    }

    public double getContextValue(int index) {
        return contextValues[index];
    }

    public Vector getVector(int index) {
        return this.vectors.get(index);
    }

    public double getGateRecurrentWeight(int gate, int neuronIndex, int weightIndex) {
        return recurrentLstmWeights[gate][neuronIndex][weightIndex];
    }

    public double getGateWeight(int gate, int neuronIndex, int weightIndex) {
        return lstmWeights[gate][neuronIndex][weightIndex];
    }

    public void addGateWeight(int gate, int neuronIndex, int weightIndex, double weight) {
        lstmWeights[gate][neuronIndex][weightIndex] += weight;
    }

    public void addGateRecurrentWeight(int gate, int neuronIndex, int weightIndex, double weight) {
        recurrentLstmWeights[gate][neuronIndex][weightIndex] += weight;
    }

    public Matrix forgetGateWeightsToMatrix() {
        Matrix weights = new Matrix(size, nextSize);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weights.setValue(i, j, lstmWeights[0][i][j]);
            }
        }
        return weights;
    }

    public Matrix addGateWeightsToMatrix() {
        Matrix weights = new Matrix(size, nextSize);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weights.setValue(i, j, lstmWeights[1][i][j]);
            }
        }
        return weights;
    }

    public Matrix gGateWeightsToMatrix() {
        Matrix weights = new Matrix(size, nextSize);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < nextSize; j++) {
                weights.setValue(i, j, lstmWeights[2][i][j]);
            }
        }
        return weights;
    }
}
