import ActivationFunctions.Activation;
import ActivationFunctions.ActivationFunction;
import ActivationFunctions.ReLU;
import ActivationFunctions.Sigmoid;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.LinkedList;
import java.util.Scanner;

public class NeuralNetwork implements Serializable {

    private final Layer[] layers;
    private final InstanceList instanceList;
    private final int seed;
    private final ActivationFunction function;
    private final Bias[] biases;

    public NeuralNetwork(Scanner source, InstanceList instanceList) {
        this.instanceList = instanceList;
        int layerSize = Integer.parseInt(source.nextLine());
        this.layers = new Layer[layerSize];
        for (int i = 0; i < layers.length; i++) {
            int neuronSize = Integer.parseInt(source.nextLine());
            layers[i] = new Layer(neuronSize);
            for (int j = 0; j < neuronSize; j++) {
                String[] neuron = source.nextLine().split(" ");
                layers[i].getNeuron(j).setValue(Double.parseDouble(neuron[0]));
                if (neuron.length > 1) {
                    layers[i].getNeuron(j).initializeWeight(neuron.length - 1);
                    for (int k = 1; k < neuron.length; k++) {
                        layers[i].getNeuron(j).setWeight(k - 1, Double.parseDouble(neuron[k]));
                    }
                }
            }
        }
        int biasSize = Integer.parseInt(source.nextLine());
        biases = new Bias[biasSize];
        for (int i = 0; i < biases.length; i++) {
            String[] bias = source.nextLine().split(" ");
            biases[i] = new Bias(bias.length);
            for (int j = 0; j < bias.length; j++) {
                biases[i].setWeight(j, Double.parseDouble(bias[j]));
            }
        }
        this.seed = Integer.parseInt(source.nextLine());
        String activation = source.nextLine();
        switch (activation) {
            case "SIGMOID":
                this.function = new Sigmoid();
                break;
            case "RELU":
                this.function = new ReLU();
                break;
            default:
                this.function = null;
                break;
        }
    }

    public NeuralNetwork(int seed, LinkedList<Integer> hiddenLayers, InstanceList instanceList, Activation activation) {
        ActivationFunction function;
        switch (activation) {
            case SIGMOID:
                function = new Sigmoid();
                break;
            case RELU:
                function = new ReLU();
                break;
            default:
                function = null;
                break;
        }
        this.function = function;
        this.seed = seed;
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

    private void feedForward() {
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

    private void createInputVector(Instance instance) {
        for (int i = 0; i < layers[0].size(); i++) {
            layers[0].getNeuron(i).setValue(Double.parseDouble(instance.get(i)));
        }
    }

    private double[][] calculateError(int i, LinkedList<double[][]> deltaWeights) {
        deltaWeights.set(0, multiplyMatrices(layers[i + 1].weightsToMatrix(), deltaWeights.getFirst()));
        deltaWeights.set(0, hadamardProduct(deltaWeights.getFirst(), function.calculateBack(layers[i + 1].neuronsToVector())));
        return deltaWeights.getFirst();
    }

    private double[][] hadamardProduct(double[][] matrix1, double[][] matrix2) {
        double[][] matrix = new double[matrix1.length][matrix1[0].length];
        for (int i = 0; i < matrix1.length; i++) {
            for (int j = 0; j < matrix1[i].length; j++) {
                matrix[i][j] = matrix1[i][j] * matrix2[i][j];
            }
        }
        return matrix;
    }

    private double dotProduct(double[] vector, double[][] matrix, int j) {
        double sum = 0.0;
        for (int i = 0; i < vector.length; i++) {
            sum += vector[i] * matrix[i][j];
        }
        return sum;
    }

    private double[][] multiplyMatrices(double[][] matrix1, double[][] matrix2) {
        double[][] matrix = new double[matrix1.length][matrix2[0].length];
        for (int j = 0; j < matrix[0].length; j++) {
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][j] = dotProduct(matrix1[i], matrix2, j);
            }
        }
        return matrix;
    }

    private void setWeights(LinkedList<double[][]> deltaWeights) {
        for (int t = 0; t < deltaWeights.size(); t++) {
            double[][] weights = deltaWeights.get(t);
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    if (j > 0) {
                        layers[t].getNeuron(j - 1).addWeight(i, weights[i][j]);
                    } else {
                        biases[t].addWeight(i, weights[i][j]);
                    }
                }
            }
        }
    }

    private void calculateRMinusY(LinkedList<double[][]> deltaWeights, int classInfo, double learningRate) {
        double[][] matrix = new double[layers[layers.length - 1].size()][1];
        if (matrix.length > 1) {
            for (int j = 0; j < matrix.length; j++) {
                if (classInfo == j) {
                    matrix[j][0] = learningRate * (1 - layers[layers.length - 1].getNeuron(j).getValue());
                } else {
                    matrix[j][0] = learningRate * -layers[layers.length - 1].getNeuron(j).getValue();
                }
            }
        } else {
            matrix[0][0] = learningRate * (classInfo - layers[layers.length - 1].getNeuron(0).getValue());
        }
        deltaWeights.addFirst(matrix);
        deltaWeights.set(0, multiplyMatrices(deltaWeights.getFirst(), layers[layers.length - 2].neuronsToMatrix()));
        if (layers.length > 2) {
            deltaWeights.addFirst(matrix);
        }
    }

    private void backpropagation(int classInfo, double learningRate) {
        LinkedList<double[][]> deltaWeights = new LinkedList<>();
        calculateRMinusY(deltaWeights, classInfo, learningRate);
        for (int i = layers.length - 3; i > -1; i--) {
            double[][] currentError = calculateError(i, deltaWeights);
            deltaWeights.set(0, multiplyMatrices(deltaWeights.getFirst(), layers[i].neuronsToMatrix()));
            if (i > 0) {
                deltaWeights.addFirst(currentError);
            }
        }
        setWeights(deltaWeights);
    }

    public void train(int epoch, double learningRate, double etaDecrease) {
        for (int i = 0; i < epoch; i++) {
            instanceList.shuffle(seed);
            for (int j = 0; j < instanceList.size(); j++) {
                createInputVector(instanceList.getInstance(j));
                feedForward();
                backpropagation(instanceList.get(instanceList.getInstance(j).getLast()), learningRate);
            }
            learningRate *= etaDecrease;
        }
    }

    public String predict(Instance instance) {
        createInputVector(instance);
        feedForward();
        if (instanceList.getOutput() == 1) {
            double outputValue = layers[layers.length - 1].getNeuron(0).getValue();
            if (outputValue > 0.5) {
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
            Instance instance = list.getInstance(i);
            if (instance.getLast().equals(predict(instance))) {
                correct++;
            }
            total++;
        }
        return correct * 100.00 / total;
    }

    public void saveAsText(String fileName) {
        try {
            BufferedWriter fw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fileName), StandardCharsets.UTF_8));
            fw.write(Integer.toString(layers.length));
            fw.newLine();
            for (Layer layer : layers) {
                fw.write(Integer.toString(layer.size()));
                fw.newLine();
                for (int j = 0; j < layer.size(); j++) {
                    fw.write(layer.getNeuron(j).toString());
                    fw.newLine();
                }
            }
            fw.write(Integer.toString(biases.length));
            fw.newLine();
            for (Bias bias : biases) {
                fw.write(bias.toString());
                fw.newLine();
            }
            fw.write(Integer.toString(seed));
            fw.newLine();
            fw.write(function.toString());
            fw.close();
        } catch (IOException ignored) {}
    }

    public void saveAsBin(String fileName) {
        try {
            FileOutputStream outFile = new FileOutputStream(fileName);
            ObjectOutputStream outObject = new ObjectOutputStream(outFile);
            outObject.writeObject(this);
        } catch (IOException var5) {
            var5.printStackTrace();
        }
    }
}