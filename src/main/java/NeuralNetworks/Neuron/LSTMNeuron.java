package NeuralNetworks.Neuron;

import java.io.Serializable;

public class LSTMNeuron extends RecurrentNeuron implements Serializable {

    private final double[] forgetGate;
    private final double[] forgetGateRecurrent;
    private final double[] addGate;
    private final double[] addGateRecurrent;
    private final double[] gGate;
    private final double[] gGateRecurrent;
    private double contextValue;

    public LSTMNeuron(double[] weights, double[] recurrentWeights, double[] forgetGate, double[] forgetGateRecurrent, double[] addGate, double[] addGateRecurrent, double[] gGate, double[] gGateRecurrent) {
        super(weights, recurrentWeights);
        this.forgetGate = forgetGate;
        this.forgetGateRecurrent = forgetGateRecurrent;
        this.addGate = addGate;
        this.addGateRecurrent = addGateRecurrent;
        this.gGate = gGate;
        this.gGateRecurrent = gGateRecurrent;
        this.contextValue = 0.0;
    }

    public LSTMNeuron(double[] weights, double[] forgetGate, double[] addGate, double[] gGate) {
        super(weights);
        this.forgetGate = forgetGate;
        this.forgetGateRecurrent = null;
        this.addGate = addGate;
        this.addGateRecurrent = null;
        this.gGate = gGate;
        this.gGateRecurrent = null;
        this.contextValue = 0.0;
    }

    public double getForgetGateWeight(int i) {
        return forgetGate[i];
    }

    public double getForgetGateRecurrentWeight(int i) {
        return forgetGateRecurrent[i];
    }

    public double getAddGateWeight(int i) {
        return addGate[i];
    }

    public double getAddGateRecurrentWeight(int i) {
        return addGateRecurrent[i];
    }

    public double getGGateWeight(int i) {
        return gGate[i];
    }

    public double getGGateRecurrentWeight(int i) {
        return gGateRecurrent[i];
    }

    public double getContextValue() {
        return contextValue;
    }

    public void setContextValue(double contextValue) {
        this.contextValue = contextValue;
    }
}
