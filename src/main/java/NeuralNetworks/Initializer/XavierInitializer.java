package NeuralNetworks.Initializer;

import java.util.Random;

public class XavierInitializer implements InitializerFunction {

    private final int n;

    public XavierInitializer(int n) {
        this.n = n;
    }

    @Override
    public double calculate(Random random) {
        return (2 * random.nextDouble() - 1) / Math.sqrt(n);
    }
}
