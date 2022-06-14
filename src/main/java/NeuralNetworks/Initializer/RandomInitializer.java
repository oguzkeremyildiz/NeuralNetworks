package NeuralNetworks.Initializer;

import java.util.Random;

public class RandomInitializer implements InitializerFunction {
    @Override
    public double calculate(Random random) {
        return 2 * random.nextDouble() - 1;
    }
}
