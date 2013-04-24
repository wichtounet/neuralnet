package org.wicht.neuralnet;

import com.sun.istack.internal.NotNull;
import org.wicht.neuralnet.functions.ActivationFunction;
import org.wicht.neuralnet.util.IdentityNormalizer;
import org.wicht.neuralnet.util.InputNormalizer;
import org.wicht.neuralnet.util.Normalizer;
import org.wicht.neuralnet.util.OutputNormalizer;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final List<List<AbstractNeuron>> layers = new ArrayList<>();

    private final InputNeuron bias = new InputNeuron();

    private static final double epsilon = 0.00000000001;
    private static final double learningRate = 0.9f;
    private static final double momentum = 0.3f;

    private ActivationFunction[] functions;

    @NotNull
    private Normalizer inputNormalizer = new IdentityNormalizer();

    @NotNull
    private Normalizer outputNormalizer = new IdentityNormalizer();

    public void setFunctions(ActivationFunction... functions) {
        this.functions = functions;
    }

    public void setInputRange(double low, double high) {
        inputNormalizer = new InputNormalizer(low, high);
    }

    public void setOutputRange(double low, double high) {
        outputNormalizer = new OutputNormalizer(low, high);
    }

    public void build(int i, int... sizes) {
        //0. Init the bias neuron

        bias.setOutput(1);

        //1. Build the intput layer

        layers.add(new ArrayList<AbstractNeuron>(i));

        for (int n = 0; n < i; ++n) {
            getInputLayer().add(new InputNeuron());
        }

        //2. Build the hidden and output layers

        for (int k = 0; k < sizes.length; ++k) {
            int size = sizes[k];
            List<AbstractNeuron> layer = new ArrayList<>(size);

            for (int n = 0; n < size; ++n) {
                NormalNeuron neuron = new NormalNeuron();
                neuron.setFunction(functions[k]);

                for (AbstractNeuron inputNeuron : layers.get(k)) {
                    Synapse synapse = new Synapse(inputNeuron, neuron);

                    inputNeuron.addOutputSynapse(synapse);
                    neuron.addInputSynapse(synapse);
                }

                neuron.addInputSynapse(new Synapse(bias, neuron));

                layer.add(neuron);
            }

            layers.add(layer);
        }
    }

    private List<AbstractNeuron> getInputLayer() {
        return layers.get(0);
    }

    private List<AbstractNeuron> getOutputLayer() {
        return layers.get(layers.size() - 1);
    }

    public void train(List<List<Double>> inputs, List<List<Double>> expected, int maxIterations, double maxError) {
        System.out.println("Start training");

        //Normalize if necessary
        List<List<Double>> normalizedInputs = inputNormalizer.normalize2D(inputs);
        List<List<Double>> normalizedExpected = outputNormalizer.normalize2D(expected);

        int iteration;

        double error = 1;
        for (iteration = 0; iteration < maxIterations && error > maxError; ++iteration) {
            error = 0;

            for (int p = 0; p < normalizedInputs.size(); ++p) {
                List<Double> output = activateImpl(normalizedInputs.get(p));

                for (int j = 0; j < normalizedExpected.get(p).size(); ++j) {
                    error += Math.pow(output.get(j) - normalizedExpected.get(p).get(j), 2);
                }

                backPropagate(normalizedExpected.get(p));
            }
        }

        if (iteration == maxIterations) {
            System.out.println("Failed to reach precision");
        }

        System.out.println("Sum of squared errors = " + error);
        System.out.println("##### EPOCH " + iteration);
        System.out.println();
    }

    private void backPropagate(List<Double> expected) {
        for (int i = 0; i < expected.size(); ++i) {
            double d = expected.get(i);

            if (d < 0) {
                expected.set(i, 0 + epsilon);
            } else if (d > 1) {
                expected.set(i, 1 - epsilon);
            }
        }

        //1. Train the output layer

        for (int i = 0; i < expected.size(); ++i) {
            Neuron neuron = getOutputLayer().get(i);

            double ak = neuron.getOutput();
            double desired = expected.get(i);

            for (Synapse synapse : neuron.getInConnections()) {
                double ai = synapse.getInputNeuron().getOutput();

                double partialDerivative = -1 * (desired - ak) * ak * (1 - ak) * ai;
                double delta = -learningRate * partialDerivative;
                double newWeight = synapse.getWeight() + delta;

                synapse.setDeltaWeight(delta);
                synapse.setWeight(newWeight + momentum * synapse.getPrevDeltaWeight());
            }
        }

        //2. Train the hidden layers from right to left

        for (int l = layers.size() - 2; l > 0; --l) {
            for (Neuron neuron : layers.get(l)) {
                double aj = neuron.getOutput();

                double outputDerivativeSum = 0;
                for (Synapse outSynapse : neuron.getOutConnections()) {
                    outputDerivativeSum += outSynapse.getDeltaWeight() * outSynapse.getWeight();
                }

                for (Synapse inSynapse : neuron.getInConnections()) {
                    double ai = inSynapse.getInputNeuron().getOutput();

                    double partialDerivative = (1 - aj) * ai * outputDerivativeSum;
                    double deltaWeight = 1 * partialDerivative;
                    double newWeight = inSynapse.getWeight() + deltaWeight;

                    inSynapse.setDeltaWeight(deltaWeight);
                    inSynapse.setWeight(newWeight + momentum * inSynapse.getPrevDeltaWeight());
                }
            }
        }
    }

    /**
     * Activate the neural network on the given inputs and return its output.
     * <p/>
     * The inputs must be of the normalized range or the range indicating via setInputRange.
     * The outputs must of the normalized range or the range indicated via setOutputRange.
     *
     * @param inputs The inputs.
     * @return The outputs.
     */
    public List<Double> activate(List<Double> inputs) {
        return outputNormalizer.denormalize(activateImpl(inputNormalizer.normalize(inputs)));
    }

    /**
     * Activate the neural network on the given inputs and return its output.
     * <p/>
     * The inputs must be normalized and the output will be normalized.
     *
     * @param inputs The normalized inputs.
     * @return The normalized outputs.
     */
    private List<Double> activateImpl(List<Double> inputs) {
        //1. Set inputs in the leftmost layer

        for (int i = 0; i < getInputLayer().size(); ++i) {
            getInputLayer().get(i).setOutput(inputs.get(i));
        }

        //2. Activate each neuron from left to right

        for (List<AbstractNeuron> layer : layers) {
            for (Neuron neuron : layer) {
                neuron.activate();
            }
        }

        //3. Collect the results

        List<Double> results = new ArrayList<>();

        for (Neuron neuron : getOutputLayer()) {
            results.add(neuron.getOutput());
        }

        return results;
    }
}