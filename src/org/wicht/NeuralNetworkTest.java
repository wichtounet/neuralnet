package org.wicht;

import org.wicht.api.NeuralNetwork;
import org.wicht.api.Sigmoid;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class NeuralNetworkTest {
    private NeuralNetworkTest() {
        throw new AssertionError();
    }

    public static void main(String[] args) {
        xorTest();
    }

    private static void xorTest() {
        System.out.println("Test the neural network for XOR function");

        List<List<Double>> inputs = new ArrayList<>();

        inputs.add(Arrays.asList(1.0, 1.0));
        inputs.add(Arrays.asList(1.0, 0.0));
        inputs.add(Arrays.asList(0.0, 1.0));
        inputs.add(Arrays.asList(0.0, 0.0));

        List<List<Double>> outputs = new ArrayList<>();

        outputs.add(Collections.singletonList(0.0));
        outputs.add(Collections.singletonList(1.0));
        outputs.add(Collections.singletonList(1.0));
        outputs.add(Collections.singletonList(0.0));

        NeuralNetwork network = new NeuralNetwork();
        network.setFunction(new Sigmoid());
        network.build(2, 24, 500, 64, 1);
        network.train(inputs, outputs, 50000, 0.001);

        for (List<Double> input : inputs) {
            List<Double> output = network.activate(input);

            System.out.println("Input: " + input);
            System.out.println("Output: " + output);
        }
    }
}