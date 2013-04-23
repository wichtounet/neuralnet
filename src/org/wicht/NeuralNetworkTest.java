package org.wicht;

import org.wicht.api.BinarySigmoid;
import org.wicht.api.NeuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public final class NeuralNetworkTest {
    private NeuralNetworkTest() {
        throw new AssertionError();
    }

    public static void main(String[] args) {
        sqrtTest();
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
        network.setFunction(new BinarySigmoid());
        network.build(2, 4, 1);
        network.train(inputs, outputs, 50000, 0.001);

        for (List<Double> input : inputs) {
            List<Double> output = network.activate(input);

            System.out.println("Input: " + input);
            System.out.println("Output: " + output);
        }
    }

    private static void sqrtTest() {
        System.out.println("Test the neural network for sqrt function");

        List<List<Double>> inputs = new ArrayList<>();

        for (int i = 0; i < 100; ++i) {
            inputs.add(Arrays.asList(i / 100.0));
        }

        List<List<Double>> outputs = new ArrayList<>();

        for (int i = 0; i < 100; ++i) {
            outputs.add(Arrays.asList(Math.sqrt(i) / 10.0));
        }

        //System.out.println(outputs);

        NeuralNetwork network = new NeuralNetwork();
        network.setFunction(new BinarySigmoid());
        network.build(1, 10, 1);
        network.train(inputs, outputs, 50000, 0.01);

        for (List<Double> input : inputs) {
            List<Double> output = network.activate(input);

            System.out.println("sqrt(" + (input.get(0) * 100.0) + ") = " + (output.get(0) * 10.0) + ", error = " + (Math.sqrt(input.get(0)) / 10.0 - output.get(0)));
        }
    }
}