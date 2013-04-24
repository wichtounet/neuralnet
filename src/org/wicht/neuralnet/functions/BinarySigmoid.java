package org.wicht.neuralnet.functions;

public class BinarySigmoid implements ActivationFunction {
    @Override
    public double activate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}