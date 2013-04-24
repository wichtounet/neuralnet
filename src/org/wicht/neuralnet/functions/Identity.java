package org.wicht.neuralnet.functions;

public class Identity implements ActivationFunction {
    @Override
    public double activate(double x) {
        return x;
    }
}