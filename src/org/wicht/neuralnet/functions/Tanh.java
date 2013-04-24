package org.wicht.neuralnet.functions;

public class Tanh implements ActivationFunction {
    @Override
    public double activate(double x) {
        double local = Math.tanh(x);
        return local;
    }
}