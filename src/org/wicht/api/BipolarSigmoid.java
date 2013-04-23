package org.wicht.api;

public class BipolarSigmoid implements NeuronFunction {
    @Override
    public double activate(double x) {
        return -1.0 + 2.0 / (1.0 + Math.exp(-x));
    }
}