package org.wicht.api;

public class Sigmoid implements NeuronFunction {
    @Override
    public double activate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}