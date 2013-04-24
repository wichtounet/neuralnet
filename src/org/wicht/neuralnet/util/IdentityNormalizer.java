package org.wicht.neuralnet.util;

import java.util.List;

public class IdentityNormalizer implements Normalizer {
    @Override
    public double normalize(double x) {
        return x;
    }

    @Override
    public double denormalize(double x) {
        return x;
    }

    @Override
    public List<Double> normalize(List<Double> inputs) {
        return inputs;
    }

    @Override
    public List<Double> denormalize(List<Double> inputs) {
        return inputs;
    }

    @Override
    public List<List<Double>> normalize2D(List<List<Double>> inputs) {
        return inputs;
    }
}