package org.wicht.neuralnet.util;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractNormalizer implements Normalizer {
    @Override
    public List<Double> normalize(List<Double> inputs) {
        List<Double> normalizedInputs = new ArrayList<>();

        for (Double d : inputs) {
            normalizedInputs.add(normalize(d));
        }

        return normalizedInputs;
    }

    @Override
    public List<Double> denormalize(List<Double> inputs) {
        List<Double> normalizedInputs = new ArrayList<>();

        for (Double d : inputs) {
            normalizedInputs.add(denormalize(d));
        }

        return normalizedInputs;
    }

    @Override
    public List<List<Double>> normalize2D(List<List<Double>> inputs) {
        List<List<Double>> normalized = new ArrayList<>();

        for (List<Double> list : inputs) {
            normalized.add(normalize(list));
        }

        return normalized;
    }
}