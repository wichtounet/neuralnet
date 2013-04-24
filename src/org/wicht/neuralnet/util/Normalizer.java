package org.wicht.neuralnet.util;

public class Normalizer {
    public static final double LOW = -1;
    public static final double HIGH = 1;

    private final double low;
    private final double high;

    public Normalizer(double low, double high) {
        this.low = low;
        this.high = high;
    }

    public double normalize(double x) {
        return (x - low) / (high - low) * (HIGH - LOW) + LOW;
    }

    public double denormalize(double x) {
        return ((low - high) * x - HIGH * low + high * LOW) / (LOW - HIGH);
    }
}