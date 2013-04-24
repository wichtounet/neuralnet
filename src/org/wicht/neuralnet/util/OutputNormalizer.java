package org.wicht.neuralnet.util;

public class OutputNormalizer extends AbstractNormalizer {
    private final double LOW = 0;
    private final double HIGH = 1;

    private final double low;
    private final double high;

    public OutputNormalizer(double low, double high) {
        this.low = low;
        this.high = high;
    }

    @Override
    public double normalize(double x) {
        return (x - low) / (high - low) * (HIGH - LOW) + LOW;
    }

    @Override
    public double denormalize(double x) {
        return ((low - high) * x - HIGH * low + high * LOW) / (LOW - HIGH);
    }
}