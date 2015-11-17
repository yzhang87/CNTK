class RollingWindowRandomizer : IRandomizer {
    RollingWindowRandomizer(ISource& source, int rollingWindowSize, int rank, int numWorkers);
}
